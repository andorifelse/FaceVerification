import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from config import ROOT
from detection import FaceDetector, similarity_transform
from metrics import read_pairs, roc_curve, summarize, threshold_search
from verification import FaceVerifier, load_backbone


def test_threshold_search_ties_and_extremes():
    for scores,labels in [([.5,.5,.8,.1],[1,0,1,0]),([.1,.9],[1,0]),([1.,1.],[0,1])]:
        acc,threshold = threshold_search(scores,labels)
        options = np.r_[np.nextafter(max(scores),np.inf),np.unique(scores)]
        brute = max(summarize(scores,labels,t)['balanced_accuracy'] for t in options)
        assert acc == pytest.approx(brute)
        assert summarize(scores,labels,threshold)['balanced_accuracy'] == pytest.approx(brute)
    with pytest.raises(ValueError):threshold_search([np.nan,0],[1,0])


def test_confusion_and_roc():
    result = summarize([.8,.6,.3,.1],[1,0,1,0],.5)
    assert (result['tp'],result['fp'],result['tn'],result['fn']) == (1,1,1,1)
    assert result['auc'] == pytest.approx(.75)
    assert summarize([.9,.8,.2,.1],[1,1,0,0],.5)['eer'] == pytest.approx(0)
    fpr,tpr = roc_curve([.2,.2],[1,0])
    assert list(fpr) == [0.,1.] and list(tpr) == [0.,1.]


def test_pair_formats_and_header(tmp_path):
    for name in ['Alice/Alice_0001.jpg','Alice/Alice_0002.jpg','Bob/Bob_0001.jpg']:
        file = tmp_path/name; file.parent.mkdir(exist_ok=True);Image.new('RGB',(16,16)).save(file)
    protocol = tmp_path/'pairs.txt'
    protocol.write_text('1\nAlice 1 2\nAlice 1 Bob 1\n')
    pairs,header = read_pairs(protocol,tmp_path)
    assert len(pairs) == 2 and header == [1] and [p[2] for p in pairs] == [1,0]
    protocol.write_text('Alice/Alice_0001.jpg Alice/Alice_0002.jpg 1\n')
    assert read_pairs(protocol,tmp_path)[0][0][2] == 1
    protocol.write_text('2\nAlice 1 2\n')
    with pytest.raises(ValueError):read_pairs(protocol,tmp_path)
    protocol.write_text('Alice 1 99\n')
    with pytest.raises(FileNotFoundError):read_pairs(protocol,tmp_path)


def test_similarity_transform():
    source = np.array([[1,2],[5,2],[3,4],[2,6],[4,6]],dtype=np.float32)
    rotation = np.array([[0.,-1.],[1.,0.]])
    target = 2*source@rotation+np.array([15,20])
    matrix = similarity_transform(source,target)
    assert np.allclose(np.c_[source,np.ones(5)]@matrix.T,target,atol=1e-5)


@pytest.fixture(scope='module')
def verifier():
    return FaceVerifier(ROOT/'checkpoints/18.pth',threshold=.5,tta=True,device='cpu')


def test_checkpoint_and_head_equivalence(verifier):
    with torch.inference_mode():
        image = torch.randn(2,1,128,128)
        expected = verifier.net(image)
        cached = verifier.net.bn(verifier.net.linear(verifier.net.forward_features(image)))
    assert torch.equal(expected,cached)
    assert expected.shape == (2,512)


def test_verification_symmetric_and_identical(verifier):
    image = ROOT/'samples/009.jpg'
    first = verifier.compare(image,image)
    assert first['same_person'] == 1
    assert first['similarity'] == pytest.approx(1.,abs=1e-5)
    second_image = Image.new('RGB',(128,128),(20,100,130))
    a = verifier.compare(image,second_image)
    b = verifier.compare(second_image,image)
    assert a['similarity'] == pytest.approx(b['similarity'],abs=1e-6)
    with pytest.raises(FileNotFoundError):verifier.compare('/nonexistent.jpg',image)


def test_calibration_hash_guard(tmp_path):
    file = tmp_path/'calibration.json'
    file.write_text(json.dumps({'threshold':.5,'tta':True,'checkpoint_sha256':'wrong'}))
    with pytest.raises(ValueError,match='不匹配'):
        FaceVerifier(ROOT/'checkpoints/18.pth',calibration=file,device='cpu')


def test_detector_no_face_single_and_multi():
    detector = FaceDetector()
    with pytest.raises(ValueError,match='未检测到'):
        detector.single(Image.new('RGB',(320,240),'white'))
    image = Image.open(ROOT/'samples/009.jpg').convert('RGB')
    result = detector.single(image)
    assert result['face'].size == (128,128) and result['confidence'] > .8
    mosaic = Image.new('RGB',(image.width*2,image.height))
    mosaic.paste(image,(0,0));mosaic.paste(image,(image.width,0))
    with pytest.raises(ValueError,match='张人脸'):
        detector.single(mosaic)


def test_web_api_upload_and_detection(verifier):
    from app import create_app
    client = create_app(verifier=verifier).test_client()
    page = client.get('/')
    assert page.status_code == 200 and '摄像头'.encode() in page.data
    assert client.post('/api/verify').status_code == 400
    def photo():return io.BytesIO((ROOT/'samples/009.jpg').read_bytes())
    for aligned in ['true','false']:
        response = client.post('/api/verify',data={'image1':(photo(),'a.jpg'),'image2':(photo(),'b.jpg'),'aligned':aligned})
        assert response.status_code == 200,response.json
        assert response.json['same_person'] == 1
        assert len(response.json['previews']) == 2
    bad = client.post('/api/verify',data={'image1':(io.BytesIO(b'broken'),'a.jpg'),'image2':(photo(),'b.jpg')})
    assert bad.status_code == 400


def test_required_entry_point():
    if not (ROOT/'artifacts/calibration.json').exists():
        pytest.skip('Full evaluation has not yet generated a calibration file')
    from FaceVerification import FaceVerification
    value = FaceVerification(str(ROOT/'samples/009.jpg'),str(ROOT/'samples/009.jpg'))
    assert type(value) is int and value == 1


def test_transformer_attention_matches_explicit_attention():
    from model.face_vit import Attention
    attention=Attention(32,4).eval()
    x=torch.randn(2,7,32)
    q,k,v=attention.qkv(x).reshape(2,7,3,4,8).permute(2,0,3,1,4).unbind(0)
    expected=((q@k.transpose(-2,-1))/(8**.5)).softmax(-1)@v
    expected=attention.proj(expected.transpose(1,2).reshape(2,7,32))
    assert torch.allclose(attention(x),expected,atol=1e-6)


def test_transformer_rgb_preprocessing():
    from verification import make_transform
    transform=make_transform({'backbone':'face_vit'})
    image=Image.new('RGB',(96,112),(255,0,0))
    tensor=transform(image)
    assert tensor.shape==(3,112,112)
    assert tensor[0].mean()==1 and tensor[1].mean()==-1 and tensor[2].mean()==-1


def test_transformer_freezing_and_gradients():
    from model.face_vit import FaceViT
    # Small architecture exercises the same code path without loading large weights.
    net=FaceViT(embedding_size=8,dim=16,depth=3,heads=4,ratio=2,image_size=16,patch_size=8)
    net.set_trainable_blocks(1);net.train()
    before=net.feature[1].running_mean.clone()
    net(torch.randn(2,3,16,16)).square().mean().backward()
    assert not net.patch_embed.proj.weight.requires_grad and net.patch_embed.proj.weight.grad is None
    assert net.blocks[0].attn.qkv.weight.grad is None
    assert net.blocks[-1].attn.qkv.weight.grad.abs().sum()>0
    assert net.feature[0].weight.grad.abs().sum()>0
    assert torch.equal(net.feature[1].running_mean,before)


def test_default_transformer_checkpoint_and_api():
    from config import config
    if not (config.checkpoints/config.test_model).exists():
        pytest.skip('Transformer training has not completed')
    verifier=FaceVerifier(device='cpu')
    assert verifier.metadata['backbone']=='face_vit'
    result=verifier.compare(ROOT/'samples/009.jpg',ROOT/'samples/009.jpg',detect=True)
    assert result['same_person']==1 and result['similarity']==pytest.approx(1,abs=1e-5)
    from app import create_app
    client=create_app(verifier=verifier).test_client()
    response=client.post('/api/verify',data={
        'image1':(io.BytesIO((ROOT/'samples/009.jpg').read_bytes()),'a.jpg'),
        'image2':(io.BytesIO((ROOT/'samples/009.jpg').read_bytes()),'b.jpg')})
    assert response.status_code==200 and response.json['same_person']==1
