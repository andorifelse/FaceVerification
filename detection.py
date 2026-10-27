"""YuNet face detection and five-landmark similarity alignment."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from config import ROOT
from verification import open_image


class FaceDetector:
    def __init__(self, model=None, confidence=0.8):
        model = Path(model or ROOT/'assets/face_detection_yunet_2023mar.onnx')
        if not model.is_file():
            raise FileNotFoundError(f'缺少人脸检测模型: {model}')
        self.net = cv2.FaceDetectorYN.create(str(model),'',(320,320),confidence,0.3,5000)

    def detect(self, image):
        image = open_image(image)
        width,height = image.size
        scale = min(1.,960/max(width,height))
        small = image.resize((max(32,int(width*scale)),max(32,int(height*scale))))
        bgr = cv2.cvtColor(np.asarray(small),cv2.COLOR_RGB2BGR)
        self.net.setInputSize(small.size)
        _, faces = self.net.detect(bgr)
        result = []
        if faces is None:
            return result
        scale_xy = np.array([width/small.width,height/small.height],dtype=np.float32)
        rgb = np.asarray(image)
        for row in faces:
            xy = row[:2]*scale_xy; wh = row[2:4]*scale_xy
            x1,y1 = np.maximum(xy,0).astype(int)
            x2,y2 = np.minimum(xy+wh,[width,height]).astype(int)
            if x2-x1 < 12 or y2-y1 < 12:
                continue
            landmarks = row[4:14].reshape(5,2)*scale_xy
            # YuNet: right eye, left eye, nose, right mouth, left mouth.
            template = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                                 [41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)*(128/112)
            matrix = similarity_transform(landmarks,template)
            aligned = cv2.warpAffine(rgb,matrix,(128,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            result.append({'box':[int(x1),int(y1),int(x2),int(y2)],
                           'landmarks':landmarks.tolist(),'confidence':float(row[14]),
                           'face':Image.fromarray(aligned)})
        return result

    def single(self,image):
        faces = self.detect(image)
        if not faces:
            raise ValueError('未检测到清晰人脸，请调整照片或拍摄角度。')
        if len(faces) != 1:
            raise ValueError(f'检测到 {len(faces)} 张人脸，请使用只包含一个人的照片。')
        return faces[0]


def similarity_transform(source,target):
    """Least-squares scale + rotation + translation, without shear or reflection."""
    source,target = np.asarray(source,dtype=np.float64),np.asarray(target,dtype=np.float64)
    mean_s,mean_t = source.mean(0),target.mean(0)
    a,b = source-mean_s,target-mean_t
    variance = np.sum(a*a)
    if variance < 1e-8:
        raise ValueError('无效的人脸关键点。')
    u,s,vh = np.linalg.svd(a.T@b)
    sign = np.eye(2); sign[-1,-1] = np.linalg.det(u@vh)
    rotation = u@sign@vh
    scale = np.sum(s*np.diag(sign))/variance
    translation = mean_t - scale*mean_s@rotation
    return np.column_stack((scale*rotation.T,translation)).astype(np.float32)
