"""Local upload / webcam face verification demo: python app.py."""
import argparse
import base64
from io import BytesIO
from threading import Lock

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, UnidentifiedImageError

from detection import FaceDetector
from verification import FaceVerifier


def image_data(image,box=None):
    image = image.copy()
    if box:
        ImageDraw.Draw(image).rectangle(box,outline='#1ecba2',width=max(2,image.width//150))
    image.thumbnail((640,640))
    stream = BytesIO(); image.save(stream,format='JPEG',quality=85)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')


def create_app(verifier=None,detector=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 12*1024*1024
    resources, lock = {'verifier':verifier,'detector':detector},Lock()

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.post('/api/verify')
    def verify():
        try:
            images = []
            for field in ['image1','image2']:
                if field not in request.files:
                    raise ValueError('请选择两张图片，或先拍摄两张照片。')
                with Image.open(request.files[field].stream) as image:
                    if image.width*image.height > 20_000_000:
                        raise ValueError('图片过大，请缩小到两千万像素以下。')
                    from PIL import ImageOps
                    images.append(ImageOps.exif_transpose(image).convert('RGB'))
            aligned = request.form.get('aligned','false') == 'true'
            with lock:
                if resources['verifier'] is None:
                    resources['verifier'] = FaceVerifier()
                if not aligned and resources['detector'] is None:
                    resources['detector'] = FaceDetector()
                result = resources['verifier'].compare(*images,detect=not aligned,detector=resources['detector'])
            result['previews'] = [image_data(image,box) for image,box in zip(images,result['boxes'])]
            return jsonify(result)
        except (ValueError,UnidentifiedImageError,Image.DecompressionBombError) as error:
            return jsonify(error=str(error)),400
        except (OSError,RuntimeError) as error:
            return jsonify(error=f'读取图片或模型失败: {error}'),400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='上传内容超过 12 MB，请缩小图片。'),413

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7860)
    args = parser.parse_args()
    create_app().run(host=args.host,port=args.port,debug=False)


if __name__ == '__main__':
    main()
