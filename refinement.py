"""Deterministic degradation for separately labelled robustness diagnostics."""
from io import BytesIO
from PIL import Image


def diagnostic_image(image):
    """Same fixed 40px/JPEG-45 condition for every image and every method."""
    small=image.convert('RGB').resize((40,40),Image.Resampling.BILINEAR)
    stream=BytesIO();small.save(stream,format='JPEG',quality=45)
    stream.seek(0)
    with Image.open(stream) as decoded:return decoded.convert('RGB').resize((112,112),Image.Resampling.BILINEAR)


class DiagnosticTransform:
    def __init__(self,transform):self.transform=transform
    def __call__(self,image):return self.transform(diagnostic_image(image))
