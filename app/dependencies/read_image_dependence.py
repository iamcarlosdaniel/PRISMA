import numpy as np
import cv2
from fastapi import File, UploadFile

async def read_image(file: UploadFile = File(...)):
    image_bytes = np.fromstring(file.file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return image