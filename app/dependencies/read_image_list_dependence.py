from typing import List
import numpy as np
import cv2
from fastapi import File, UploadFile

async def read_image_list(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        image_bytes = np.fromstring(await file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        images.append(image)
    return images