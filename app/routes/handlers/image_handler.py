from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies.read_image_dependence import read_image
from app.dependencies.read_image_list_dependence import read_image_list
from app.services.image_service import ImageService

import numpy as np
from io import BytesIO
from typing import List

image_router = APIRouter()

#1. CARGA Y LECTURA DE IMAGENES

@image_router.post("/display")
async def display_image_endpoint(image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.display_image(image)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/webp")

@image_router.post("/shape")
async def image_shape_endpoint(image: np.ndarray = Depends(read_image)):
    return {"shape": await ImageService.image_shape(image)}

#2. SEGMENTACION DE IMAGENES
@image_router.post("/segmentation-color/")
async def segment_by_color_endpoint(color: str, image: np.ndarray = Depends(read_image)):
    adjusted_image = await ImageService.segment_by_color(image, color)
    return StreamingResponse(BytesIO(adjusted_image), media_type="image/png")

@image_router.post("/segmentation-kmeans/")
async def segment_by_kmeans_endpoint(value: int, image: np.ndarray = Depends(read_image)):
    adjusted_image = await ImageService.segment_by_kmeans(image, value)
    return StreamingResponse(BytesIO(adjusted_image), media_type="image/png")

#3. CONVERSION Y AJUSTES DE IMAGENES
@image_router.post("/brightness/")
async def adjust_brightness_endpoint(value: int = 0, image: np.ndarray = Depends(read_image)):
    adjusted_image = await ImageService.adjust_brightness(image, value)
    return StreamingResponse(BytesIO(adjusted_image), media_type="image/png")

@image_router.post("/contrast/")
async def adjust_contrast_endpoint( value: float = 0, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.adjust_contrast(image, value)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

@image_router.post("/shadows/")
async def adjust_shadows_endpoint(value: int = 0, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.adjust_shadows(image, value)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

@image_router.post("/exposure/")
async def adjust_exposure_endpoint(value: int = 0, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.adjust_exposure(image, value)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

@image_router.post("/highlights/")
async def adjust_highlights_endpoint(value: int = 0, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.adjust_highlights(image, value)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

@image_router.post("/grayscale")
async def grayscale_endpoint(image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.grayscale(image)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

#!CORREGIR ENVIO DE PARAMETROS
@image_router.post("/grayscale-rgb/")
async def grayscale_rgb_endpoint(red: float = 0, green: float = 0, blue: float = 0, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.grayscale_rgb(image, red, green, blue)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

@image_router.post("/negative")
async def negative_endpoint(image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.negative(image)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

#4. TRANSFORMACIONES Y ESCALADO DE IMAGENES
@image_router.post("/scale/")
async def scale_image_endpoint(scale_x: float=1.5, scale_y: float=1.5, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.scale_image(image, scale_x, scale_y)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

@image_router.post("/rotate/")
async def rotate_image_endpoint(angle: int = 45, image: np.ndarray = Depends(read_image)):
    img_encoded = await ImageService.rotate_image(image, angle)
    return StreamingResponse(BytesIO(img_encoded), media_type="image/png")

#5. ANALISIS Y ESTADISTICAS DE IMAGENES
@image_router.post("/histogram")
async def image_histogram_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.image_histogram(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/equalize-image")
async def equalize_image_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.equalize_image(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/equalize-histogram")
async def equalize_histogram_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.equalize_histogram(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

#6. DETECCION DE CARACTERISTICAS Y OBJETOS
@image_router.post("/detection/edge/")
async def edge_detection_endpoint(min_area: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.edge_detection(image, min_area)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/detection/objects/")
async def objects_detection_endpoint(min_area: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.objects_detection(image, min_area)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

#7. FILTROS Y MASCARAS
#MASKS
@image_router.post("/mask/filter-color/")
async def filter_color_mask_endpoint(color: str, return_image: bool, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.filter_color_mask(image, color, return_image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

#KERNELS
@image_router.post("/kernel/blur")
async def blur_kernel_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.blur_kernel(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/kernel/sobel")
async def sobel_kernel_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.sobel_kernel(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/kernel/sharpen")
async def sharpen_kernel_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.sharpen_kernel(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/kernel/laplacian")
async def laplacian_kernel_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.laplacian_kernel(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/kernel/embossing")
async def embossing_kernel_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.embossing_kernel(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

#FILTERS
@image_router.post("/filter/gaussian-blur/")
async def gaussian_blur_filter_endpoint(kernel_size: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.gaussian_blur_filter(image, kernel_size)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/filter/blur/")
async def average_blur_filter_endpoint(kernel_size: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.average_blur_filter(image, kernel_size)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/filter/edge-detection")
async def edge_detection_filter_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.edge_detection_filter(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/filter/laplace")
async def laplace_filter_filter_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.laplace_filter(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/filter/sobel/")
async def sobel_filter_endpoint(x_order: int, y_order: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.sobel_filter(image, x_order, y_order)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/filter/canny")
async def canny_filter_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.canny_filter(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")
#8. OPERACIONES CON IMAGENES
@image_router.post("/convolution/")
async def manual_convolution_endpoint(kernel_type: str, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.manual_convolution(image, kernel_type)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/fourier")
async def fourier_transform_endpoint(image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.fourier_transform(image)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/dilatation/")
async def dilatation_endpoint(iterations: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.dilatation(image, iterations)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/erosion/")
async def erosion_endpoint(iterations: int, image: np.ndarray = Depends(read_image)):
    histogram = await ImageService.erosion(image, iterations)
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/addition")
async def sum_images_endpoint(images: List[np.ndarray] = Depends(read_image_list)):
    histogram = await ImageService.sum_images(images[0], images[1])
    return StreamingResponse(BytesIO(histogram), media_type="image/png")

@image_router.post("/subtraction")
async def subtract_images_endpoint(images: List[np.ndarray] = Depends(read_image_list)):
    histogram = await ImageService.subtract_images(images[0], images[1])
    return StreamingResponse(BytesIO(histogram), media_type="image/png")
#9. APLICACIONES EN TIEMPO REAL