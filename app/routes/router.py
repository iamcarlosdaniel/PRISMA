from fastapi import APIRouter

from app.routes.handlers import image_handler

router = APIRouter()

router.include_router(image_handler.image_router, prefix="/image", tags=["image"])