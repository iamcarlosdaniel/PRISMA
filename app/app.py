from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.router import router

app = FastAPI()

#Define the CORS origins
origins = [
    "http://localhost:5173",
    "localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.include_router(router, prefix="/api/v1")
