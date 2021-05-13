from fastapi import APIRouter
from api import root, hello_world, predict

api_router = APIRouter()

api_router.include_router(root.router)
api_router.include_router(hello_world.router, prefix="/hello-world", tags=["Hello World!"])
api_router.include_router(predict.router, prefix="/predict", tags=["Predict PM2.5"])
