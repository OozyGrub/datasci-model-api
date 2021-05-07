from fastapi import APIRouter

from api import root, hello_world

api_router = APIRouter()

api_router.include_router(root.router)
api_router.include_router(
    hello_world.router, prefix="/hello-world", tags=["Title Generation"])
