
from fastapi import APIRouter, Response
import io

router = APIRouter()

@router.get("/")
async def get_hello_world():
    res = 'hello world'
    return res
