import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Response
import io
import pickle5
from pydantic import BaseModel
from model.model import predict, get_provinces

router = APIRouter()

class data(BaseModel):
    time: str
    province: str
    
    class Config:
        schema_extra = {
            "example": {
                "time":'2018-01-01 07:00:00',
                "province":"Bangkok"
            }
        }

input_time = '2018-01-01 07:00:00'
input_province = 'Bangkok'

@router.post("/")
async def get_predict(data: data):
    res = predict(data.time, data.province)
    return res

@router.get("/province")
async def get_province():
    provinces = get_provinces()
    return json.dumps({ "res": list(provinces) })