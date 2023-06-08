import time
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from service.inference import ModelSerivce


app = FastAPI()
service = ModelSerivce()


class Query(BaseModel):
    query: str


@app.post("/cls/")
async def call(query: Query) -> Dict[str, str]:
    start = time.perf_counter()
    predict = service.predict(query.query)
    elapsed = time.perf_counter() - start
    result = {
        "predict": predict,
        "elapsed": elapsed,
    }
    return result
