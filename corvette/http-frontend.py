import asyncio
import time
from time import perf_counter

import numpy as np
import ray
import uvloop
from ray.experimental import async_api
from ray.experimental.named_actors import get_actor, register_actor
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from uvicorn import run

from driver import send_predict
from frontend import QueryFrontend
from model import (
    ArmadaAverageEnsembleModel,
    ArmadaPreprocessModel,
    ArmadaSqueezenetModel,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

previous_time = time.perf_counter()


def log_with_timestamp(msg):
    global previous_time
    now = time.perf_counter()
    print(f"[delta_ms={(now-previous_time)*1000}] {msg}")
    previous_time = now


ray.init()
async_api.init()

# Creating QueryFrontend
batchsize = 1
query_frontend = QueryFrontend.remote(batchsize)
query_frontend.loop.remote(query_frontend)
register_actor("query_frontend", query_frontend)

n_replicas = 3
query_frontend.add_model.remote("preprocess", ArmadaPreprocessModel)
for i in range(n_replicas):
    query_frontend.add_model.remote(f"squeezenet_{i}", ArmadaSqueezenetModel)
query_frontend.add_model.remote("ensemble", ArmadaAverageEnsembleModel)

input_image = np.random.randn(1, 224, 224, 3)


async def predict():
    log_with_timestamp("entering predict")

    upstream = send_predict(query_frontend, "preprocess", input_image)
    middle_tier_results = [
        send_predict(query_frontend, f"squeezenet_{i}", upstream)
        for i in range(n_replicas)
    ]
    downstream = send_predict(query_frontend, "ensemble", middle_tier_results)

    result = await async_api.as_future(downstream)

    log_with_timestamp("exiting predict")


app = Starlette()


@app.route("/")
async def index(request):
    start = perf_counter()
    await predict()
    return PlainTextResponse(str(perf_counter() - start) + "\n")


if __name__ == "__main__":
    run(app, loop=asyncio.get_event_loop())
