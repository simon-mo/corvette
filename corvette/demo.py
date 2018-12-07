import time

import numpy as np
import ray
from ray.experimental.named_actors import get_actor, register_actor

from driver import send_predict
from frontend import QueryFrontend
from model import (
    ArmadaAverageEnsembleModel,
    ArmadaPreprocessModel,
    ArmadaSqueezenetModel,
)

previous_time = time.perf_counter()


def log_with_timestamp(msg):
    global previous_time
    now = time.perf_counter()
    print(f"[delta_ms={(now-previous_time)*1000}] {msg}")
    previous_time = now


ray.init()

# Creating QueryFrontend
batchsize = 1
query_frontend = QueryFrontend.remote(batchsize)
query_frontend.loop.remote(query_frontend)
register_actor("query_frontend", query_frontend)

# Adding our Pipeline
#
#            -> squeezenet_1
# preprocess -> squeezenet_2 -> ensemble
#            -> squeezenet_3
#
n_replicas = 3
query_frontend.add_model.remote("preprocess", ArmadaPreprocessModel)
for i in range(n_replicas):
    query_frontend.add_model.remote(f"squeezenet_{i}", ArmadaSqueezenetModel)
query_frontend.add_model.remote("ensemble", ArmadaAverageEnsembleModel)


def process_request():
    input_image = np.random.randn(batchsize, 224, 224, 3)
    log_with_timestamp("Request sent!")
    upstream = send_predict(query_frontend, "preprocess", input_image)
    middle_tier_results = [
        send_predict(query_frontend, f"squeezenet_{i}", upstream)
        for i in range(n_replicas)
    ]
    downstream = send_predict(query_frontend, "ensemble", middle_tier_results)
    log_with_timestamp(f"Result get! {ray.get([downstream])[0][:10]}")


while True:
    process_request()
    log_with_timestamp("sleep 0.1s")
    time.sleep(0.1)
