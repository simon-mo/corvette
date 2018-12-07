import os
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import ray
from prometheus_client import Summary, start_http_server
from ray.experimental.named_actors import get_actor, register_actor

from driver import send_predict
from frontend import QueryFrontend
from model import (
    ArmadaAverageEnsembleModel,
    ArmadaPreprocessModel,
    ArmadaSqueezenetModel,
)

os.environ["RAY_USE_XRAY"] = "1"


@click.group()
def demo():
    pass


@demo.command()
@click.option("--port", "-p", help="port for metric export", default=8000)
@click.option(
    "--n-replicas", "-n", help="number of replica for squeezenet model", default=1
)
@click.option("--batch-size", "-b", help="batch size of input", default=1)
def pipeline(port, n_replicas, batch_size):
    ray.init()

    # batchsize 1
    query_frontend = QueryFrontend.remote(batch_size)
    query_frontend.loop.remote(query_frontend)
    register_actor("query_frontend", query_frontend)

    query_frontend.add_model.remote("preprocess", ArmadaPreprocessModel)
    for i in range(n_replicas):
        query_frontend.add_model.remote(f"squeezenet_{i}", ArmadaSqueezenetModel)
    query_frontend.add_model.remote("ensemble", ArmadaAverageEnsembleModel)

    start_http_server(port, addr="0.0.0.0")
    click.echo(f"Server started at http://0.0.0.0:{port}")
    metric = Summary("pipeline_seconds", "Time spent processing one noop prediction")

    input_image = np.random.randn(batch_size, 224, 224, 3)

    @metric.time()
    def process_request(input_image):
        upstream = send_predict(query_frontend, "preprocess", input_image)
        middle_tier_results = [
            send_predict(query_frontend, f"squeezenet_{i}", upstream)
            for i in range(n_replicas)
        ]
        downstream = send_predict(query_frontend, "ensemble", middle_tier_results)
        ray.get([downstream])

    while True:
        process_request(input_image)


demo()
