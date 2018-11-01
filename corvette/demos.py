import os
from concurrent.futures import ThreadPoolExecutor

import click
import ray
from prometheus_client import Summary, start_http_server
from ray.experimental.named_actors import get_actor, register_actor

from driver import send_predict
from frontend import QueryFrontend
from model import IdentityModelActor, NoopModelActor, SKLearnModelActor, SleepModelActor

os.environ["RAY_USE_XRAY"] = "1"


@click.group()
def demo():

    pass


@demo.command()
@click.option("--port", "-p", help="port for metric export", default=8000)
@click.option(
    "--num-replicas", "-n", type=int, help="number of replica to spin up", default=1
)
def noop(port, num_replicas):
    ray.init(use_raylet=True)

    query_frontend = QueryFrontend.remote(1)
    query_frontend.loop.remote(query_frontend)
    register_actor("query_frontend", query_frontend)

    model_name = "noop"
    query_frontend.add_model.remote(model_name, NoopModelActor)
    query_frontend.scale_up_model.remote(model_name, NoopModelActor, num_replicas)

    start_http_server(port, addr="0.0.0.0")

    metric = Summary("noop_seconds", "Time spent processing one noop prediction")

    @metric.time()
    def process_request():
        oids = []
        for _ in range(num_replicas):
            oids.append(send_predict(query_frontend, model_name, ""))
        ray.get(oids)

    while True:
        process_request()

    click.echo(f"Server started at http://0.0.0.0:{port}")


@demo.command()
@click.option("--port", "-p", help="port for metric export", default=8000)
@click.option("--materialize/--no-materialize", required=True)
def pipeline(port, materialize):
    ray.init(use_raylet=True)

    query_frontend = QueryFrontend.remote(1)
    query_frontend.loop.remote(query_frontend)
    register_actor("query_frontend", query_frontend)

    query_frontend.add_model.remote("sleep_1", SleepModelActor)
    query_frontend.add_model.remote("sleep_2", SleepModelActor)
    query_frontend.add_model.remote("sleep_3", SleepModelActor)

    start_http_server(port, addr="0.0.0.0")

    metric = Summary("pipeline_seconds", "Time spent processing one noop prediction")

    @metric.time()
    def process_request():
        upstream = send_predict(query_frontend, "sleep_1", "")
        if materialize:
            upstream = ray.get(upstream)
        downstream_1 = send_predict(query_frontend, "sleep_2", "")
        downstream_2 = send_predict(query_frontend, "sleep_3", "")
        ray.get([downstream_1, downstream_2])

    while True:
        process_request()

    click.echo(f"Server started at http://0.0.0.0:{port}")


demo()
