import numpy as np
import pytest
import ray
from sklearn.linear_model import LinearRegression

from driver import send_predict
from frontend import QueryFrontend
from model import IdentityModelActor, NoopModelActor, SKLearnModelActor, SleepModelActor


@pytest.fixture(scope="session")
def initialize_ray():
    ray.init()


@pytest.fixture(scope="session")
def query_frontend():
    # batch_size = 2
    qf = QueryFrontend.remote(2)
    qf.loop.remote(qf)
    return qf


def test_identity(initialize_ray, query_frontend):
    model_name = "identity"
    query_frontend.add_model.remote(model_name, IdentityModelActor)

    result_oids = []
    inputs = list(range(10))
    for i in inputs:
        result_oids.append(send_predict(query_frontend, model_name, i))
    result = ray.get(result_oids)

    assert result == inputs


def test_batching(initialize_ray, query_frontend):
    # We use a sleep actor to test batching is the model queue has
    # time to build up
    model_name = "sleep"
    query_frontend.add_model.remote(model_name, SleepModelActor)

    result_oids = []
    inputs = list(range(10))
    for i in inputs:
        result_oids.append(send_predict(query_frontend, model_name, i))
    ray.get(result_oids)

    metrics = ray.get(query_frontend.get_metric.remote())[("sleep", 0)]["batch_size"]
    assert 2 in metrics


@pytest.mark.timeout(10)
def test_sklearn(initialize_ray, query_frontend):
    model_name = "linear_regression"
    query_frontend.add_model.remote(model_name, SKLearnModelActor)

    model = LinearRegression()
    X = np.ones(shape=(100, 2))
    y = np.ones(100)
    model.fit(X, y)

    result_oids = []
    inputs = np.ones(shape=(100, 2)) * 2
    for i in inputs:
        result_oids.append(send_predict(query_frontend, model_name, i))
    result = np.array(ray.get(result_oids)).flatten()
    truth = model.predict(inputs).flatten()

    assert np.array_equal(result, truth)


def test_noop(initialize_ray, query_frontend, benchmark):
    model_name = "noop"
    query_frontend.add_model.remote(model_name, NoopModelActor)

    @benchmark
    def send_noop_and_wait():
        ray.get(send_predict(query_frontend, model_name, ""))


def test_replicas(initialize_ray, query_frontend):
    model_name = "sleep_replicated"
    query_frontend.add_model.remote(model_name, SleepModelActor)
    query_frontend.scale_up_model.remote(model_name, SleepModelActor, 3)

    result_oids = []
    inputs = list(range(30))
    for i in inputs:
        result_oids.append(send_predict(query_frontend, model_name, i))
    ray.get(result_oids)

    metrics = ray.get(query_frontend.get_metric.remote())
    for (name, replica_id), metric in metrics.items():
        if name == model_name:
            assert len(metric["batch_size"]) != 0


def test_pipeline(initialize_ray, query_frontend):
    model_1 = "noop_1"
    model_2 = "noop_2"
    query_frontend.add_model.remote(model_1, NoopModelActor)
    query_frontend.add_model.remote(model_2, NoopModelActor)

    # test materialization
    model_1_result = ray.get(send_predict(query_frontend, model_1, ""))
    model_2_result = ray.get(send_predict(query_frontend, model_2, model_1_result))
    assert model_2_result == ""

    # test no materialization
    model_1_result = send_predict(query_frontend, model_1, "")
    model_2_result = ray.get(send_predict(query_frontend, model_2, model_1_result))
    assert model_2_result == ""
