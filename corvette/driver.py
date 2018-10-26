import ray


def send_predict(qf_actor, model_name, query):
    return ray.get(qf_actor.predict.remote(model_name, query))[0]
