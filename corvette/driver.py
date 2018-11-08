import ray


def send_predict(qf_actor, model_name, query):
    if isinstance(query, ray.ObjectID):
        # We want to prevent intermediate materialization in frontend
        return ray.get(qf_actor.predict.remote(model_name, [query]))[0]
    else:
        return ray.get(qf_actor.predict.remote(model_name, [ray.put(query)]))[0]
