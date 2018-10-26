import time

import numpy as np
import ray
from sklearn.linear_model import LinearRegression


class ModelActorBase(object):
    def __init__(self):
        self.metric = {"batch_size": []}

    def predict(self, input_batch):
        """Base method for accepting a batch of request. 

        Note:
            input_batch has the following format:
            [
                (inp_1, return_oid_1),
                (inp_2, return_oid_2),
                ... 
            ]
        
        Arguments:
            input_batch {List of tuples} -- shape outlined above
        """

        raise NotImplementedError("To be implemented in base class!")

    def get_metric(self):
        return self.metric

    def put_object(self, oid, obj):
        """Helper function for putting an object into worker when this
        oid is already provisioned
        
        Arguments:
            oid {ray.ObjectID} -- The provisioned object id
            obj {object} -- Any object plasma accepts
        """

        ray.worker.global_worker.put_object(oid, obj)


@ray.remote
class NoopModelActor(ModelActorBase):
    def predict(self, input_batch):
        for _, return_oid in input_batch:
            self.put_object(return_oid, "")


@ray.remote
class SleepModelActor(ModelActorBase):
    def predict(self, input_batch):
        time.sleep(0.1)

        self.metric["batch_size"].append(len(input_batch))

        for _, return_oid in input_batch:
            self.put_object(return_oid, "")


@ray.remote
class IdentityModelActor(ModelActorBase):
    def predict(self, input_batch):
        for input_data, return_oid in input_batch:
            self.put_object(return_oid, ray.get(input_data))


@ray.remote
class SKLearnModelActor(ModelActorBase):
    def __init__(self):
        self.model = LinearRegression()
        X = np.ones(shape=(100, 2))
        y = np.ones(100)
        self.model.fit(X, y)

        super().__init__()

    def predict(self, input_batch):
        for input_data, return_oid in input_batch:
            inp = ray.get(input_data)
            if inp.ndim == 1:
                inp = inp.reshape(1, -1)
            self.put_object(return_oid, self.model.predict(inp))
