import time

import numpy as np
import ray
import torch
from skimage import filters
from sklearn.linear_model import LinearRegression
from torchvision import models


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
        # def ensure_nested_get(doid):
        #     if isinstance(doid, list) and isinstance(doid[0], ray.ObjectID):
        #         return ray.get(doid)
        #     return doid
        get_name = lambda: self.__class__.__name__
        print(get_name(), input_batch)
        data_oids = [ray.get(doid) for (doid, _) in input_batch]
        result_oids = [roid for (_, roid) in input_batch]
        input_batch = data_oids
        result_batch = self.predict_batch(input_batch)
        for result, result_oid in zip(result_batch, result_oids):
            self.put_object(result_oid, result)

    def predict_batch(self, input_batch):
        raise NotImplementedError("Abstract method: to be implemented")

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
    def predict_batch(self, input_batch):
        return ["" for _ in range(len(input_batch))]


@ray.remote
class SleepModelActor(ModelActorBase):
    def predict_batch(self, input_batch):
        time.sleep(0.2)

        self.metric["batch_size"].append(len(input_batch))

        return ["" for _ in range(len(input_batch))]


@ray.remote
class IdentityModelActor(ModelActorBase):
    def predict_batch(self, input_batch):
        return input_batch


@ray.remote
class SKLearnModelActor(ModelActorBase):
    def __init__(self):
        self.model = LinearRegression()
        X = np.ones(shape=(100, 2))
        y = np.ones(100)
        self.model.fit(X, y)

        super().__init__()

    def predict_batch(self, input_batch):
        inp = np.array(input_batch)
        if inp.ndim == 1:
            inp = inp.reshape(1, -1)
        return self.model.predict(inp)


#  Armada Pipeline
@ray.remote
class ArmadaPreprocessModel(ModelActorBase):
    def predict_batch(self, input_batch):
        input_batch = np.array(input_batch)
        assert input_batch.shape == (len(input_batch), 224, 224, 3)
        return filters.gaussian(input_batch).reshape(len(input_batch), 3, 224, 224)


@ray.remote
class ArmadaSqueezenetModel(ModelActorBase):
    def __init__(self):
        self.model = models.squeezenet1_1()

    def predict_batch(self, input_batch):
        input_batch = np.array(input_batch)
        assert input_batch.shape == (len(input_batch), 3, 224, 224)

        with torch.no_grad():
            result = (
                self.model(torch.tensor(input_batch.astype(np.float32)))
                .detach()
                .numpy()
            )

        return result


@ray.remote
class ArmadaAverageEnsembleModel(ModelActorBase):
    def predict_batch(self, input_batch):
        return np.mean(np.array(input_batch), axis=0)
