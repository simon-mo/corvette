Corvette
-------

## Prediction serving on Ray

This is a prototype for prediction serving on Ray. It provides:
- Variable size batching. It will dynamically batch input to an actor as input come in, and dispatch this microbatch to one of the model actor.
- Scale up API. You can adjust the number of replica directly. 
- Model pipelines. You can easily chain models together.

Features in the timeline:
- Custom resource constraint like GPUs
- Push-based object store optimization to transfer prediction input across nodes.

Again, this is a prototype only. For production usage, we built [Clipper](http://clipper.ai), a real-time online prediction serving system. 

## Dependencies
Our only hard dependency is `ray[dev]`. Please install it via `pip install ray[dev]`.

For full development dependency, please see `Pipfile` or `requirements.txt`.

## Example
Try out `cd corvette; python demo.py` (requires sklearn and pytorch).

```python
from corvette.models import SKlearnModelActor, NoopModelActor
from corvette.driver import send_predict
import ray
ray.init()

qf = QueryFrontend.remote(2)
qf.loop.remote(qf) # start query frontend actor

qf.add_model.remote("linear_regression", SKLearnModelActor)
result_object_id = send_predict(qf, "linear_regression", [1,2,3])

qf.add_model.remote("noop_downstream", NoopModelActor)
final_result_object_id = send_predict(qf, "noop_downstream", result_object_id)

# to get the final result
ray.get(final_result_object_id)
```
