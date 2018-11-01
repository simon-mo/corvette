import ray


@ray.remote
class QueryFrontend:
    def __init__(self, batch_size=1):
        self.model_actors = {}
        self.models_queues = {}

        # result_oid -> model_name
        self.pending_queries = {}

        self.worker = ray.worker.global_worker
        self.batch_size = batch_size

    def _new_oid(self):
        oid = self.worker.local_scheduler_client.compute_put_id(
            self.worker.current_task_id, self.worker.put_index, self.worker.use_raylet
        )
        self.worker.put_index += 1
        return oid

    def add_model(self, model_name, model_class):
        if model_name in self.model_actors:
            return

        self.model_actors[model_name] = [model_class.remote()]
        self.models_queues[model_name] = []

    def scale_up_model(self, model_name, model_class, new_replica_count):
        replicas_to_add = new_replica_count - len(self.model_actors[model_name])
        if replicas_to_add <= 0:
            return
        for _ in range(replicas_to_add):
            self.model_actors[model_name].append(model_class.remote())

    def predict(self, model_name, query):
        data_oid = ray.put(query)
        result_oid = self._new_oid()
        self.models_queues[model_name].append((data_oid, result_oid))
        return [result_oid]

    def loop(self, actor_obj):
        ready, _ = ray.wait(list(self.pending_queries.keys()), timeout=0)
        models_finished = set()
        for ready_oid in ready:
            models_finished.add(self.pending_queries.pop(ready_oid))

        models_pending = set(self.pending_queries.values())
        all_models = set()
        for model_name, actors in self.model_actors.items():
            for i in range(len(actors)):
                all_models.add((model_name, i))
        models_finished |= all_models - models_pending

        for model_name, replica_id in models_finished:
            next_batch = self._get_next_batch(model_name)
            if len(next_batch) == 0:
                continue
            self.model_actors[model_name][replica_id].predict.remote(next_batch)
            for _, result_oid in next_batch:
                self.pending_queries[result_oid] = (model_name, replica_id)

        actor_obj.loop.remote(actor_obj)

    def _get_next_batch(self, model_name):
        inputs = []
        model_queue = self.models_queues[model_name]

        for _ in range(self.batch_size):
            if len(model_queue) == 0:
                break
            inputs.append(model_queue.pop(0))

        return inputs

    def get_metric(self):
        return {
            (model_name, replica_id): ray.get(model_actor.get_metric.remote())
            for model_name, model_actor_lst in self.model_actors.items()
            for replica_id, model_actor in enumerate(model_actor_lst)
        }
