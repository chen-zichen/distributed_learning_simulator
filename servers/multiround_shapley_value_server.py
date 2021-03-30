import math

from cyy_naive_pytorch_lib.model_util import ModelUtil

from shapley_value_server import ShapleyValueServer


class MultiRoundShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_model = ModelUtil(self.tester().model).get_parameter_dict()
        self.shapley_values = dict()

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        metrics = dict()
        for subset in self.powerset(range(self.worker_number)):
            subset_model = self.get_subset_model(subset, self._prev_model)
            metric = self.get_metric(subset_model)
            metrics[set(subset)] = metric

        round_shapley_values = dict()
        for subset, metric in metrics.items():
            if not subset:
                continue
            for client_id in subset:
                marginal_contribution = (
                    metric - metrics[set(i for i in subset if i != client_id)]
                )
                if client_id not in round_shapley_values:
                    round_shapley_values[client_id] = 0
                round_shapley_values[client_id] += marginal_contribution / (
                    (math.comb(self.worker_number - 1, len(subset) - 1))
                    * self.worker_number
                )

        self.shapley_values[self.round] = round_shapley_values
        self._prev_model = aggregated_parameter
