import math
import pickle

from cyy_naive_lib.log import get_logger

from .shapley_value_server import ShapleyValueServer


class MultiRoundShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shapley_values = dict()
        self.round_trunc_threshold = kwargs.get("round_trunc_threshold", None)

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        metrics = dict()
        if self.round_trunc_threshold is not None:
            last_round_metric = self.get_metric(self.prev_model)
            this_round_metric = self.get_metric(aggregated_parameter)
            metrics[()] = last_round_metric
            metrics[tuple(sorted(range(self.worker_number)))] = this_round_metric
            get_logger().info(
                "this_round_metric %s last_round_metric %s round_trunc_threshold %s",
                this_round_metric,
                last_round_metric,
                self.round_trunc_threshold,
            )
            if abs(this_round_metric - last_round_metric) <= self.round_trunc_threshold:
                self.shapley_values[self.round] = {
                    i: 0 for i in range(self.worker_number)
                }
                return aggregated_parameter

        for subset in self.powerset(range(self.worker_number)):
            key = tuple(sorted(subset))
            if key not in metrics:
                subset_model = self.get_subset_model(subset)
                metric = self.get_metric(subset_model)
                metrics[key] = metric
                get_logger().info("round %s key %s metric %s", self.round, key, metric)

        round_shapley_values = dict()
        for subset, metric in metrics.items():
            if not subset:
                continue
            for client_id in subset:
                marginal_contribution = (
                    metric - metrics[tuple(sorted(i for i in subset if i != client_id))]
                )
                if client_id not in round_shapley_values:
                    round_shapley_values[client_id] = 0
                round_shapley_values[client_id] += marginal_contribution / (
                    (math.comb(self.worker_number - 1, len(subset) - 1))
                    * self.worker_number
                )
        with open("metric_" + str(self.round), "wb") as f:
            pickle.dump(metrics, f)

        self.shapley_values[self.round] = round_shapley_values
        get_logger().error("shapley_values %s", self.shapley_values)
        return aggregated_parameter
