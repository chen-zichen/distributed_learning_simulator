import numpy as np
from cyy_naive_lib.log import get_logger

from .shapley_value_server import ShapleyValueServer


class GTGShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shapley_values = dict()
        # trunc paras
        self.eps = 0.001
        self.round_trunc_threshold = 0.01

        # converge paras
        self.converge_min = max(30, self.worker_number)
        self.last_k = 10
        self.converge_criteria = 0.05

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        last_round_metric = self.get_metric(self.prev_model)
        this_round_metric = self.get_metric(aggregated_parameter)
        get_logger().info(
            "last_round_metric %s ,this_round_metric %s, round_trunc_threshold %s",
            last_round_metric,
            this_round_metric,
            self.round_trunc_threshold,
        )
        if abs(last_round_metric - this_round_metric) <= self.round_trunc_threshold:
            self.shapley_values[self.round] = {i: 0 for i in range(self.worker_number)}
            return aggregated_parameter
        metrics = dict()

        index = 0
        contribution_records: list = []
        while self.not_convergent(index, contribution_records):
            for worker_id in range(self.worker_number):
                index += 1
                v = [0 for i in range(self.worker_number + 1)]
                v[0] = last_round_metric
                marginal_contribution = [0 for i in range(self.worker_number)]
                perturbed_indices = np.concatenate(
                    (
                        np.array([worker_id]),
                        np.random.permutation(
                            [i for i in range(self.worker_number) if i != worker_id]
                        ),
                    )
                ).astype(int)

                for j in range(1, self.worker_number + 1):
                    subset = tuple(sorted(perturbed_indices[:j].tolist()))
                    # truncation
                    if abs(this_round_metric - v[j - 1]) >= self.eps:
                        if subset not in metrics:
                            subset_model = self.get_subset_model(subset)
                            metric = self.get_metric(subset_model)
                            metrics[subset] = metric
                        v[j] = metrics[subset]
                    else:
                        v[j] = v[j - 1]

                    # update SV
                    marginal_contribution[perturbed_indices[j - 1]] = v[j] - v[j - 1]
                    contribution_records.append(marginal_contribution)

        # shapley value calculation
        shapley_value = np.sum(contribution_records, 0) / len(contribution_records)

        assert len(shapley_value) == self.worker_number

        # store round t results
        self.shapley_values[self.round] = {
            key: sv for key, sv in enumerate(shapley_value)
        }
        get_logger().info("shapley_value %s", self.shapley_values[self.round])
        return aggregated_parameter

    def not_convergent(self, index, contribution_records):
        if index <= self.converge_min:
            return True
        all_vals = (
            np.cumsum(contribution_records, 0)
            / np.reshape(np.arange(1, len(contribution_records) + 1), (-1, 1))
        )[-self.last_k:]
        errors = np.mean(
            np.abs(all_vals[-self.last_k:] - all_vals[-1:])
            / (np.abs(all_vals[-1:]) + 1e-12),
            -1,
        )
        if np.max(errors) > self.converge_criteria:
            return True
        get_logger().info(
            "not convergent for index %s and converge_min %s max error %s converge_criteria %s",
            index,
            self.converge_min,
            np.max(errors),
            self.converge_criteria,
        )
        return False
