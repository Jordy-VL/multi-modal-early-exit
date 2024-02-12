import torch
import scipy
import numpy as np


# create class for policy
class Policy:
    def __init__(self, logits, config) -> None:
        self.logits = logits
        self.config = config

    def max_confidence_global_thresholding_policy(self):
        """Given self.logits of shape (num_exits + 1, num_samples, num_labels) return the exit index and the predictions of shape (num_samples, num_labels)"""

        num_exits = self.logits.shape[0]  # number of exits
        num_samples = self.logits.shape[1]  # number of samples
        num_labels = self.logits.shape[2]  # number of labels
        threshold = self.config["exit_threshold"]  # global_threshold

        exits_store = np.zeros(num_samples, dtype=np.int32)
        # create torch tensor of shape (N, model.self.config.num_labels)
        predictions = torch.zeros(
            (num_samples, num_labels), dtype=torch.float64, device=self.config["device"]
        )
        # given early exits of self.logits of shape nr_exits + 1, NumSamples, model.self.config.num_labels) save the exit index if any of the self.logits is above the threshold
        # if no self.logits are above the threshold, save the last exit index
        # for each sample, save the exit index
        for sample_id in range(0, num_samples):
            for exit_id in range(self.logits.shape[0]):
                current_score = np.max(
                    scipy.special.softmax(self.logits[exit_id][sample_id])
                )
                if current_score > threshold:
                    exits_store[sample_id] = exit_id
                    # save self.logits of exit
                    predictions[sample_id] = torch.from_numpy(
                        self.logits[exit_id][sample_id]
                    )
                    break
                # if no exit is above the threshold, save the last exit
                if exit_id == self.logits.shape[0] - 1:
                    exits_store[sample_id] = exit_id
                    predictions[sample_id] = torch.from_numpy(
                        self.logits[exit_id][sample_id]
                    )
            # for each exit, calculate the percentage of samples that exited

        exit_distribution = {
            exit_id: np.count_nonzero(exits_store == exit_id) / num_samples
            for exit_id in range(0, num_exits)
        }

        return exits_store, predictions, exit_distribution

    def accuracy_calibration_heuristic(self):
        # for each exit calculate 1-accuracy/calibration error

        # if config doesn't have calibration_metrics key, raise exception
        if "calibration_metrics" not in self.config:
            raise Exception(
                "calibration_metrics not in config -> Set calibrate flag to True"
            )

        num_exits = self.logits.shape[0]  # number of exits
        num_samples = self.logits.shape[1]  # number of samples
        num_labels = self.logits.shape[2]  # number of labels

        accuracies = self.config["calibration_metrics"]["accuracy"]
        ece = self.config["calibration_metrics"]["ece"]
        average_confidence = self.config["calibration_metrics"]["average_confidence"]
        metrics = [1 - (accuracies[i] / ece[i]) for i in range(0, num_exits)]
        # scale metrics to ]0,1[
        epsilon = self.config["epsilon"]
        thresholds = (np.array(metrics) - (np.min(metrics) - epsilon)) / (
            (np.max(metrics) + epsilon) - (np.min(metrics) - epsilon)
        )
        exit_thresholds = {
            exit_id: thresholds[exit_id] for exit_id in range(0, num_exits)
        }

        exits_store = np.zeros(num_samples, dtype=np.int32)
        # create torch tensor of shape (N, model.self.config.num_labels)
        predictions = torch.zeros(
            (num_samples, num_labels), dtype=torch.float64, device=self.config["device"]
        )

        for sample_id in range(0, num_samples):
            for exit_id in range(num_exits):
                current_score = np.max(
                    scipy.special.softmax(self.logits[exit_id][sample_id])
                )
                if current_score > exit_thresholds[exit_id]:
                    exits_store[sample_id] = exit_id
                    # save self.logits of exit
                    predictions[sample_id] = torch.from_numpy(
                        self.logits[exit_id][sample_id]
                    )
                    break
                # if no exit is above the threshold, save the last exit
                if exit_id == self.logits.shape[0] - 1:
                    exits_store[sample_id] = exit_id
                    predictions[sample_id] = torch.from_numpy(
                        self.logits[exit_id][sample_id]
                    )
            # for each exit, calculate the percentage of samples that exited

        exit_distribution = {
            exit_id: np.count_nonzero(exits_store == exit_id) / num_samples
            for exit_id in range(0, num_exits)
        }
        return exits_store, predictions, exit_distribution
