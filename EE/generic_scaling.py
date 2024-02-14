import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from metrics import ece_logits
from sklearn.linear_model import LogisticRegression


def get_platt_scaler(model_probs, labels):
    clf = LogisticRegression(C=1e10, solver="lbfgs")
    eps = 1e-12
    model_probs = model_probs.astype(dtype=np.float64)
    model_probs = np.expand_dims(model_probs, axis=-1)
    model_probs = np.clip(model_probs, eps, 1 - eps)
    model_probs = np.log(model_probs / (1 - model_probs))
    clf.fit(model_probs, labels)

    def calibrator(probs):
        x = np.array(probs, dtype=np.float64)
        x = np.clip(x, eps, 1 - eps)
        x = np.log(x / (1 - x))
        x = x * clf.coef_[0] + clf.intercept_
        output = 1 / (1 + np.exp(-x))
        return output

    return calibrator


def manual_NLL(y_true, P):
    log_sum_exp = logsumexp(a=P, axis=1)
    tsb_logits_trueclass = np.sum(P * y_true, axis=1)
    log_likelihoods = tsb_logits_trueclass - log_sum_exp
    nll = -np.mean(log_likelihoods)
    return nll


class TemperatureScaler:
    """
    Class taking in validation logits, learning temperature for modulating test
    """

    def __init__(self, temperature=None):
        if not temperature:
            self.temperature = np.ones(1)  ##DEV: trick to warmstart *1.5
        else:
            self.temperature = np.ones(1) * temperature

    def fit(self, labels, logits):
        return self.set_temperature(labels, logits)

    def transform(self, logits):  # softmax?
        return softmax(self.temperature_scale(logits), -1)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = np.resize(self.temperature, logits.shape)
        # temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / (temperature)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, labels, logits):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        """

        def nll(labels, P):
            indices = np.arange(logits.shape[-1])
            return log_loss(labels, P, labels=indices)

        def objective(temperature, labels, logits):
            return log_loss(
                labels,
                softmax(logits / temperature, -1),
                labels=np.arange(logits.shape[-1]),
            )  # should give some score

        # Calculate NLL and ECE before temperature scaling
        preprobs = self.transform(logits)
        before_temperature_nll = nll(labels, preprobs)
        before_temperature_ece = ece_logits(labels, preprobs)
        print(
            "Before temperature - NLL: %.6f, ECE: %.6f"
            % (before_temperature_nll, before_temperature_ece)
        )

        result = minimize(
            objective,
            x0=self.temperature,
            method="L-BFGS-B",
            args=(labels, logits),
            bounds=[(1e-32, None)],
        )  # , options={'gtol': 1e-6, 'disp': True})
        assert result.success == True

        self.temperature = result.x

        # Calculate NLL and ECE after temperature scaling
        probs = self.transform(logits)  # temperature applied
        after_temperature_nll = nll(labels, probs)
        after_temperature_ece = ece_logits(labels, probs)
        print("Optimal temperature: %.6f" % self.temperature)
        print(
            "After temperذature - NLL: %.6f, ECE: %.6f"
            % (after_temperature_nll, after_temperature_ece)
        )

        return self.temperature


def test_temperature_scaler():
    sample_batch_logits = np.random.randn(3, 5)
    sample_batch_y = np.random.choice(range(3), 3)

    T = TemperatureScaler()
    T.fit(sample_batch_y, sample_batch_logits)

    scaled_P = T.transform(sample_batch_logits)


if __name__ == "__main__":
    test_temperature_scaler()
