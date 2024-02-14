import numpy as np
from collections import OrderedDict
from scipy.special import softmax
from joblib import Parallel, delayed
import multiprocessing
import time
from utils import calc_metrics, logger_message
from analysis import Analysis
from utils import save_json


CSF_dict = OrderedDict(
    {
        "msp": lambda x: np.max(softmax(x, axis=-1), -1),  # maximum softmax probability
        "entropy": lambda x: -entropy(x),  # negative entropy
        "margin": lambda x: top12_margin_np(x),  # top-1 - top-2
    }
)


def entropy(x):
    exp_x = np.exp(x)
    A = np.sum(exp_x, axis=-1)  # sum of exp(x_i)
    B = np.sum(x * exp_x, axis=-1)  # sum of x_i * exp(x_i)
    return np.log(A) - B / A


def top12_margin_np(x):
    values = np.sort(x, axis=-1)
    if x.ndim == 1:
        return values[0] - values[1]
    return values[:, 0] - values[:, 1]


def parallel_process(fx, CSF_logits, thresholds):
    results = Parallel(n_jobs=8)(
        delayed(fx)(CSF_logits, thresh) for thresh in thresholds
    )
    return results


def check_2D_threshold(CSF_logits, threshold):
    return (CSF_logits >= threshold[:, None]).argmax(0)


def generate_thresholds(logits, references):
    CSF = CSF_dict["msp"]
    np.random.seed(42)
    num_exits = logits.shape[0]
    exit_thresholds = np.zeros((num_exits, num_per_exit))
    percentiles = np.linspace(0, 100, num_per_exit)
    for exit_id in range(num_exits - 1):
        CSF_at_exit = CSF(logits[exit_id])

        for p, perc in enumerate(percentiles):
            exit_thresholds[exit_id, p] = np.percentile(
                CSF_at_exit, perc
            )  # percentiles starting from 1- potential accuracy?

    mixture_selection = [
        np.random.randint(0, num_per_exit, num_exits) for _ in range(num_mixtures)
    ]
    thresholds_2D = exit_thresholds[np.arange(num_exits), mixture_selection]

    return thresholds_2D


def opt0_2D(references, logits, thresholds_2D):
    CSF = CSF_dict["msp"]
    num_exits = logits.shape[0]  # number of exits
    num_samples = logits.shape[1]  # number of samples
    num_thresholds = len(thresholds_2D)  # number of sets of thresholds

    # precompute CSF for all logits (nr_exits, N)
    CSF_logits = np.apply_along_axis(CSF, -1, logits)

    collect_exits_store = np.ones((num_thresholds, num_samples), dtype=np.int32) * (
        num_exits
    )  # create numpy array of shape (N, ) and fill with nr_exits

    collect_exits_store = parallel_process(
        check_2D_threshold, CSF_logits, thresholds_2D
    )
    return collect_exits_store


def evaluate_exit_logits(exit_distribution):
    predictive_performance = {}

    # predictions = logits[exit_distribution, np.arange(len(references))]
    # predictive_performance = calc_metrics(predictions, references)
    accuracy = np.mean(
        np.argmax(logits[exit_distribution, np.arange(len(references))], axis=-1)
        == references
    )
    average_exit = np.mean(exit_distribution)  # proxy for average latency/throughput
    # convert exit_distribution to exit_distribution per sample
    # for each sample, calculate the percentage of samples that exited

    exit_distribution = {
        exit_id: np.count_nonzero(exit_distribution == exit_id) / len(references)
        for exit_id in range(0, num_exits)
    }

    (
        full_param_count,
        multiexit_param_count,
        full_flops_count,
        multiexit_flops_count,
        full_latency,
        multiexit_latency,
    ) = analysis.exit_to_params_and_flops(exit_distribution, model)
    efficiency_log = {
        "#Params(M) used": multiexit_param_count / 1e6,
        "#Params(M) total": full_param_count / 1e6,
        "#GFLOPs used": multiexit_flops_count / 1e9,
        "#GFLOPs total": full_flops_count / 1e9,
        "GFLOPs reduction": 1 - float(multiexit_flops_count / full_flops_count),
        "Params reduction": 1 - float(multiexit_param_count / full_param_count),
        "Latency reduction": 1 - float(multiexit_latency / full_latency),
        "exit_distribution": exit_distribution,
    }

    predictive_performance["average_exit"] = average_exit
    predictive_performance["accuracy"] = accuracy
    predictive_performance["efficiency_log"] = efficiency_log

    return predictive_performance


def runtime_wrapper(input_function, *args, **kwargs):
    start_value = time.perf_counter()
    return_value = input_function(*args, **kwargs)
    end_value = time.perf_counter()
    runtime_value = end_value - start_value
    logger_message(f"Early exit indices calulated in {runtime_value}", type="info")
    return return_value, runtime_value


def evaluate(thres):
    return evaluate_exit_logits(exit_distribution=res2d[thres])


def calculate_large_scale_metrics(
    res2d, thresholds_2D, logits, references, analysis, model
):
    start = time.time()
    pool = multiprocessing.Pool(8)  # create a pool of processes
    from functools import partial

    g = partial(
        evaluate,
        res2d=res2d,
        logits=logits,
        references=references,
        analysis=analysis,
        model=model,
    )

    results = pool.map(
        g, range(num_mixtures)
    )  # apply the function to each element of the range
    pool.close()  # close the pool
    pool.join()  # wait for the processes to finish
    end = time.time()
    logger_message(f"Large scale metrics calculated in {end-start}", type="info")

    # combne results and thresholds2D
    results = [(thresholds_2D[i], results[i]) for i in range(len(results))]
    return results


if __name__ == "__main__":
    from utils import load_json, build_model
    import os
    from thresh import checkpoint_logits
    import argparse

    num_per_exit = 10
    num_mixtures = 1500000
    n_jobs = 8
    seed = 42

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=False,
        help="path for calibrated logits",
    )
    args = parser.parse_args()
    path = args.path
    # path = os.path.join("results/", path)
    references, logits = checkpoint_logits(path)
    num_exits = logits.shape[0]
    config = load_json(os.path.join(path, "config.json"))
    model, config = build_model(config)
    analysis = Analysis(model)
    thresholds_2D = generate_thresholds(logits, references)
    import json

    with open("thresholds_2D.json", "w") as f:
        json.dump(thresholds_2D.tolist(), f)

    res2d, t2d = runtime_wrapper(opt0_2D, references, logits, thresholds_2D)

    pool = multiprocessing.Pool(8)  # create a pool of processes
    from functools import partial

    start = time.time()
    results = pool.map(
        evaluate, range(num_mixtures)
    )  # apply the function to each element of the range
    pool.close()  # close the pool
    pool.join()  # wait for the processes to finish
    end = time.time()
    logger_message(f"time taken for {path} is {end-start}")

    # results = calculate_large_scale_metrics(
    #     res2d, thresholds_2D, logits, references, analysis, model
    # )
    dir = os.path.join(path, "multi_threshold")
    os.makedirs(dir, exist_ok=True)
    save_json(os.path.join(dir, "reults.json"), results)
