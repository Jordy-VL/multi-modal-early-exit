# %%
import os
import sys
import numpy as np  # logits are np?
from collections import OrderedDict
from scipy.special import softmax
import time
from joblib import Parallel, delayed
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly


def runtime_wrapper(input_function, *args, **kwargs):
    start_value = time.perf_counter()
    return_value = input_function(*args, **kwargs)
    end_value = time.perf_counter()
    runtime_value = end_value - start_value
    # print(f"Finished executing {input_function.__name__} in {runtime_value} seconds")
    return return_value, runtime_value


path = "EElayoutlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-09-05_txt_vis_con_enc_4_6_7_11_12_ramp-rvl_cdip_100_examples_per_class"
path = os.path.join("/home/jordy/code/opensource/EE/EE/results", path)


def checkpoint_logits(path):
    # load config
    # load checkpoint
    # get logits
    references = np.load(os.path.join(path, "references-test.npz"))["arr_0"]
    logits = np.load(os.path.join(path, "exit_logits-calibrated.npz"))["arr_0"]
    return references, logits


# CSF functions that will be thresholded


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


CSF_dict = OrderedDict(
    {
        "msp": lambda x: np.max(softmax(x, axis=-1), -1),  # maximum softmax probability
        "entropy": lambda x: -entropy(x),  # negative entropy
        "margin": lambda x: top12_margin_np(x),  # top-1 - top-2
    }
)


# a function that does thresholding on a Tensor with 3 dimensions (nr_exits, N, K) and returns a Tensor with 2 dimensions (N, K): predictions
## follow-up with that matrix, calculate metrics respective to returned logits - index problem


# naieve implementation ; 400 samples, 5 exits, 16 labels
def naieve(references, logits, thresholds=[0.5]):
    num_exits = logits.shape[0]  # number of exits
    num_samples = logits.shape[1]  # number of samples
    num_labels = logits.shape[2]  # number of labels
    num_thresholds = len(thresholds)  # number of thresholds
    thresholds = sorted(thresholds, reverse=True)

    collect_exits_store = np.ones((num_thresholds, num_samples), dtype=np.int32) * (
        num_exits
    )  # create numpy array of shape (N, ) and fill with nr_exits
    collect_predictions = np.tile(
        logits[-1], (num_thresholds, 1, 1)
    )  # create numpy array of shape (nr_thresholds, N, K) and fill with last exit logits

    for t, threshold in enumerate(thresholds):
        # given early exits of logits of shape nr_exits + 1, NumSamples, model.self.config.num_labels) save the exit index if any of the logits is above the threshold
        # if no logits are above the threshold, save the last exit index
        # for each sample, save the exit index
        for sample_id in range(num_samples):  # N
            for exit_id in range(num_exits):  # nr_exits from left to right
                current_score = CSF(logits[exit_id][sample_id])
                if current_score >= threshold:
                    collect_exits_store[t][sample_id] = exit_id
                    # save logits of exit
                    collect_predictions[t][sample_id] = logits[exit_id][sample_id]
                    break
    return collect_predictions, collect_exits_store


# 5s for 100 global thresholds
# 5 * 100**5 / 60 / 60 / 24
# %%


# %%


def opt0(references, logits, thresholds=[0.5]):
    num_exits = logits.shape[0]  # number of exits
    num_samples = logits.shape[1]  # number of samples
    num_labels = logits.shape[2]  # number of labels
    num_thresholds = len(thresholds)  # number of thresholds

    thresholds = sorted(thresholds, reverse=True)
    collect_exits_store = np.ones((num_thresholds, num_samples), dtype=np.int32) * (
        num_exits
    )  # create numpy array of shape (N, ) and fill with nr_exits
    collect_predictions = np.tile(
        logits[-1], (num_thresholds, 1, 1)
    )  # create numpy array of shape (nr_thresholds, N, K) and fill with last exit logits

    # precompute CSF for all logits (nr_exits, N)
    CSF_logits = np.apply_along_axis(CSF, -1, logits)

    # simplest yet most likely not most efficient way to do this
    for t, threshold in enumerate(thresholds):
        collect_exits_store[t] = (CSF_logits >= threshold).argmax(
            0
        )  # for each sample, save the exit index
        collect_predictions[t] = logits[
            collect_exits_store[t], np.arange(num_samples)
        ]  # index logits with exit index
    return collect_predictions, collect_exits_store


def opt1(references, logits, thresholds=[0.5]):
    num_exits = logits.shape[0]  # number of exits
    num_samples = logits.shape[1]  # number of samples
    num_labels = logits.shape[2]  # number of labels
    num_thresholds = len(thresholds)  # number of thresholds

    thresholds = sorted(thresholds, reverse=True)
    collect_exits_store = np.ones((num_thresholds, num_samples), dtype=np.int32) * (
        num_exits
    )  # create numpy array of shape (N, ) and fill with nr_exits
    collect_predictions = np.tile(
        logits[-1], (num_thresholds, 1, 1)
    )  # create numpy array of shape (nr_thresholds, N, K) and fill with last exit logits

    # precompute CSF for all logits (nr_exits, N)
    CSF_logits = np.apply_along_axis(CSF, -1, logits)

    # simplest yet most likely not most efficient way to do this
    (CSF_logits >= 0.5).argmax(0)  # for each sample, save the exit index

    tmp_above = np.zeros(
        (num_samples), dtype=np.bool
    )  # for all previous thresholds, which samples are above; which are left to check = False

    for t, threshold in enumerate(thresholds):
        # given early exits of logits of shape nr_exits + 1, NumSamples, model.self.config.num_labels) save the exit index if any of the logits is above the threshold
        # if no logits are above the threshold, save the last exit index
        # for each sample, save the exit index
        tmp_not = np.logical_not(
            tmp_above
        )  # which samples are below the previous threshold
        if not any(tmp_not):
            continue

        for exit_id in range(num_exits):  # nr_exits from left to right
            check = CSF_logits[exit_id, np.where(tmp_not)[0]] >= threshold
            updated = np.where(tmp_not)[0][np.where(check)[0]]
            tmp_above[updated] = True
            # for each sample, save the exit index; only update those that are above the current threshold
            for sample_id in updated:
                collect_exits_store[t][sample_id] = exit_id
                collect_predictions[t][sample_id] = logits[exit_id][sample_id]

    return collect_predictions, collect_exits_store


def opt2(references, logits, thresholds=[0.5]):
    pass  # TODO; approximation of opt1 with exit CSF monotonicity


def check_2D_threshold(CSF_logits, threshold):
    return (CSF_logits >= threshold[:, None]).argmax(0)


def opt0_2D(references, logits, thresholds_2D):
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
    """
    collect_predictions = np.tile(
        logits[-1], (num_thresholds, 1, 1)
    )  # create numpy array of shape (nr_thresholds, N, K) and fill with last exit logits
    # simplest yet most likely not most efficient way to do this
    for t, threshold in enumerate(thresholds_2D):
        # collect_exits_store[t] = (CSF_logits >= threshold[:, None]).argmax(0)  # for each sample, save the exit index
        collect_predictions[t] = logits[
            collect_exits_store[t], np.arange(num_samples)
        ]  # index logits with exit index; only needed for metric calculations TBH, should not be kept in memory
    return collect_predictions, collect_exits_store
    """
    return collect_exits_store


def parallel_process(fx, CSF_logits, thresholds):  # thresholds is a
    results = Parallel(n_jobs=8)(
        delayed(fx)(CSF_logits, thresh) for thresh in thresholds
    )
    # n_jobs=-1 means using all available cores
    # should only return the exits taken, the rest can be retrieved using the exit index and tensor indexing

    return results


def evaluate_exit_logits(logits, references, exit_distribution):
    accuracy = np.mean(
        np.argmax(logits[exit_distribution, np.arange(len(references))], axis=-1)
        == references
    )
    average_exit = np.mean(exit_distribution)  # proxy for average latency/throughput
    return accuracy, average_exit


if __name__ == "__main__":
    """
    Another optimization: no need to check threshold at last exit, == classifier, so exiting anyway
    """

    references, logits = checkpoint_logits(path)
    CSF = CSF_dict["msp"]

    # simple policy:  data-driven thresholds at num of percentiles of the maximum softmax probability per exit
    num_exits = logits.shape[0]
    num_per_exit = 10
    exit_thresholds = np.zeros((num_exits, num_per_exit))
    percentiles = np.linspace(0, 100, num_per_exit)
    for exit_id in range(num_exits - 1):
        CSF_at_exit = CSF(logits[exit_id])

        for p, perc in enumerate(percentiles):
            exit_thresholds[exit_id, p] = np.percentile(
                CSF_at_exit, perc
            )  # percentiles starting from 1- potential accuracy?

    num_mixtures = 1000000
    mixture_selection = [
        np.random.randint(0, num_per_exit, num_exits) for _ in range(num_mixtures)
    ]
    thresholds_2D = exit_thresholds[np.arange(num_exits), mixture_selection]

    # TODO: version of opt0 with 2D thresholds; complexity is dependent on num_mixtures mostly
    res2d, t2d = runtime_wrapper(opt0_2D, references, logits, thresholds_2D)
    print("t2d:", t2d)

    r = evaluate_exit_logits(logits, references, res2d[0])

    # for res2d, iter and do evaluation on the fly
    tuple_results = Parallel(n_jobs=8)(
        delayed(evaluate_exit_logits)(logits, references, res2d[thresh])
        for thresh in range(num_mixtures)
    )

    # scatter plot of index 0 and 1 of tuple_results

    # g = sns.stripplot(x=[x[0] for x in tuple_results], y=[x[1] for x in tuple_results])
    # # xlabel to accuracy
    # # ylabel to average exit
    # # title to "accuracy vs average exit"
    # # save to file
    # g.set(xlabel='accuracy', ylabel='average exit', title='accuracy vs average exit')
    # #g.savefig(".png")
    # plt.show()

    # # now do this in plotly with thresholds available as hovertext
    str_thresholds_2D = [str(np.round(x, 4)) for x in thresholds_2D]

    g = px.scatter(
        x=[x[0] for x in tuple_results],
        y=[x[1] for x in tuple_results],
        hover_data=[str_thresholds_2D],
    )
    g.update_layout(
        title="accuracy vs average exit",
        xaxis_title="accuracy",
        yaxis_title="average exit",
    )
    # html file
    plotly.offline.plot(g, filename="experiment.html")
    g.show()

    # TODO: also parallelize metrics evaluation and logits indexing ;)
    # Parallel(n_jobs=8)(delayed(fx)(CSF_logits, thresh) for thresh in thresholds)
    sys.exit(1)

    thresholds = np.linspace(0.1, 0.2, 100)
    res, t1 = runtime_wrapper(naieve, references, logits, thresholds=thresholds)
    res0, t0 = runtime_wrapper(opt0, references, logits, thresholds=thresholds)
    res2, t2 = runtime_wrapper(opt1, references, logits, thresholds=thresholds)

    print("naieve:", t1)
    print("opt0:", t0)  # still checks every sample for every exit though...
    print("opt1:", t2)

    if not np.array_equal(res[1], res2[1]):
        print("not equal")

    #
    #
    # next step: 2D thresholds

    # res[1], res2[1]
