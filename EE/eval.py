#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2023 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"


from scipy.special import softmax
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import generic_scaling
from policy import Policy
from analysis import Analysis
from plots import plot_exits
import numpy as np
from collections import OrderedDict
import wandb
import time
import json
import torch
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import HfApi

sns.set(color_codes=True)
sns.set_style("white")
sns.set_context("paper", font_scale=2)
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 100
from configs import parse_args, init_wandb, build_dataset, build_model


from metrics import (
    accuracy,
    brier_loss,
    nll,
    f1_micro,
    f1_macro,
    aurc_logits,
    ece_logits,
)  # AUROC_logits

METRICS = [accuracy, brier_loss, nll, f1_micro, f1_macro, ece_logits, aurc_logits]
from utils import (
    load_json,
    save_json,
    load_assets,
    config_to_checkpoint,
    load_dataset,
    get_logits,
    dump_logits,
    calc_metrics,
    logger_message,
)


def calc_flops(exit_distribution, analysis, model):
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
        "exit_distribution": exit_distribution,  # could make more clean afterwards like #text X%, #vision X%, #1 X%, #CLF X%
        "exit_threshold": config["exit_threshold"],
        "epsilon": config["epsilon"],
    }

    return efficiency_log


def eval_model(logits, references, raw_test_dataset, config, analysis):
    """apply exit policy and save exit distribution"""

    # create instance of class policy
    policy = Policy(logits=logits, config=config)

    # apply exit policy
    (
        exits_store,
        predictions,
        exit_distribution,
    ) = getattr(policy, config["exit_policy"])()

    # call function
    to_log = {}

    predictive_performance = calc_metrics(predictions, references)
    to_log.update(predictive_performance)

    efficiency_log = calc_flops(exit_distribution, analysis, model)
    to_log.update(efficiency_log)

    if config["plot_exits"]:
        plot_exits(model, logits, references, exits_store, raw_test_dataset)

    return to_log


def main(
    model=None,
    config=None,
    test_loader=None,
    run=None,
    logits=None,
    references=None,
    raw_test_dataset=None,
    dump_outputs=False,
    analysis=None,
):
    """
    3 modes of evaluation:

    1. Given a single entropy threshold, calculate accuracy and estimated flops
        ~ plot exit distribution
    2. FIXED exit layer performance (better to collect all logits and then compute) [no threshold]
    3. Given many different thresholds (better to collect all logits and then compute)
        ~ could search for Pareto optimal threshold (plot time vs accuracy given thresholds; choose one upper left)
    """

    """
    I could also just dump all logits for all exits to file and then run with different entropy thresholds -> this would be easier
    Then we can find the one with optimal time savings. (maybe not optimal final performance, but that's not the goal)
    
    TODO: We could also already dump a representative (100 samples) flops assessment, so we can calculate the flops for a specific exit from there
    """

    ## TODO: special flag that loads logits, length of input_ids and flops from file
    ### then do a loop over different entropy values and calculate accuracy and estimated flops

    ## TODO: calculate all metrics of interest:
    # accuracy, flops, wall-clock time with a given entropy threshold?
    ### TODO: wall-clock time is harder to save as we should use the actual entropy threshold and time it

    if dump_outputs:
        dump_logits(model, logits, references, config)
    else:
        logs = eval_model(
            logits, references, raw_test_dataset, config, analysis=analysis
        )

        if not config["downsampling"]:
            wandb.log(logs)

    return logs


def evaluate_checkpoint(args, checkpoint_dir):
    config = load_json(os.path.join(checkpoint_dir, "config.json"))
    config.update(args)
    references: np.typing.NDArray = np.load(
        os.path.join(checkpoint_dir, "references-test.npz")
    )["arr_0"]
    exit_logits: np.typing.NDArray = np.load(
        os.path.join(checkpoint_dir, "exit_logits-test.npz")
    )["arr_0"]
    N = len(references)

    # 1. Fixed exit evaluation irregardless of thresholding
    METRICS = [accuracy, brier_loss, nll, f1_micro, f1_macro, ece_logits, aurc_logits]
    collect = OrderedDict()
    for exit_id in range(exit_logits.shape[0]):
        for metric in METRICS:
            collect[
                f"exit_{exit_id} _{metric.__name__.replace('_logits', '')}"
            ] = metric(references, exit_logits[exit_id])

    print(collect)
    # print(pd.DataFrame([collect]).T.to_string())

    # 2. Evaluate with different entropy/confidence thresholds

    if config["inference_strategy"] == "max_confidence":
        thresholds = np.arange(0, 1, 0.01)
        fx = lambda x: np.max(scipy.special.softmax(x, axis=-1), -1)
    elif config["inference_strategy"] == "max_entropy":
        thresholds = np.arange(0, 10, 0.1)
        fx = lambda x: scipy.stats.entropy(scipy.special.softmax(x, axis=-1))  # base=2

    collect_two = OrderedDict()
    collect_three = OrderedDict()
    for threshold in thresholds:
        threshold = round(threshold, 2)
        collect_three[f"threshold_{threshold}_exits"] = []
        adaptive_logits = np.zeros_like(exit_logits[0])
        for i in range(N):
            for exit_id in range(exit_logits.shape[0]):
                if fx(exit_logits[exit_id][i]) > threshold:
                    break
            adaptive_logits[i] = exit_logits[exit_id][i]
            collect_three[f"threshold_{threshold}_exits"].append(exit_id)

        for metric in METRICS:
            collect_two[
                f"threshold_{threshold}_{metric.__name__.replace('_logits', '')}"
            ] = metric(references, adaptive_logits)
    print(collect_two)
    print(collect_three)

    # TODOD: 3. Multiple thresholds - find Pareto optimal usage
    ## requires memoization and then still a loop over all thresholds

    results = {
        "fixed": collect,
        "adaptive": collect_two,
        "adaptive_exits": collect_three,
    }
    save_json(os.path.join(checkpoint_dir, "results.json"), results)
    return results


def full_test_iteration(
    logits,
    references,
    config,
    raw_test_dataset,
    dump_outputs,
    start_threshold,
    step,
    analysis,
):
    thresholds = np.arange(start_threshold, 1, step)
    results = []
    for threshold in thresholds:
        print(f"At threshold {threshold}:")

        if config["exit_policy"] == "accuracy_calibration_heuristic":
            config["epsilon"] = threshold
        else:
            config["exit_threshold"] = threshold
        try:
            results.append(
                main(
                    logits=logits,
                    references=references,
                    config=config,
                    raw_test_dataset=raw_test_dataset,
                    run=init_wandb(config)
                    if config["exit_threshold"] != -1 and config["downsampling"] == 0
                    else None,
                    test_loader=test_loader,
                    dump_outputs=dump_outputs,
                    analysis=analysis,
                )
            )
        except Exception as e:
            logger_message(
                f"FAILED EXPERIMENT at threshold {threshold} due to {e}", type="error"
            )

    dir = os.path.join(config_to_checkpoint(config), config["exit_policy"])
    os.makedirs(dir, exist_ok=True)
    if config["calibrate"]:
        save_json(
            os.path.join(dir, "calibrated-metrics.json"),
            results,
        )
    else:
        save_json(os.path.join(dir, "non-calibrated-metrics.json"), results)


def calibrate(test_logits):
    """calibrate the exit policy on the test set"""
    logger_message("Calibrating exit logits on validation set", type="warning")
    # load validation set
    validation_dataset = load_dataset(model, config, "validation")
    # get logits for validation set
    config["labelset"] = "validation"
    validation_logits, validation_references, _ = get_logits(
        model, config, validation_dataset
    )

    # accuracy at exit
    # accuracy at final classifier
    # accuracy at final classifier with temperature scaling

    # intialize calibration
    calibrated_logits = np.zeros_like(test_logits)  # init with zeros
    temperatures = []  # init with ones
    ece = []
    accuracy = []
    average_confidence = []
    T = generic_scaling.TemperatureScaler()

    # check if logits are cached
    output_path = config_to_checkpoint(config)
    if os.path.exists(f"{output_path}/exit_logits-calibrated.npz"):
        logger_message(f"Loading calibrated logits from {output_path}", type="warning")
        calibrated_logits = np.load(
            os.path.join(output_path, "exit_logits-calibrated.npz")
        )["arr_0"]
        calibrated_metrics_config = load_json(os.path.join(output_path, "config.json"))
        config.update(calibrated_metrics_config)
        return np.asarray(calibrated_logits)

    nr_exits = test_logits.shape[0]
    for i in range(nr_exits):
        print(
            f"calibrate exit {i} with a validation exit accuracy of {np.mean(validation_logits[i,:,:].argmax(-1) == validation_references)} and average confidence of {softmax(validation_logits[i],-1).max(-1).mean()}"
        )
        T.fit(validation_references, validation_logits[i])

        print(
            f"exit {i} with a average test confidence of {softmax(test_logits[i],-1).max(-1).mean()}"
        )
        calibrated_logits[i] = T.temperature_scale(
            test_logits[i]
        )  # expensive replace with indexing
        post_calibration_ece = ece_logits(validation_references, calibrated_logits[i])
        ece.append(post_calibration_ece)
        average_confidence.append(softmax(calibrated_logits[i], -1).max(-1).mean())
        temperatures.append(T.temperature[0])
        print(
            f"POST-CALIBRATION: exit {i} with a average test confidence of {softmax(calibrated_logits[i],-1).max(-1).mean()}"
        )
        # calculate validation exit accuracy after calibration
        accuracy.append(
            np.mean(calibrated_logits[i].argmax(-1) == validation_references)
        )

    config["calibration_metrics"] = {}
    config["calibration_metrics"]["ece"] = ece
    config["calibration_metrics"]["accuracy"] = accuracy
    config["calibration_metrics"]["temperature"] = temperatures
    config["calibration_metrics"]["average_confidence"] = average_confidence

    # dump calibrated logits to test folder
    config["labelset"] = "test"
    dump_logits(model, calibrated_logits, None, config, name="calibrated")

    return np.asarray(calibrated_logits)


if __name__ == "__main__":
    # example non-exits:
    ## py eval.py -c jordyvl/LayoutLMv3_RVL-CDIP_NK100 -d jordyvl/RVL-CDIP-N
    ### jordyvl/EElayoutlmv3_rvl-cdip_single_10_2023-04-14
    ### EElayoutlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-04-17 --> /home/jordy/code/Beyond-Document-Page-Classification/src/EE/save/EElayoutlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-04-17
    #### -> upload latest model to HF hub
    config = parse_args({})
    api = HfApi()
    repo_id = config["checkpoint"]
    try:
        api.upload_file(
            path_or_fileobj="preprocessor_config.json",
            path_in_repo="preprocessor_config.json",
            repo_id=repo_id,
            repo_type="model",
        )
    except:
        pass

    checkpoint_dir = os.path.join("results", config["checkpoint"])
    if os.path.exists(checkpoint_dir):
        print(f"Evaluating checkpoint exit logits and flops for {config['checkpoint']}")
        evaluate_checkpoint(config, checkpoint_dir)
    else:
        logger_message(
            f"Testing model with exit thresholds {config['exit_threshold']} and exit_policy {config['exit_policy']}",
            type="info",
        )
        logger_message(
            f"Calibration Flag is set to {config['calibrate']}", type="warning"
        )

        # load model and test set once and put model on eval mode
        model, config, dump_outputs = load_assets(config)
        # load dataset
        test_loader = load_dataset(model, config, "test")
        # get logits
        logits, references, raw_test_dataset = get_logits(model, config, test_loader)

        analysis = Analysis(model)

        # calibrate logits
        if config["calibrate"]:
            logits = calibrate(logits)

        if bool(config["full_test"]) and config["exit_threshold"] != -1:
            logger_message(
                f"You are running a full test, with start threshold: {config['exit_threshold']} and step: {config['step']}",
                type="warning",
            )

            full_test_iteration(
                logits=logits,
                references=references,
                raw_test_dataset=raw_test_dataset,
                config=config,
                dump_outputs=False,
                start_threshold=config["exit_threshold"],
                step=config["step"],
                analysis=analysis,
            )
        elif config["exit_threshold"] == -1:
            main(
                logits=logits,
                references=references,
                raw_test_dataset=raw_test_dataset,
                config=config,
                run=None,
                test_loader=test_loader,
                dump_outputs=True,
                analysis=analysis,
            )
        else:
            main(
                logits=logits,
                references=references,
                raw_test_dataset=raw_test_dataset,
                config=config,
                run=init_wandb(config)
                if config["exit_threshold"] != -1 and config["downsampling"] == 0
                else None,
                test_loader=test_loader,
                dump_outputs=False,
                analysis=analysis,
            )
