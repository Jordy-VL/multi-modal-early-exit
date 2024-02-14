import logging

import torch
from transformers import DataCollatorWithPadding
from configs import build_dataset, build_model
import numpy as np
import json
import os
from data import AverageMeter
import time
from tqdm import tqdm
from collections import OrderedDict
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

# logging formats
logging_formats = {
    "info": "\x1b[6;30;42m%(asctime)s - %(name)s - %(levelname)s - %(message)s\x1b[0m",
    "error": "\x1b[6;30;41m%(asctime)s - %(name)s - %(levelname)s - %(message)s\x1b[0m",
    "warning": "\x1b[6;30;43m%(asctime)s - %(name)s - %(levelname)s - %(message)s\x1b[0m",
}


# import time
# ## Define hook functions
# take_time_dict = {}

# def take_time_pre(layer_name,module, input):
#     take_time_dict[layer_name] = {}
#     take_time_dict[layer_name]["start"] = time.time()

# def take_time(layer_name,module, input, output):
#     take_time_dict[layer_name]["end"] = time.time()
#     take_time_dict[layer_name]["time"] = take_time_dict[layer_name]["end"] - take_time_dict[layer_name]["start"]
#     take_time_dict[layer_name]["accumulated_time"] = take_time_dict[layer_name]["end"] - take_time_dict["embeddings"]["start"]


def load_assets(config):
    """load model and test set once and put model on eval mode"""
    torch.set_grad_enabled(False)
    # load config if not present from model
    if config["checkpoint"]:
        config["model_weights"] = config["checkpoint"]

    # load model, fix thresholding name for model init
    if config["exit_threshold"] != -1:
        config["global_threshold"] = config["exit_threshold"]  # different naming
    model, config = build_model(config)
    from functools import partial

    ## override entropy threshold
    dump_outputs = False
    if hasattr(model.config, "exit_config"):
        if (
            config["exit_threshold"] == -1
        ):  # default for dumping outputs (impossible thresholds)
            model.config.exit_config["global_threshold"] = (
                2
                if model.config.exit_config["inference_strategy"] == "max_confidence"
                else -np.inf
            )
            dump_outputs = True
        else:  # override
            model.config.exit_config["global_threshold"] = config["exit_threshold"]
            model.config.exit_config["inference_strategy"] = config[
                "inference_strategy"
            ]

    model.eval()

    return model, config, dump_outputs


def load_dataset(model, config, labelset):
    # load dataset
    test_dataset = build_dataset(config, labelset, processor=model.processor)

    # datacollator
    tokenizer = (
        model.processor.tokenizer
        if hasattr(model.processor, "tokenizer")
        else model.processor
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset.data,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length"),
    )

    return test_loader


def save_json(file_path, data):
    logger_message(f"Saving results to {file_path}")
    with open(file_path, "w+") as json_file:
        json.dump(data, json_file, indent=4)


def load_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def config_to_checkpoint(config):
    output_path = os.path.join(
        "results",
        f"{config['checkpoint'].split('/')[-1]}-{config['test_dataset'].split('/')[-1]}",
    )
    if config["downsampling"]:
        output_path += f"-{config['downsampling']}i"
    return output_path


def get_logits(model, config, test_loader):
    raw_test_dataset = None
    if config["benchmark_OCR"] or config["plot_exits"]:
        raw_test_dataset = build_dataset(config, "test", processor=None)
        if not config["plot_exits"]:
            raw_test_dataset.processor = model.processor
            raw_test_dataset.processor.image_processor.apply_ocr = False

    # run evaluation experiment
    data_time = AverageMeter()
    correct = AverageMeter()
    references = np.array(
        [x["labels"].numpy() for i, x in enumerate(test_loader)]
    ).squeeze()  # torched
    if config["downsampling"]:
        references = references[: config["downsampling"]]
    N = len(references)
    nr_exits = len(
        model.config.exit_config["exits"]
    )  # should be +1 for final classifier

    # check if logits are already saved
    output_path = config_to_checkpoint(config)
    if os.path.exists(
        f"{output_path}/exit_logits-{config['labelset']}.npz"
    ) and os.path.exists(f"{output_path}/references-{config['labelset']}.npz"):
        logger_message(f"Loading {config['labelset']} logits from {output_path}")
        references = np.load(
            os.path.join(output_path, f"references-{config['labelset']}.npz")
        )["arr_0"]
        logits = np.load(
            os.path.join(output_path, f"exit_logits-{config['labelset']}.npz")
        )["arr_0"]
        return logits, references, raw_test_dataset
    else:
        logits_store = torch.zeros(
            (nr_exits + 1, N, model.config.num_labels),
            dtype=torch.float64,
            device=config["device"],
        )

        end = time.time()
        latencies = []

        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            if config["downsampling"] and i > 0 and i % config["downsampling"] == 0:
                break
            data_time.update(time.time() - end)
            batch = {k: v.to(config["device"]) for k, v in batch.items()}

            with torch.no_grad():
                if config["benchmark_OCR"]:  # implicitly adds tesseract OCR to timer
                    raw_test_dataset.preprocess_data(raw_test_dataset[i])

                outputs = model.forward(**batch)

                # latencies.append(take_time_dict)
                for j in range(nr_exits):
                    if (
                        outputs.gated_logits is not None
                        and len(outputs.gated_logits) > 0
                    ):
                        # save logits of exit for each sample
                        logits_store[j, i] = outputs.gated_logits[j]
                    elif outputs.lte_output is not None and len(outputs.lte_output) > 0:
                        logits_store[j, i] = outputs.lte_output[j]
                    else:
                        logits_store[j, i] = outputs.exit_states[j][0]
                    logits_store[-1, i] = outputs.logits[0]

                correct.update(
                    (outputs.logits.argmax(-1) == batch["labels"]).float().sum().item()
                )

                if i % config["print_freq"] == 0:
                    print(
                        f"i: {i}, correct: {correct.avg}, time/s: {data_time.avg}, "
                    )  # exited_ids: {exited_ids.val}
                end = time.time()

        # save all logits to the same folder but with different names
        if config["labelset"] == "test":
            name = "exit_logits"
        elif config["labelset"] == "validation":
            name = "validation"

        name = "test" if config["labelset"] == "test" else "validation"
        # if config.get("latencies", None) is not None:
        #     latencies_metrics = config["latencies"]
        #     config["latencies"] = latencies_metrics
        # if name == "test":
        #     exit_latencies = {k for k in latencies[0].keys() if k!="embeddings"}
        #     latencies_metrics = {k: sum([v[k]["accumulated_time"] for v in latencies if k!="embeddings"])/len(latencies) for k in exit_latencies}
        #     config["latencies"] = latencies_metrics

        config["labelset"] = "test"
        dump_logits(model, logits_store, references, config, name=name)

    return logits_store.cpu().data.numpy(), references, raw_test_dataset


def calc_metrics(predictions, references):
    """calculate all metrics of interest"""
    predictive_performance = OrderedDict()
    # if predictions is type tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().data.numpy()
    for metric in METRICS:
        predictive_performance[f"{metric.__name__.replace('_logits', '')}"] = metric(
            references, predictions
        )

    return predictive_performance


def dump_logits(model, logits, references, config, name="test"):
    output_path = config_to_checkpoint(config)
    os.makedirs(output_path, exist_ok=True)
    logger_message(f"Saving {name} to {output_path}")
    if references is not None:
        np.savez_compressed(
            os.path.join(output_path, f"references-{name}.npz"),
            torch.from_numpy(references).cpu().data.numpy(),
        )
    # check if logits are of type array, convert to tensor
    if type(logits) == np.ndarray:
        logits = torch.from_numpy(logits)
    np.savez_compressed(
        os.path.join(output_path, f"exit_logits-{name}.npz"),
        logits.cpu().data.numpy(),
    )
    to_save_config = config.copy()
    to_save_config.update(model.config.exit_config)
    # pop these keys exit_threshold, global_threshold, inference_strategy, exit_policy, use_lte, use_wandb, "calibrate": "True","full_test": "True","step": 0.05,"epsilon": 0.1,
    to_save_config.pop("exit_threshold", None)
    to_save_config.pop("global_threshold", None)
    to_save_config.pop("inference_strategy", None)
    to_save_config.pop("exit_policy", None)
    to_save_config.pop("use_lte", None)
    to_save_config.pop("use_wandb", None)
    to_save_config.pop("calibrate", None)
    to_save_config.pop("full_test", None)
    to_save_config.pop("step", None)
    to_save_config.pop("epsilon", None)

    save_json(os.path.join(output_path, "config.json"), to_save_config)
    tqdm.write("saved raw test outputs to {}".format(output_path))


# define method called logger_message
def logger_message(message, type="info"):
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(logging_formats[type])
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if type == "info":
        logger.info(message)
    if type == "error":
        logger.error(message)
    elif type == "warning":
        logger.warning(message)

    logger.removeHandler(handler)
    return logger
