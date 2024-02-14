import os
import random
import numpy as np
import torch
import argparse
import wandb
from datasets import load_dataset
from datasets import Features, ClassLabel

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoTokenizer,
)

from sacred import Experiment, SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # important new switch!
DEFAULT_EXPERIMENT = "DocumentClassification"
ex = Experiment(DEFAULT_EXPERIMENT)


@ex.config
def default():
    model = "LTELayoutLMv3"
    dataset = "jordyvl/rvl_cdip_100_examples_per_class"
    lowercase = False
    apply_ocr = True
    downsampling = 0
    eval_start = False

    epochs = 20
    batch_size = 2
    eval_batch_size = 1
    lr = 1e-04
    optimizer = "AdamW"
    warmup_ratio = 0
    weight_decay = 0
    gradient_accumulation_steps = 1

    seed = 42
    device = "cuda"

    use_wandb = False

    # EE config
    training_strategy = "joint_weighted_avg"
    inference_strategy = "max_confidence"
    global_threshold = 0.9
    exits = ["text_visual_concat"] + [6] # setup doesn't support vision only
    encoder_layer_strategy = "ramp"
    exit_head_num_layers = 2
    use_lte = False

    alpha = 0.5  # dunno what to use for yet
    temperature = 1  ## only relevant for NKD loss -> best to set temperature to 1 -> self supervised distillation?
    gamma = 0


@ex.named_config
def layoutlmv3():
    epochs = 20
    lr = 2e-05
    gradient_accumulation_steps = 32
    global_threshold = 1 + 1e-6


@ex.named_config
def debugEE():
    model = "LTElayoutlmv3"
    dataset = "rvl-cdip_single_10"
    epochs = 1
    lr = 2e-05
    batch_size = 1
    gradient_accumulation_steps = 1


def parse_args(_config):
    parser = argparse.ArgumentParser(description="Baselines for MP-CLF")

    # Required
    # Optional
    parser.add_argument(
        "--eval-start",
        action="store_true",
        default=False,
        help="Whether to evaluate the model before training or not.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=False,
        default="",
        help="Path to checkpoint with configuration.",
    )
    parser.add_argument(
        "-d",
        "--test_dataset",
        type=str,
        required=False,
        default="jordyvl/rvl_cdip_100_examples_per_class",
        help="Path to yml file with dataset configuration.",
    )
    parser.add_argument(
        "-l",
        "--labelset",
        type=str,
        required=False,
        default="test",
        choices=["train", "validation", "test"],
        help="Choose to test on which labelset of labels",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        default=False,
        help="Boolean to overwrite data-parallel arg in config parallelize the execution.",
    )
    parser.add_argument(
        "--exit_threshold",
        default=-1,
        type=float,
        help="Threshold for exiting early. If -1, no early exit is used.",
    )
    parser.add_argument(
        "--inference_strategy",
        default="max_confidence",
        type=str,
        help="Override criterion used for exiting",
    )
    parser.add_argument(
        "--benchmark_OCR",
        action="store_true",
        default=False,
        help="Boolean to simulate the cost of OCR (timing)",
    )
    parser.add_argument(
        "--print_freq",
        default=50,
        type=int,
        help="print logs every print_freq iterations",
    )
    parser.add_argument(
        "--plot_exits",
        action="store_true",
        default=False,
        help="plot",
    )
    parser.add_argument(
        "--downsampling",
        default=0,
        type=int,
        help="downsample the dataset to this number of samples [mainly for debugging]",
    )
    parser.add_argument(
        "--calibrate", default=False, help="calibrate logits before testing"
    )
    parser.add_argument(
        "--full_test",
        default=False,
        help="iterate over all thresholds",
    )
    parser.add_argument(
        "--step",
        default=0.1,
        type=float,
        help="threshold step for full test",
    )
    parser.add_argument(
        "--exit_policy",
        default=0.1,
        type=str,
        help="exit ploicy",
    )
    parser.add_argument(
        "--epsilon",
        default=0.1,
        type=float,
        help="epsilon for accuracy calibration heuristic",
    )
    args = parser.parse_known_args()[0]
    _config.update(args.__dict__)

    return _config


def nameit(config):
    if config["calibrate"]:
        return f"{config.get('dataset', config.get('test_dataset'))}-{config.get('checkpoint','model')}-calibrated"
    return f"{config.get('dataset', config.get('test_dataset'))}-{config.get('checkpoint','model')}"


def init_wandb(config):
    username = os.environ.get("WANDB_USERNAME")
    run = wandb.init(
        project="Beyond-Document-Page-Classification-src_EE",
        job_type="train",
        entity=username,
        name=nameit(config),
        config=config,
        reinit=True,
    )
    return run


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def check_gpu(device):
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


def process_label_ids(batch, remapper, label_column="label"):
    batch[label_column] = [remapper[label_id] for label_id in batch[label_column]]
    return batch


def build_dataset(config, split, processor=None):
    from data.RVL_CDIP import RVL_CDIP, RVL_CDIP_IO, Tobacco3482

    cache_dir = (
        "/mnt/lerna/data/HFcache" if os.path.exists("/mnt/lerna/data/HFcache") else None
    )

    # Default RVL_CDIP
    if config["dataset"].lower() == "rvl_cdip":
        data = load_dataset(config["dataset"], split=split, cache_dir=cache_dir)
        if split == "test":
            data = data.select([i for i in range(len(data)) if i != 33669])

    ## with OCR
    # self.config.data_dir
    # oG: jordyvl/rvl-cdip_easyOCR
    if config["dataset"] == "jordyvl/rvl_cdip_easyocr":
        data = load_dataset(
            "jordyvl/rvl_cdip_easyocr", split=split, cache_dir=cache_dir
        )
        # if split == "test":  # how would we know to find it?
        #     data = data.select((i for i in range(len(data)) if i != 33669))

    # NA, N, O
    if config["dataset"] == "jordyvl/RVL-CDIP-N":
        data = load_dataset(config["dataset"], split=split, cache_dir=cache_dir)
        label2idx = {
            label.replace(" ", "_"): i for label, i in config["model_label2idx"].items()
        }
        data_idx2label = dict(enumerate(data.features["label"].names))
        data_label2idx = {
            label: i for i, label in enumerate(data.features["label"].names)
        }
        model_idx2label = dict(zip(label2idx.values(), label2idx.keys()))
        diff = [
            i
            for i in range(len(data_idx2label))
            if data_idx2label[i] != model_idx2label[i]
        ]
        if diff:
            remapper = {}
            for k, v in label2idx.items():
                if k in data_label2idx:
                    remapper[data_label2idx[k]] = v
            new_features = Features(
                {
                    **{k: v for k, v in data.features.items() if k != "label"},
                    "label": ClassLabel(
                        num_classes=len(label2idx), names=list(label2idx.keys())
                    ),
                }
            )

            data = data.map(
                lambda example: process_label_ids(example, remapper),
                features=new_features,
                batched=True,
                batch_size=100,
                desc="Aligning the labels",
            )

    ## PDF versions

    if config["dataset"] == "maveriq/tobacco3482":
        """
        In our setting, we random sampled three subsets to be used for train,
        validation and test, fixing their cardinality to 800, 200, and 2482 respectively, as in [8, 19].
        [8:https://arxiv.org/abs/1907.06370]
        """
        data = load_dataset(config["dataset"], cache_dir=cache_dir)
        # Split sizes
        train_size = 800
        val_size = 200
        test_size = 2482

        # Perform the train-validation-test split
        train_dataset = data["train"].shuffle(seed=42).select(range(train_size))
        val_dataset = (
            data["train"]
            .shuffle(seed=42)
            .select(range(train_size, train_size + val_size))
        )
        test_dataset = (
            data["train"]
            .shuffle(seed=42)
            .select(range(train_size + val_size, train_size + val_size + test_size))
        )

        if split == "train":
            data = train_dataset
        elif split == "validation":
            data = val_dataset
        elif split == "test":
            data = test_dataset

    if config["dataset"] == "rvl-cdip_single_10":
        split = "test" if split == "validation" else split
        data = load_dataset(
            "nielsr/rvl_cdip_10_examples_per_class", split=split, cache_dir=cache_dir
        )
    if config["dataset"] == "jordyvl/rvl_cdip_100_examples_per_class":
        data = load_dataset(
            "jordyvl/rvl_cdip_100_examples_per_class", split=split, cache_dir=cache_dir
        )

    if "rvl" in config["dataset"].lower() or "tobacco" in config["dataset"].lower():
        if config["dataset"] == "jordyvl/rvl_cdip_easyocr":
            loader = RVL_CDIP_IO
        elif config["dataset"] == "maveriq/tobacco3482":
            loader = Tobacco3482
        else:
            loader = RVL_CDIP

        # data = data.select([0,1,2,3]) #TODO: evil hack to make testing faster
        dataset = loader(
            data,
            split,
            use_images=config["use_images"],
            get_raw_ocr_data=config["get_raw_ocr_data"],
            processor=processor,
            forward_signature=config["forward_signature"],
        )

    # for multipage:
    ## https://github.com/QuickSign/ocrized-text-dataset/blob/master/to_text.py
    return dataset


def build_model(config):
    def load_HF_config(config):
        hf_model_config = AutoConfig.from_pretrained(config["model_weights"])
        hf_model_config = update_config(hf_model_config, config)
        return hf_model_config

    def update_config(hf_model_config, config, key="EE_config"):
        hf_model_config.update({key: config})

        ##based on data make updates
        if "rvl" in config["dataset"].lower():  # and key == "EE_config"
            from data.RVL_CDIP import RVL_CDIP

            hf_model_config.num_labels = 16
            hf_model_config.id2label = RVL_CDIP.id2label
            hf_model_config.label2id = RVL_CDIP.label2id

        elif "tobacco" in config["dataset"].lower() and key == "EE_config":
            from data.RVL_CDIP import Tobacco3482

            hf_model_config.num_labels = 10
            hf_model_config.id2label = Tobacco3482.id2label
            hf_model_config.label2id = Tobacco3482.label2id

        # TODO: add more datasets here
        return hf_model_config

    hf_model_config = None
    if config["checkpoint"]:  # load config from model
        hf_model_config = AutoConfig.from_pretrained(config["model_weights"])
        # means it already has EE config, so update with current config
        hf_model_config.EE_config.update(config)
        config = hf_model_config.EE_config  # override config without model config
        config["train_dataset"] = config["dataset"]  # override dataset
        config["dataset"] = config["test_dataset"]  # override dataset

    if config["model"] == "EElayoutlmv3":
        if hf_model_config is None:
            config["model_weights"] = "microsoft/layoutlmv3-base"
            config["use_images"] = True
            config["get_raw_ocr_data"] = True
            hf_model_config = load_HF_config(config)

        from models.LayoutLMv3 import LayoutLMv3EEForSequenceClassification

        model = LayoutLMv3EEForSequenceClassification.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
        )
        print("Override warning: using default processor, assuming nothing changed")
        model.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

    if config["model"] == "LTElayoutlmv3":
        if hf_model_config is None:
            config["model_weights"] = "microsoft/layoutlmv3-base"
            config["use_images"] = True
            config["get_raw_ocr_data"] = True
            hf_model_config = load_HF_config(config)

        from models.LayoutLMv3LTE import LayoutLMv3LTEForSequenceClassification

        model = LayoutLMv3LTEForSequenceClassification.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
        )
        print("Override warning: using default processor, assuming nothing changed")
        model.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

    if "dit" in config["model"].lower():
        # https://github.com/microsoft/unilm/blob/master/dit/classification/README.md
        if hf_model_config is None:  # which means starting from a saved checkpoint
            if config["model"] == "dit":
                config["model_weights"] = "microsoft/dit-base"
                ignore_mismatched_sizes = False

            else:
                config["model_weights"] = "microsoft/dit-base-finetuned-rvlcdip"
                ignore_mismatched_sizes = True

            config["use_images"] = True
            config["get_raw_ocr_data"] = False
            hf_model_config = load_HF_config(config)

        model = AutoModelForImageClassification.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        model.processor = AutoProcessor.from_pretrained(config["model_weights"])

    if config["model"].lower() == "layoutlmv2":
        if hf_model_config is None:
            config["model_weights"] = "microsoft/layoutlmv2-base-uncased"
            config["use_images"] = True
            config["get_raw_ocr_data"] = True
            hf_model_config = load_HF_config(config)

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
        )
        model.processor = AutoProcessor.from_pretrained(config["model_weights"])

    if config["model"].lower() == "layoutlmv3":
        if hf_model_config is None:
            config["model_weights"] = "microsoft/layoutlmv3-base"
            config["use_images"] = True
            config["get_raw_ocr_data"] = True
            hf_model_config = load_HF_config(config)

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
        )

        if not "easyocr" in config["dataset"]:
            print("Warning: using default processor, assuming nothing changed")
            model.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
        else:
            model.processor = AutoProcessor.from_pretrained(config["model_weights"])

    if "bert" in config["model"].lower():
        if hf_model_config is None:
            config["model_weights"] = "bert-base-cased"
            config["use_images"] = False
            config["get_raw_ocr_data"] = True
            hf_model_config = load_HF_config(config)

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
        )
        model.processor = AutoTokenizer.from_pretrained(config["model_weights"])

    if config["model"].lower() == "pix2struct":
        if hf_model_config is None:
            config["model_weights"] = "google/pix2struct-base"
            config["use_images"] = True
            config["get_raw_ocr_data"] = False
            hf_model_config = load_HF_config(config)

        from models.Pix2Struct import Pix2Struct

        model = Pix2Struct.from_pretrained(
            config["model_weights"],
            config=hf_model_config,
        )
        raise NotImplementedError("Pix2Struct is not implemented yet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["dataset"] == "jordyvl/RVL-CDIP-N":
        config["model_label2idx"] = model.config.label2id

    model = model.to(device)
    return model, config
