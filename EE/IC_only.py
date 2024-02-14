#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2023 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
from tqdm import tqdm
import numpy as np
import torch
import evaluate
import inspect

from data import collate_fn, CustomTrainer

from configs import (
    ex,
    DEFAULT_EXPERIMENT,
    parse_args,
    seed_everything,
    build_dataset,
    build_model,
)

from transformers import TrainingArguments, Trainer, AdamW
from models.EE_modules import EETrainer, EETrainingArguments


SAVEROOT = "./save"
# from arkham.default_config import MODELROOT, SAVEROOT, DATAROOT
# from sklearn.metrics import accuracy_score, f1_score


def debug_step(trainer, config):
    for batch in trainer.get_train_dataloader():
        break

    batch = {k: v.to(config["device"]) for k, v in batch.items()}
    trainer.create_optimizer()

    for _ in range(5):
        outputs = trainer.model(**batch)
        loss = outputs.loss
        print(loss)

        # torch.nn.CrossEntropyLoss(ignore_index=1)(outputs.logits, batch['labels']

        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    with torch.no_grad():
        outputs = trainer.model(**batch)
    preds = outputs.logits
    accuracy = (preds.argmax(-1) == batch["labels"]).float().mean()
    print(accuracy)


@ex.automain
def main(_config, _run, _seed):
    config = parse_args(_config)

    seed_everything(config["seed"])

    # build model and update config if needed
    model, config = build_model(config)

    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    config["forward_signature"] = signature_columns
    print(signature_columns)

    # dataset
    train_dataset = build_dataset(config, "train", processor=model.processor)
    val_dataset = build_dataset(config, "validation", processor=model.processor)
    test_dataset = build_dataset(config, "test", processor=model.processor)

    # TODO: could check forward signature
    # dataset.set_format(type='torch', columns=signature_columns)

    ct = datetime.datetime.now()

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        if isinstance(eval_preds[0], tuple):
            logits, exit_losses, exit_criteria, exit_states, gated_logits = eval_preds[
                0
            ]
            labels = eval_preds[1]
            # np.hstack(exit_losses)
        else:
            logits, labels = eval_preds  # output of forward, right?
        predictions = np.argmax(logits, axis=-1)
        # accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        # recall = recall_score(y_true=labels, y_pred=predictions)
        # precision = precision_score(y_true=labels, y_pred=predictions)
        # f1 = f1_score(y_true=labels, y_pred=predictions)
        # Zreturn {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        results = metric.compute(predictions=predictions, references=labels)

        try:
            if isinstance(
                eval_preds[0], tuple
            ):  # could do the same for exit_losses and exit_criteria
                for exit_id in range(
                    len(exit_states)
                ):  # do we have that for each sample?
                    exit_logits, exit_crits = exit_states[exit_id]
                    if (
                        gated_logits is not None and len(gated_logits) > 0
                    ):  # missing switch
                        exit_logits = gated_logits[exit_id]

                    else:
                        exit_logits = exit_states[exit_id][0]

                    exit_predictions = np.argmax(exit_logits, axis=-1)
                    results[f"exit_{exit_id}_accuracy"] = metric.compute(
                        predictions=exit_predictions, references=labels
                    )["accuracy"]
        except Exception as e:
            print(e)
        return results

    model_name = (
        f"{config['model']}_{config['dataset'].replace('/','_')}_{str(ct)[:10]}"
    )
    experiment_name = (
        model_name + "_" + _run.experiment_info["name"]
        if _run.experiment_info["name"] != DEFAULT_EXPERIMENT
        else model_name
    )
    # check_gpu(model.model.device)

    # TODO: check model type if seq2seq, then use https://github.com/huggingface/transformers/issues/15313

    args = EETrainingArguments(
        training_strategy=config["training_strategy"],
        gamma=config["gamma"],
        alpha=config["alpha"],
        temperature=config["temperature"],
        output_dir=os.path.join(SAVEROOT, experiment_name),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_total_limit=3,
        group_by_length=hasattr(model.processor, "tokenizer"),
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        run_name=experiment_name,
        hub_model_id=experiment_name  # this was the missing argument
        # remove_unused_columns=True,
        # eval_accumulation_steps=1,  # logits on GPU
    )

    # put the model in training mode
    model.train()

    # alternatively
    # trainer.init_git_repo(at_init=True)

    trainer_class = EETrainer if "one_stage" in config["training_strategy"] else Trainer

    trainer = trainer_class(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.processor.tokenizer
        if hasattr(model.processor, "tokenizer")
        else None,  # model.processor,
        compute_metrics=compute_metrics,
    )

    if "two_stage" in config["training_strategy"]:
        trainer_class = EETrainer
        trainer = trainer_class(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=model.processor.tokenizer
            if hasattr(model.processor, "tokenizer")
            else None,  # model.processor,
            compute_metrics=compute_metrics,
        )

        print("Freezing backbone weights")
        with open("model_parameters.txt", "w") as f:
            for name, param in model.named_parameters():
                if "exit" not in name and "classifier" not in name:
                    f.write(str(name) + " ---> " + str(param.requires_grad) + "\n")
                    param.requires_grad = False

    # debug_step(trainer, config
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        print(e)

    # trainer.save_model(os.path.join(SAVEROOT, experiment_name))
    trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    trainer.push_to_hub("Saving best model to hub")  # using *self.args.hub_model_id*.

    # try:
    # except Exception as e:
    #     print(e)
    #     print("Trying again anyway")
    #     trainer.push_to_hub()
