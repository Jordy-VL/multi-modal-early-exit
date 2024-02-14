import json
import torch

from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import (
    ModelOutput,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from fvcore.nn import FlopCountAnalysis, parameter_count


"""notes:
    override main layoutlmv3 class to include early exit hyperparameterization
    
    possible exits:
        1. average embedding of all tokens [text] (+ pos?)
        2. average embedding of all patches [visual] + pos + pos2D
        3. concat of 1 and 2 -> non-contextualized representation
        
        4. binary gates on some encoder layers: 1, 4
            4.1 if binary gate confidence > threshold, exit
             
    -> start with 1
    
    Each exit is a linear layer with unique identifier and predefined threshold
    It also has its own loss function?
    
    Intermediate classifiers:
        A) Binary gate and pass on to final classifier
        B) Off-ramps
    
    Options for training:
    
    a) Train by passing through each exit and passing output
    OR
    b) Train by exiting already when CSF is high enough
    
    Coding inspiration: 
    # https://github.com/georgian-io/Multimodal-Toolkit/blob/master/notebooks/text_w_tabular_classification.ipynb
    # https://github.com/huggingface/setfit -> could use for two-stage training
    # https://github.com/georgian-io/Multimodal-Toolkit/blob/master/multimodal_transformers/model/tabular_transformers.py
    # https://github.com/huggingface/transformers/blob/main/examples/research_projects/deebert/src/modeling_highway_bert.py
"""

import operator
from enum import Enum


class Enhnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self):
        return f"<{self.__class__.__name__}-{self._name_}>"

    def _generate_next_value_(self, start, count, last_values) -> str:
        return self.lower()

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one from {cls.all()}"
        )

    @classmethod
    def all(cls) -> List[str]:
        return list(map(lambda c: c.value, cls))


class EarlyExitStrategy(Enhnum):
    JOINT = "joint"  # end-to-end combine all losses

    # https://proceedings.neurips.cc/paper/2020/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf [PABEE]
    JOINT_W_AVG = "joint_weighted_avg"  # equal weight for all exits -> irrealistic since exit imbalance
    ## -> gamma=0.5

    JOINT_W = "joint_weighted"  # could be weighted, higher penalty for earlier exits]
    """Weighting
    
    linearly decaying schedule
    ...
    
    #https://openaccess.thecvf.com/content/CVPR2021/papers/
    # Ghodrati_FrameExit_Conditional_Early_Exiting_for_Efficient_Video_Recognition_CVPR_2021_paper.pdf [FrameExit]
    exponentially decaying schedule
    \lambda (\sum_{l=1}^{E} \exp(\frac{l}{2}}) + final_layer_loss
    
    """
    # https://arxiv.org/pdf/2004.12993.pdf [DeeBERT]
    TWO_STAGE = "two-stage"  # train full with freezing exit layers, then freeze all but exit layers -> manual

    # https://aclanthology.org/2021.eacl-main.8.pdf [BERTxit]
    ALTERNATING = "alternating"  # train all in even epochs, exits only in odd epochs; seems ad-hoc

    # https://www.bmvc2021-virtualconference.com/assets/papers/1225.pdf [MultiExitViT]
    LAYERWISE = "layerwise"  # train upto and including first exit; freeze layers before first exit; train upto and including second exit; freeze layers before second exit; etc.

    # https://github.com/romebert/RomeBERT [RomeBert]
    ONE_STAGE_SUBGRAPHS = "one_stage_subgraphs"  # train single stage with loss updates per exit [retain graph]

    # Custom
    ONE_STAGE_SUBGRAPHS_WEIGHTED = "one_stage_subgraphs_weighted"  # train single stage with weighted loss updates per exit [retain graph]; enter gamma
    ONE_STAGE_SUBGRAPHS_ENTROPYREG = "one_stage_subgraphs_entropyreg"  # train single stage with weighted loss updates per exit [retain graph] and entropy reg for branches
    ONE_STAGE_SUBGRAPHS_WEIGHTED_ENTROPYREG = "one_stage_subgraphs_weighted_entropyreg"  # train single stage with weighted loss updates per exit [retain graph] and entropy reg for branches

    ## two stage variants
    TWO_STAGE_SUBGRAPHS = "two_stage_subgraphs"  # train single stage with loss updates per exit [retain graph]

    # Custom
    TWO_STAGE_SUBGRAPHS_WEIGHTED = "two_stage_subgraphs_weighted"  # train single stage with weighted loss updates per exit [retain graph]; enter gamma
    TWO_STAGE_SUBGRAPHS_ENTROPYREG = "two_stage_subgraphs_entropyreg"  # train single stage with weighted loss updates per exit [retain graph] and entropy reg for branches
    TWO_STAGE_SUBGRAPHS_WEIGHTED_ENTROPYREG = "two_stage_subgraphs_weighted_entropyreg"  # train single stage with weighted loss updates per exit [retain graph] and entropy reg for branches


class EarlyExitInference(Enhnum):
    # https://arxiv.org/abs/1709.01686 [BranchyNet]
    MAX_CONFIDENCE = "max_confidence"  # inference stops when confidence of intermediate predictions is above threshold

    # https://aclanthology.org/2021.eacl-main.8.pdf [BERTxit]
    ENTROPY = "entropy"  # inference stops when entropy of intermediate predictions is below threshold

    # https://proceedings.neurips.cc/paper/2020/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf [PABEE]
    PATIENCE = "patience"  # inference stops when intermediate predictions remain unchanged for $t$ exit times
    # less interesting due to multiple layers consistency and additional patience hyperparameter

    # https://aclanthology.org/2021.eacl-main.8.pdf [BERTxit]
    LTE = "lte"  # seemingly like a gate with a separate threshold for each exit ~ regression like

    def get_function(self) -> Callable:
        if self == EarlyExitInference.MAX_CONFIDENCE:
            return max_confidence
        elif self == EarlyExitInference.ENTROPY:
            return entropy
        elif self == EarlyExitInference.LTE:
            return lte
        raise NotImplementedError(f"{self} not implemented")

    def get_sign(self) -> Callable:
        if self == EarlyExitInference.MAX_CONFIDENCE:
            return operator.gt  # higher is better
        elif self == EarlyExitInference.ENTROPY:
            return operator.lt  # lower is better
        elif self == EarlyExitInference.LTE:
            return operator.lt  # lower is better
        raise NotImplementedError(f"{self} not implemented")


def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B / A


def max_confidence(x):
    # x: torch.Tensor, logits BEFORE softmax
    x = torch.nn.functional.softmax(x, dim=1)
    return torch.max(x, dim=1)[0]


def lte(x):
    # x: torch.Tensor, logits BEFORE softmax
    raise NotImplementedError("TODO: implement lte")


class EarlyExitHead(Enhnum):
    # https://aclanthology.org/2021.eacl-main.8.pdf [BERTxit] -> regression
    GATE = "gate"  # binary f_e: X -> {0,1} [MSE loss/binaryCE]
    RAMP = "ramp"  # linear f_e: X -> [K] [CE loss] -> assumed for encoder
    EMBEXIT = "embexit"  # linear f: X -> [K] [CE loss] -> assumed before encoder


class ExitConfig:
    def __init__(self, **kwargs):
        # super().__init__(config=kwargs)
        self.training_strategy = EarlyExitStrategy(
            kwargs.get("training_strategy", "joint_weighted_avg")
        )
        self.inference_strategy = EarlyExitInference(
            kwargs.get("inference_strategy", "max_confidence")
        )
        # TODO: define thresholds per exit - which can be overridden when loading the model
        self.global_threshold = kwargs.get("global_threshold", 0.9)
        self.exits = kwargs.get(
            "exits", ["text_avg", "vision_avg", 1, 4, 8]
        )  # "text_visual_concat"

        # TODO: fixed category for all exits
        self.encoder_layer_strategy = EarlyExitHead(
            kwargs.get("encoder_layer_strategy", "ramp")
        )
        self.exit_head_num_layers = kwargs.get("exit_head_num_layers", 2)
        # TODO: make number of layers 1 for gates?


@dataclass
class EEModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        exit_states (`tuple(torch.FloatTensor)` of shape `(batch_size, num_exits, K)`, *optional*):
            Classification scores (before SoftMax) relative to exits
        gate_inputs (`tuple(torch.FloatTensor)` of shape `(batch_size, num_exits, d)`, *optional*):
            Intermediate representations that should be passed on to the final classifier
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    exit_states: Optional[Tuple[torch.FloatTensor]] = None
    gate_inputs: Optional[Tuple[torch.FloatTensor]] = None
    lte_output: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class EESequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of early exit classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        exit_losses (`tuple(torch.FloatTensor)` of shape `(batch_size, num_exits)`, *optional*, returned when `labels` is provided and `exits` are used):
            Tuple of `torch.FloatTensor` (one for each exit layer)

            Classification loss relative to exits logits
        exit_states (`tuple(torch.FloatTensor)` of shape `(batch_size, num_exits, K)`, *optional*):
            Classification scores (before SoftMax) relative to exits
        gated_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_exits, K)`, *optional*):
            Intermediate representations that have passed through the final classifier
    """

    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    exit_losses: Optional[Tuple[torch.FloatTensor]] = None  # loss per exit
    exit_criteria: Optional[
        Tuple[torch.FloatTensor]
    ] = None  # entropy/confidences per exit (before softmax)
    exit_states: Optional[Tuple[torch.FloatTensor]] = None
    gated_logits: Optional[Tuple[torch.FloatTensor]] = None
    lte_output: Optional[Tuple[torch.FloatTensor]] = None


class LogitNormLoss(torch.nn.Module):
    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return torch.nn.functional.cross_entropy(logit_norm, target)


class EETrainingArguments(TrainingArguments):
    """Extending original TrainingArguments to include temperature and alpha as hyperparameters"""

    def __init__(
        self, *args, alpha=1, temperature=1, gamma=1, training_strategy="", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature  # -> self supervised distillation?
        self.gamma = gamma  # only relevant for NKD loss -> best to set temperature to 1 -> self supervised distillation?
        self.training_strategy = training_strategy


class EETrainer(Trainer):
    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            exit_losses = outputs["exit_losses"]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            exit_losses = tuple(map(lambda x: x.mean(), exit_losses))

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            exit_losses = tuple(
                map(lambda x: x / self.args.gradient_accumulation_steps, exit_losses)
            )

        exit_named_params, exit_loss_weights, exit_branch_params = params_per_exit(
            model
        )

        # TODO: might have to check what to do with gated logits and losses?

        final_loss = 0
        for j, exit_loss in enumerate(exit_losses):
            if "weighted" in self.args.training_strategy:
                exit_loss = exit_loss * exit_loss_weights[j]
            if self.args.gamma != 0:  # gamma gives more weight to the exits
                exit_loss = exit_loss * (
                    self.args.gamma / len(exit_losses)
                )  # each loss takes a part of gamma
            exit_loss.backward(retain_graph=True)
            final_loss += exit_loss.detach()
        if self.args.gamma != 0:  # gamma gives more weight to the exits
            loss = loss * (1 - self.args.gamma)  # the main loss takes the rest of gamma
        loss.backward()  # final call through the graph, which can be released
        final_loss += loss.detach()
        if "entropyreg" in self.args.training_strategy:
            # exit_criteria are undefined when self.training
            exit_criteria = torch.zeros(
                len(exit_losses) + 1, device=exit_losses[0].device
            )
            for j, exit_state in enumerate(outputs["exit_states"]):
                exit_criteria[j] = entropy(exit_state).mean()  # average over BS
            exit_criteria[-1] = entropy(outputs["logits"]).mean()  # average over BS

            norm_entropy = torch.nn.functional.softmax(exit_criteria, dim=0) * (
                len(exit_criteria)
            )
            norm_entropy[norm_entropy > 1.0] = 1.0
            neg_norm_entropy = 1 - norm_entropy

            # https://github.com/vgaraujov/ESP-CL/blob/main/NLP/train.py
            for j, criterion in enumerate(neg_norm_entropy):
                for name, p in model.named_parameters():
                    if name in exit_branch_params[j]:
                        p.grad *= criterion
        return final_loss


def count_named_parameters(model, modules=None):
    found = [name for name, _ in model.named_parameters() if name in modules]
    total_params = sum(
        [
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in modules
        ]
    )
    try:
        assert set(found) == set(modules)
    except AssertionError:
        difference = set([mod for mod in modules if not "exit" in mod]) - set(found)
        # if difference:
        #     print(f"Not all modules [param count] found: {difference}")
    return total_params


def flops_named_parameters(flops_named_params, modules=None):
    found = [name for name in flops_named_params if name in modules]
    total_flops = sum([flops_named_params[module] for module in modules])
    try:
        assert set(found) == set(modules)
    except AssertionError:
        difference = set([mod for mod in modules if not "exit" in mod]) - set(found)
        # if difference:
        #     print(f"Not all modules [flops] found: {difference}")
    return total_flops


def filter_encoder_exits(lst, encoder_counter):
    return [
        item
        for item in lst
        if (
            item.startswith("encoder.early_exits.")
            and int(item.split(".")[-3]) <= encoder_counter
        )
        or not item.startswith("encoder.early_exits.")
    ]


def translate_exit_identifier(exit, encoder_counter=0):
    if isinstance(exit, int):
        return f"encoder.early_exits.{encoder_counter}"
    elif "avg" in exit:
        return exit.replace("avg", "exit") + "_embeddings"
    elif "concat" in exit:
        return "concat_exit_embeddings"
    return "classifier"


def params_per_exit(model, beta=1):
    """
    Returns a dictionary of exit names and their corresponding
    1) named parameters per exit
    2) parameters count
    3) branch named parameters
    """
    # translate to counter
    with open("models/EELayoutLM_exit_named_parameters-wotherexits.json", "r") as f:
        exit_named_params = json.load(f)

    # module_param_counts = parameter_count(model)  # FULL COUNT

    ## TO BE TESTED 03/07/2023
    if not hasattr(model, "config"):  # distributeddataparallel
        model = model.module

    exit_config = model.config.exit_config
    exit_param_count = {}
    exit_branch_params = {}  # subset of exit_param_count
    encoder_counter = 0

    for exit_index, exit in enumerate(exit_config["exits"]):
        # LEGACY swap for old exits code where first text, then vision exits
        if exit == "text_avg" and exit_index == 0:
            exit = "vision_avg"
            exit_index = 1
        elif exit == "vision_avg" and exit_index == 1:
            exit = "text_avg"
            exit_index = 0

        modules = exit_named_params[str(exit)]
        # remove any modules with encoder index higher than counter f"encoder.early_exits.{encoder_counter}"
        if isinstance(exit, int):
            modules = exit_named_params[str(exit)] = filter_encoder_exits(
                modules, encoder_counter
            )

        exit_name = translate_exit_identifier(exit, encoder_counter)
        exit_branch_params[exit_index] = [mod for mod in modules if exit_name in mod]

        # need to filter for encoder if not all defined
        param_count = count_named_parameters(
            model, modules=modules
        )  # could also use module_param_counts

        exit_param_count[exit_index] = beta / (
            param_count
        )  # as a percentage of total  / module_param_counts[""]

        if isinstance(exit, int):  # encoder indices update relative to exit counter
            encoder_counter += 1

    exit_branch_params[len(exit_config["exits"])] = [
        mod for mod, _ in model.named_parameters() if "classifier" in mod
    ]

    exit_loss_weights = {
        k: (v / sum(exit_param_count.values())) for k, v in exit_param_count.items()
    }  # locally normalized
    return exit_named_params, exit_loss_weights, exit_branch_params
