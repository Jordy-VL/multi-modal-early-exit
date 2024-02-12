import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoConfig,
    LayoutLMv3ForSequenceClassification,
)

from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3Model,
    # LayoutLMv3TextEmbeddings,
    # LayoutLMv3PatchEmbeddings,
    # LayoutLMv3Layer,
    LayoutLMv3Encoder,
    LayoutLMv3Encoder,
)

try:
    from .EE_modules import (
        EarlyExitHead,
        ExitConfig,
        EEModelOutput,
        EESequenceClassifierOutput,
    )
except:  # for testing
    from EE_modules import (
        EarlyExitHead,
        ExitConfig,
        EEModelOutput,
        EESequenceClassifierOutput,
    )


from typing import Optional, Tuple, Union, List

POSSIBLE_EXITS = ["vision_avg", "text_avg", "text_visual_concat"] + list(range(1, 13))


def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B / A


class EarlyExitException(Exception):
    def __init__(self, payload, exit_layer):
        self.payload = payload
        self.exit_layer = exit_layer
        # print(f"Early exit at layer {exit_layer} with payload {payload}")


class LayoutLMv3Exit(nn.Module):
    """
    Head for classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, input_dim, identifier):
        super().__init__()

        exit_config = config.exit_config
        if type(exit_config) is dict:  # when loading from saved model or initializing
            exit_config = ExitConfig(**exit_config)
        # else:
        #     self.config.exit_config = exit_config.__dict__ #only when setting it as attribute

        if exit_config.exit_head_num_layers == 2:
            self.dense = nn.Linear(
                input_dim, input_dim
            )  # consistency with original Roberta
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        head_type = exit_config.encoder_layer_strategy
        self.identifier = f"{identifier}_{head_type}"
        output_dim = config.num_labels if head_type == EarlyExitHead.RAMP else 2
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if hasattr(self, "dense"):  # 2 linear layers
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LayoutLMv3EncoderEE(LayoutLMv3Encoder):
    def __init__(self, config):
        super().__init__(config)
        possible_exits = []
        if isinstance(config.exit_config["exits"], str):
            for exit in config.exit_config["exits"].split(","):
                try:
                    # for encoders, exits are integers
                    possible_exits.append(int(exit))
                except:
                    # only for embedding exits
                    possible_exits.append(exit)
            config.exit_config["exits"] = possible_exits
        # self.config = config
        # self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        self.exit_encoder_layers = [
            i for i in config.exit_config["exits"] if isinstance(i, int)
        ]
        self.has_exits = len(self.exit_encoder_layers) > 0
        self.early_exits = nn.ModuleList(
            [
                LayoutLMv3Exit(config, config.hidden_size, f"encoder_exit_{i}")
                for i in self.exit_encoder_layers
            ]
        )
        self.apply_gating = (
            config.exit_config["encoder_layer_strategy"] == "gate"
        )  # TODO: replace with layer-specific config

        if self.has_exits:
            self.exit_criterion = self.config.exit_config[
                "inference_strategy"
            ].get_function()
            self.threshold_sign = self.config.exit_config[
                "inference_strategy"
            ].get_sign()  # if positive: > threshold, if negative: < threshold
            self.exit_threshold = self.config.exit_config[
                "global_threshold"
            ]  # TODO: could do per exit

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]
        # TODO: another init weights?

        self.num_layers = len(self.exit_encoder_layers)
        self.use_lte = self.config.EE_config.get("use_lte", False)

    def init_lte(self, config):
        self.lte_th = [0.005] * config.num_hidden_layers
        self.lte_classifier = nn.Linear(config.hidden_size, 1)
        self.lte_activation = nn.Sigmoid()

    def enable_lte(self):
        if self.exit_threshold is not None:
            self.lte_th = [float(self.exit_threshold)] * self.num_layers

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        patch_height=None,
        patch_width=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_early_exits = () if self.has_exits else None
        all_gate_inputs = () if self.apply_gating else None
        lte_outputs = []

        rel_pos = (
            self._cal_1d_pos_emb(hidden_states, position_ids)
            if self.has_relative_attention_bias
            else None
        )
        rel_2d_pos = (
            self._cal_2d_pos_emb(hidden_states, bbox)
            if self.has_spatial_attention_bias
            else None
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                        # return module(*inputs, past_key_value, output_attentions, rel_pos, rel_2d_pos)
                        # The above line will cause error:
                        # RuntimeError: Trying to backward through the graph a second time
                        # (or directly access saved tensors after they have already been freed).

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if (i + 1) in self.exit_encoder_layers:  # TODO: CONTINUE FROM HERE
                current_exit = self.early_exits[
                    len(all_early_exits)
                ]  # proxy to find layer number
                exit_input = hidden_states[:, 0, :]
                exit_output = current_exit(exit_input)  # only predict for CLS token

                highway_entropy = entropy(exit_output)
                # the block for lte
                if self.use_lte:
                    # lte_input = exit_input  # hidden states --------> TODO ASK about this
                    lte_input = exit_input
                    lte_output = self.lte_activation(
                        self.lte_classifier(lte_input)
                    ).squeeze()
                    lte_outputs.append(lte_output)

                if not self.training:
                    exit_crit = self.exit_criterion(
                        exit_output
                    )  # calls entropy or max confidence
                    exit_output = (exit_output, exit_crit)

                all_early_exits = all_early_exits + (exit_output,)

                if self.apply_gating:
                    all_gate_inputs = all_gate_inputs + (exit_input,)

                if not self.training:
                    if self.use_lte:
                        self.enable_lte()
                        if i + 1 < self.num_layers:
                            if (self.use_lte and lte_output < self.lte_th[i]) or (
                                not self.use_lte
                                and highway_entropy < self.early_exit_entropy[i]
                            ):
                                output = EEModelOutput(
                                    last_hidden_state=exit_output,
                                    hidden_states=all_hidden_states,
                                    attentions=all_self_attentions,
                                    exit_states=all_early_exits,  # up until this exit point
                                    gate_inputs=all_gate_inputs,
                                    lte_output=lte_outputs,
                                )
                                raise EarlyExitException(
                                    output, current_exit.identifier
                                )

                    # else:
                    # i
                    # f self.threshold_sign(exit_crit, self.exit_threshold):
                    #     output = EEModelOutput(
                    #     last_hidden_state=exit_output,
                    #     hidden_states=all_hidden_states,
                    #     attentions=all_self_attentions,
                    #     exit_states=all_early_exits,  # up until this exit point
                    #     gate_inputs=all_gate_inputs,
                    #     lte_output=lte_outputs,
                    #     )
                    #     raise EarlyExitException(output, current_exit.identifier)

        # this adds the final layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return EEModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            exit_states=all_early_exits,
            gate_inputs=all_gate_inputs,
            lte_output=lte_outputs,
        )


class LayoutLMv3ModelEE(LayoutLMv3Model):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.has_exits = False
        self.apply_gating = (
            config.exit_config["encoder_layer_strategy"] == "gate"
        )  # TODO: replace with layer-specific config

        # add exits here related to embeddings
        if "vision_avg" in config.exit_config["exits"]:
            self.vision_exit_embeddings = LayoutLMv3Exit(
                config,
                config.hidden_size,  # int((config.input_size**2) / (config.patch_size**2) + 1),
                "vision_avg",  # HW/P^2 + 1
            )
            self.has_exits = True

        if "text_avg" in config.exit_config["exits"]:
            self.text_exit_embeddings = LayoutLMv3Exit(
                config, config.hidden_size, "text_avg"
            )
            self.has_exits = True

        if "text_visual_concat" in config.exit_config["exits"]:
            self.concat_exit_embeddings = LayoutLMv3Exit(
                config,
                int(config.hidden_size),
                "text_visual_concat",  # TODO: not true, this is dynamic! input_shape (+padding?) + 197 (visual); so average it
            )
            self.has_exits = True

        if self.has_exits:
            self.exit_criterion = self.config.exit_config[
                "inference_strategy"
            ].get_function()
            self.threshold_sign = self.config.exit_config[
                "inference_strategy"
            ].get_sign()  # if positive: > threshold, if negative: < threshold
            self.exit_threshold = self.config.exit_config[
                "global_threshold"
            ]  # TODO: could do per exit

        self.encoder = LayoutLMv3EncoderEE(config)
        if self.encoder.use_lte:
            self.encoder.init_lte(config)
        self.init_weights()

    def forward_image(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EEModelOutput]:
        r"""
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        all_early_exits = () if self.has_exits else None
        all_gate_inputs = () if self.apply_gating else None
        lte_outputs = []

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds or pixel_values"
            )

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )
            if bbox is None:
                bbox = torch.zeros(
                    tuple(list(input_shape) + [4]), dtype=torch.long, device=device
                )

        ## inversed -> first vision [assumed to be cheaper?]
        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = int(
                pixel_values.shape[2] / self.config.patch_size
            ), int(pixel_values.shape[3] / self.config.patch_size)
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]),
                dtype=torch.long,
                device=device,
            )

            if (
                self.config.has_relative_attention_bias
                or self.config.has_spatial_attention_bias
            ):
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(
                        device, dtype=torch.long, batch_size=batch_size
                    )

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)

            if hasattr(self, "vision_exit_embeddings"):
                exit_input = visual_embeddings.mean(1)

                vision_exit = self.vision_exit_embeddings(
                    exit_input
                )  # BS x hidden_size

                if not self.training:  # append exit criterion
                    exit_crit = self.exit_criterion(
                        vision_exit
                    )  # calls entropy or max confidence
                    vision_exit = (
                        vision_exit,
                        exit_crit,
                    )

                all_early_exits = all_early_exits + (vision_exit,)
                if self.apply_gating:
                    all_gate_inputs = all_gate_inputs + (exit_input,)

                # if not self.training:
                #     if self.threshold_sign(exit_crit, self.exit_threshold):
                #         output = EEModelOutput(
                #             last_hidden_state=vision_exit,
                #             hidden_states=None,
                #             attentions=None,
                #             exit_states=all_early_exits,  # up until this exit point
                #             gate_inputs=all_gate_inputs,
                #         )
                #         raise EarlyExitException(
                #             output, self.vision_exit_embeddings.identifier
                #         )

        # fix for changing order of modalities
        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )
            if bbox is None:
                bbox = torch.zeros(
                    tuple(list(input_shape) + [4]), dtype=torch.long, device=device
                )

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            if hasattr(self, "text_exit_embeddings"):
                exit_input = embedding_output.mean(1)
                text_exit = self.text_exit_embeddings(exit_input)  # BS x hidden_size

                if not self.training:
                    exit_crit = self.exit_criterion(
                        text_exit
                    )  # calls entropy or max confidence
                    text_exit = (
                        text_exit,
                        exit_crit,
                    )  # logits, entropy? why not just logits?

                all_early_exits = all_early_exits + (text_exit,)
                if self.apply_gating:
                    all_gate_inputs = all_gate_inputs + (exit_input,)

                # if not self.training:
                #     if self.threshold_sign(exit_crit, self.exit_threshold):
                #         output = EEModelOutput(
                #             last_hidden_state=text_exit,  # TODO: if it is not a ramp, then it should pass the input to the gate, rather than the gate logits --> check if gate_inputs are there
                #             hidden_states=None,
                #             attentions=None,
                #             exit_states=all_early_exits,  # up until this exit point
                #             gate_inputs=all_gate_inputs,
                #         )
                #         raise EarlyExitException(
                #             output, self.text_exit_embeddings.identifier
                #         )

        if input_ids is not None or inputs_embeds is not None:
            embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            # TODO: have to do the same for the attention mask
            # extended_attention_mask
            attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            # TODO: for final bbox as well?
            # final_bbox
            final_bbox = torch.cat([bbox, visual_bbox], dim=1)

            # TODO for final position ids as well
            position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
            position_ids = position_ids.expand(input_shape)

            # might be invalid if not using input_ids or inputs_embeds
            final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        embedding_output = self.LayerNorm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        if pixel_values is not None:
            pass
        elif (
            self.config.has_relative_attention_bias
            or self.config.has_spatial_attention_bias
        ):
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        if hasattr(self, "concat_exit_embeddings"):
            exit_input = embedding_output.mean(1)

            joint_exit = self.concat_exit_embeddings(exit_input)  # BS x hidden_size

            if not self.training:
                exit_crit = self.exit_criterion(
                    joint_exit
                )  # calls entropy or max confidence
                joint_exit = (
                    joint_exit,
                    exit_crit,
                )  # logits, entropy? why not just logits?

            all_early_exits = all_early_exits + (joint_exit,)

            if self.encoder.use_lte:
                lte_input = exit_input  # hidden states --------> TODO ASK about this
                lte_output = self.encoder.lte_activation(
                    self.encoder.lte_classifier(lte_input)
                ).squeeze()
                lte_outputs.append(lte_output)

            if self.apply_gating:
                all_gate_inputs = all_gate_inputs + (exit_input,)

            # if not self.training:
            #     if self.threshold_sign(exit_crit, self.exit_threshold):
            #         output = EEModelOutput(
            #             last_hidden_state=joint_exit,
            #             hidden_states=None,
            #             attentions=None,
            #             exit_states=all_early_exits,  # up until this exit point
            #             gate_inputs=all_gate_inputs,
            #         )
            #         raise EarlyExitException(
            #             output, self.concat_exit_embeddings.identifier
            #         )

        # sequence_length = input_shape[1]

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # try except here?

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        if all_early_exits:
            all_early_exits = all_early_exits + encoder_outputs.exit_states
        if self.apply_gating:
            all_gate_inputs = all_gate_inputs + encoder_outputs.gate_inputs

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return EEModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            exit_states=all_early_exits,
            gate_inputs=all_gate_inputs,
            lte_output=lte_outputs + encoder_outputs.lte_output,
        )


# subclass huggingface LayoutLMv3 model
class LayoutLMv3EEForSequenceClassification(LayoutLMv3ForSequenceClassification):
    def __init__(self, hf_model_config, **kwargs):
        super().__init__(hf_model_config, **kwargs)

        # could init processor here
        self.processor = AutoProcessor.from_pretrained(
            self.config.EE_config["model_weights"],
            apply_ocr=self.config.EE_config.get("apply_ocr", False),
        )
        self.config.exit_config = ExitConfig(
            **self.config.EE_config
        ).__dict__  # will fill and check all applicable config values

        self.training_strategy = self.config.exit_config["training_strategy"]

        self.apply_gating = (
            self.config.exit_config["encoder_layer_strategy"] == "gate"
        )  # TODO: replace with layer-specific config

        self.use_lte = self.config.EE_config.get("use_lte", False)
        self.num_layers = len(
            [i for i in self.config.exit_config["exits"] if isinstance(i, int)]
        )
        # build MLPs based on the rest within encoder
        self.layoutlmv3 = LayoutLMv3ModelEE(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        exited = False
        try:
            outputs = self.layoutlmv3(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            sequence_output = outputs[0][:, 0, :]
            logits = self.classifier(sequence_output)
        except (
            EarlyExitException
        ) as e:  # wraps any early exit in embeddings or encoder layers
            exited = True
            outputs = e.payload
            if self.apply_gating:
                # sequence_output = outputs[0][
                #     :, 0, :
                # ]
                logits = self.classifier(
                    outputs.gate_inputs[-1]
                )  # final prediction is based on last gate input
            else:
                logits, _ = outputs.exit_states[
                    -1
                ]  # last exit state -> where exited :) WITHOUT ENTROPY
                # logits = outputs[0] ~ last hidden state was overwritten by exit ramp

        # implemented training strategies here + loss updates
        # outputs will contain exit logits as well -> calculate final loss; apply potential weighting
        loss = None
        exit_criteria = []  # either entropy or max confidence
        exit_losses = []
        all_gated_logits = ()
        if labels is not None:
            # final layer loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # collecte early exit losses

            ## change it to binary CE if not ramp
            if self.apply_gating:
                gate_losses = []
                for j, intermediate_rep in enumerate(outputs.gate_inputs):
                    ## Binary CE loss on gates (correctly gated or not)
                    gated_logits = self.classifier(intermediate_rep)
                    correctly_gated = gated_logits.argmax(-1) == labels

                    # have to onehot for Binary CE
                    y = torch.zeros(correctly_gated.shape[0], 2).to(labels.device)
                    y[range(y.shape[0]), correctly_gated.long()] = 1

                    loss_fct = nn.BCEWithLogitsLoss()
                    exit_logits = outputs.exit_states[j]
                    if isinstance(exit_logits, tuple):
                        exit_logits, exit_criterion = exit_logits
                        exit_criteria.append(exit_criterion)
                    gate_loss = loss_fct(exit_logits, y)
                    gate_losses.append(gate_loss)
                    all_gated_logits = all_gated_logits + (gated_logits,)

                    ## optionally add loss on gated logits?
                    # continue

                    loss_fct = nn.CrossEntropyLoss()
                    labels = torch.flatten(labels)
                    exit_loss = loss_fct(gated_logits, labels)
                    exit_losses.append(exit_loss)

                exit_losses = gate_losses
                # exit_losses = np.add(gate_losses,exit_losses)

            elif self.use_lte:
                layer_acc = []
                exit_pred = []
                losses = []
                exit_criteria = []
                inter_losses = []
                for j, exit_logits in enumerate(outputs.exit_states):
                    if self.num_labels == 1:
                        loss_fct = torch.nn.MSELoss()
                        highway_loss = loss_fct(exit_logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = torch.nn.CrossEntropyLoss()

                        if isinstance(exit_logits, tuple):
                            exit_logits, exit_criterion = exit_logits
                            exit_criteria.append(exit_criterion)

                        highway_loss = loss_fct(
                            exit_logits.view(-1, self.num_labels), labels.view(-1)
                        )
                    inter_losses.append(highway_loss)

                    lte_loss_fct = torch.nn.MSELoss()

                    # uncertainty / prob to continue
                    layer_acc = []
                    exit_pred = []
                    exit_pred.append(outputs.lte_output[j].to(labels.device))

                    # label
                    if j + 1 == self.num_layers:
                        layer_output = logits
                    else:
                        layer_output = exit_logits
                    if self.num_labels == 1:
                        correctness_loss = torch.tanh(
                            layer_output.squeeze() - labels
                        ).abs()
                    else:
                        if isinstance(layer_output, tuple):
                            layer_output, exit_criterion = layer_output
                            exit_criteria.append(exit_criterion)

                        lte_gold = torch.eq(
                            torch.argmax(layer_output, dim=1), labels
                        ).to(
                            labels.device
                        )  # 0 for wrong/continue, 1 for right/exit

                        correctness_loss = (
                            1 - lte_gold.float()
                        )  # 1 for continue, match exit_pred

                    layer_acc.append(correctness_loss)
                    exit_pred = torch.stack(exit_pred)
                    exit_label = torch.stack(layer_acc)

                    total_loss = (
                        sum(inter_losses) + loss + lte_loss_fct(exit_pred, exit_label)
                    )
                    losses.append(total_loss)

                exit_losses = losses

            else:
                for j, exit_logits in enumerate(outputs.exit_states):
                    if isinstance(exit_logits, tuple):
                        exit_logits, exit_criterion = exit_logits
                        exit_criteria.append(exit_criterion)

                    loss_fct = nn.CrossEntropyLoss()
                    exit_loss = loss_fct(
                        exit_logits.view(-1, self.num_labels), labels.view(-1)
                    )
                    exit_losses.append(exit_loss)

        if not exited:
            exit_criteria.append(self.layoutlmv3.exit_criterion(logits))

        if self.training_strategy == "raw":
            outputs = (loss,) + outputs

        elif self.training_strategy == "joint_weighted_avg":
            loss = loss + sum(exit_losses)

        elif "one_stage" in self.training_strategy:
            pass  # adapted trainer's losses backward calls in training step

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return EESequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            exit_losses=exit_losses,
            exit_criteria=exit_criteria,
            exit_states=outputs.exit_states,  # TODO: can deduce exit layer from length of exit_states and checking config.exit_config.exits
            gated_logits=all_gated_logits,
        )


def test_model():
    config = {
        "training_strategy": "one_stage_subgraphs",
        "exits": POSSIBLE_EXITS,
        "encoder_layer_strategy": "gate",
    }  # {"exits": ["text_avg", 1]}  # test with two different types of exits
    config["model_weights"] = "microsoft/layoutlmv3-base"
    HF_config = AutoConfig.from_pretrained(config["model_weights"], num_labels=4)
    HF_config.EE_config = config
    model = LayoutLMv3EEForSequenceClassification.from_pretrained(
        config["model_weights"],
        config=HF_config,
    )
    test_input = test_processor()
    outputs = model(**test_input)
    return model, outputs


def test_processor(simulated_sequence_length=None):
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)  # dummy image
    words = ["pencil", "0000000000000000", "phone"]
    boxes = [[1, 2, 3, 4], [10, 11, 12, 13], [20, 21, 22, 23]]
    if simulated_sequence_length is not None:
        words = [words[0]] * simulated_sequence_length
        boxes = [boxes[0]] * simulated_sequence_length
    word_labels = [0]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    encoding["labels"] = torch.Tensor([word_labels]).long()
    # print(encoding["input_ids"])
    # print(processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].flatten()))
    # print(encoding["labels"])
    return encoding


if __name__ == "__main__":
    # test_processor()
    test_model()
