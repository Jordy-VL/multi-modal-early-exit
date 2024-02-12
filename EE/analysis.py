import json
from models.LayoutLMv3 import test_processor
from models.EE_modules import (
    count_named_parameters,
    flops_named_parameters,
    filter_encoder_exits,
)
from fvcore.nn import FlopCountAnalysis, parameter_count


class Analysis:
    # translate to counter
    def __init__(self, model) -> None:
        with open("models/EELayoutLM_exit_named_parameters-wotherexits.json", "r") as f:
            self.exit_named_params = json.load(f)

        self.module_param_counts = parameter_count(model)  # FULL COUNT
        self.flops_named_params = self.modules_to_flops(model, average_length=512)

    def modules_to_flops(self, model, average_length=512):
        test_input = test_processor(simulated_sequence_length=average_length)
        test_input = tuple(
            [v.to(model.device) for k, v in test_input.items()]
        )  # requires tuples on same device
        flops = FlopCountAnalysis(model, test_input)

        return flops.by_module()

    def exit_to_params_and_flops(self, exit_distribution, model):
        """
        For an exit distribution, return the number of parameters and flops
        For both, we calculate the full usage and then relative reduction from exits
        """
        # worst case analysis TODO: subclass and override to return output and forward pass flops
        exit_config = model.config.exit_config
        exit_param_count = {}
        exit_flops_count = {}
        encoder_counter = 0

        exit_latency = {}
        for exit_index, exit in enumerate(exit_config["exits"]):
            # LEGACY swap for old exits code where first text, then vision exits
            if exit == "text_avg" and exit_index == 0:
                exit = "vision_avg"
                exit_index = 1
            elif exit == "vision_avg" and exit_index == 1:
                exit = "text_avg"
                exit_index = 0

            modules = self.exit_named_params[str(exit)]
            # remove any modules with encoder index higher than counter f"encoder.early_exits.{encoder_counter}"
            if isinstance(exit, int):
                modules = filter_encoder_exits(modules, encoder_counter)

            new_modules = []
            for module in modules:
                if ".bias" in module or ".weight" in module:
                    module = module.replace(".bias", "").replace(".weight", "")
                new_modules.append(module)
            modules = list(set(new_modules))

            # need to filter for encoder if not all defined
            param_count = count_named_parameters(model, modules=modules)
            flops_count = flops_named_parameters(
                self.flops_named_params, modules=modules
            )  # approximation, but should be good enough for now

            exit_param_count[exit_index] = param_count * (
                exit_distribution[exit_index] * N
            )
            exit_flops_count[exit_index] = flops_count * (exit_distribution[exit_index])
            if isinstance(exit, int):  # encoder indices update relative to exit counter
                encoder_counter += 1

            exit_latency[exit_index] = (
                exit_distribution[exit_index]
                * (exit_index + 1)
                / (len(exit_config["exits"]) + 1)
            )

        classifier_index = len(exit_config["exits"])  # implicit +1
        full_param_count = self.module_param_counts[""] * N
        multiexit_param_count = sum(exit_param_count.values()) + (
            exit_distribution[classifier_index] * N * self.module_param_counts[""]
        )

        exit_flops_count[classifier_index] = (
            exit_distribution[classifier_index] * self.flops_named_params[""]
        )
        full_flops_count = self.flops_named_params[""]
        multiexit_flops_count = sum(exit_flops_count.values())
        exit_latency[classifier_index] = 1 * exit_distribution[classifier_index]
        multiexit_latency = sum(exit_latency.values())

        return (
            full_param_count,
            multiexit_param_count,
            full_flops_count,
            multiexit_flops_count,
            1,
            multiexit_latency,
        )
