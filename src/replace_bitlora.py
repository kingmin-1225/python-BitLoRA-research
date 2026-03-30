from peft.tuners.lora.layer import LoraLayer
import torch
from torch import nn
from src.utils import BitLinear158, BitLinear1

from peft.utils.warning import PeftWarning
from peft import ArrowConfig
from peft.utils.integrations import gather_params_ctx
import warnings
import math

class BitLoraLayer158(LoraLayer):
    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        use_alora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        arrow_config: ArrowConfig = None,
        qalora_group_size: int = 32,
        inference_mode: bool = False,
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        if lora_bias and (getattr(self.get_base_layer(), "bias", None) is None):
            warnings.warn(
                f"`lora_bias=True` was passed but the targeted layer of type {type(self.get_base_layer()).__name__} "
                "has no bias. This means that merging LoRA weights won't be possible.",
                PeftWarning,
            )

        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora,
            use_alora=use_alora,
            use_qalora=use_qalora,
            qalora_group_size=qalora_group_size,
            arrow_config=arrow_config,
        )
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        self.lora_A[adapter_name] = BitLinear158(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = BitLinear158(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_rslora[adapter_name] = use_rslora

        self.use_dora[adapter_name] = use_dora

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights == "orthogonal":
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before init of the lora variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

        # Check for adapters that were added or removed from the arrow_model.
        # The arrow model may be modified after creation by adding new experts
        # (pre-trained or trainable) or by removing existing ones. Whenever such
        # a change occurs, on_adapter_change() is called to update the set of
        # active task-specific experts and, if needed, to handle recomputing prototypes
        # and doing general knowledge subtraction (GKS) again.
        if hasattr(self, "lora_arrow"):
            for adapter in self.lora_variant:
                if adapter in self.lora_arrow:
                    self.lora_arrow[adapter].on_adapter_change(self.lora_A, self.lora_B)

class BitLoraLayer1(LoraLayer):
    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        use_alora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        arrow_config: ArrowConfig = None,
        qalora_group_size: int = 32,
        inference_mode: bool = False,
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        if lora_bias and (getattr(self.get_base_layer(), "bias", None) is None):
            warnings.warn(
                f"`lora_bias=True` was passed but the targeted layer of type {type(self.get_base_layer()).__name__} "
                "has no bias. This means that merging LoRA weights won't be possible.",
                PeftWarning,
            )

        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora,
            use_alora=use_alora,
            use_qalora=use_qalora,
            qalora_group_size=qalora_group_size,
            arrow_config=arrow_config,
        )
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        self.lora_A[adapter_name] = BitLinear1(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = BitLinear1(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_rslora[adapter_name] = use_rslora

        self.use_dora[adapter_name] = use_dora

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights == "orthogonal":
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before init of the lora variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

        # Check for adapters that were added or removed from the arrow_model.
        # The arrow model may be modified after creation by adding new experts
        # (pre-trained or trainable) or by removing existing ones. Whenever such
        # a change occurs, on_adapter_change() is called to update the set of
        # active task-specific experts and, if needed, to handle recomputing prototypes
        # and doing general knowledge subtraction (GKS) again.
        if hasattr(self, "lora_arrow"):
            for adapter in self.lora_variant:
                if adapter in self.lora_arrow:
                    self.lora_arrow[adapter].on_adapter_change(self.lora_A, self.lora_B)