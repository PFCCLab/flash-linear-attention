import paddleformers

from fla.models.delta_net.configuration_delta_net import DeltaNetConfig

from fla.models.delta_net.modeling_delta_net import DeltaNetForCausalLM, DeltaNetModel

paddleformers.transformers.AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(DeltaNetConfig, DeltaNetModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM, exist_ok=True)

__all__ = ['DeltaNetConfig', 'DeltaNetForCausalLM', 'DeltaNetModel']
