import paddleformers

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig

from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetModel

paddleformers.transformers.AutoConfig.register(GatedDeltaNetConfig.model_type, GatedDeltaNetConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(GatedDeltaNetConfig, GatedDeltaNetModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM, exist_ok=True)

__all__ = ['GatedDeltaNetConfig', 'GatedDeltaNetForCausalLM', 'GatedDeltaNetModel']
