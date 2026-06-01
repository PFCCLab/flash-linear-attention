import paddleformers

from fla.models.kda.configuration_kda import KDAConfig

from fla.models.kda.modeling_kda import KDAForCausalLM, KDAModel

paddleformers.transformers.AutoConfig.register(KDAConfig.model_type, KDAConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(KDAConfig, KDAModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(KDAConfig, KDAForCausalLM, exist_ok=True)

__all__ = ['KDAConfig', 'KDAForCausalLM', 'KDAModel']
