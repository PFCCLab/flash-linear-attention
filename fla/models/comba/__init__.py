import paddleformers

from fla.models.comba.configuration_comba import CombaConfig

from fla.models.comba.modeling_comba import CombaForCausalLM, CombaModel

paddleformers.transformers.AutoConfig.register(CombaConfig.model_type, CombaConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(CombaConfig, CombaModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(CombaConfig, CombaForCausalLM, exist_ok=True)

__all__ = ['CombaConfig', 'CombaForCausalLM', 'CombaModel']
