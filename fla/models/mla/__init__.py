import paddleformers

from fla.models.mla.configuration_mla import MLAConfig
from fla.models.mla.modeling_mla import MLAForCausalLM, MLAModel

paddleformers.transformers.AutoConfig.register(MLAConfig.model_type, MLAConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(MLAConfig, MLAModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(MLAConfig, MLAForCausalLM, exist_ok=True)


__all__ = ['MLAConfig', 'MLAForCausalLM', 'MLAModel']
