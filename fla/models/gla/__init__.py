import paddleformers

from fla.models.gla.configuration_gla import GLAConfig

from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel

paddleformers.transformers.AutoConfig.register(GLAConfig.model_type, GLAConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(GLAConfig, GLAModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM, exist_ok=True)


__all__ = ['GLAConfig', 'GLAForCausalLM', 'GLAModel']