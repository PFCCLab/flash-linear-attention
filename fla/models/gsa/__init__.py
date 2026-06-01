import paddleformers

from fla.models.gsa.configuration_gsa import GSAConfig

from fla.models.gsa.modeling_gsa import GSAForCausalLM, GSAModel

paddleformers.transformers.AutoConfig.register(GSAConfig.model_type, GSAConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(GSAConfig, GSAModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(GSAConfig, GSAForCausalLM, exist_ok=True)


__all__ = ['GSAConfig', 'GSAForCausalLM', 'GSAModel']
