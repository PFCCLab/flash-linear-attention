import paddleformers

from fla.models.nsa.configuration_nsa import NSAConfig
from fla.models.nsa.modeling_nsa import NSAForCausalLM, NSAModel

paddleformers.transformers.AutoConfig.register(NSAConfig.model_type, NSAConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(NSAConfig, NSAModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(NSAConfig, NSAForCausalLM, exist_ok=True)


__all__ = ['NSAConfig', 'NSAForCausalLM', 'NSAModel']
