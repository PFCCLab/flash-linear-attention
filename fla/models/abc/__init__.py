import paddleformers

from fla.models.abc.configuration_abc import ABCConfig

from fla.models.abc.modeling_abc import ABCForCausalLM, ABCModel

paddleformers.transformers.AutoConfig.register(ABCConfig.model_type, ABCConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(ABCConfig, ABCModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM, exist_ok=True)


__all__ = ['ABCConfig', 'ABCForCausalLM', 'ABCModel']
