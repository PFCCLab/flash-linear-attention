import paddleformers

from fla.models.retnet.configuration_retnet import RetNetConfig

from fla.models.retnet.modeling_retnet import RetNetForCausalLM, RetNetModel

paddleformers.transformers.AutoConfig.register(RetNetConfig.model_type, RetNetConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(RetNetConfig, RetNetModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(RetNetConfig, RetNetForCausalLM, exist_ok=True)


__all__ = ['RetNetConfig', 'RetNetForCausalLM', 'RetNetModel']
