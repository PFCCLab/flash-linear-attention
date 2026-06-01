import paddleformers

from fla.models.lightnet.configuration_lightnet import LightNetConfig

from fla.models.lightnet.modeling_lightnet import LightNetForCausalLM, LightNetModel

paddleformers.transformers.AutoConfig.register(LightNetConfig.model_type, LightNetConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(LightNetConfig, LightNetModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(LightNetConfig, LightNetForCausalLM, exist_ok=True)


__all__ = ['LightNetConfig', 'LightNetForCausalLM', 'LightNetModel']
