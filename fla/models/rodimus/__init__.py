import paddleformers

from fla.models.rodimus.configuration_rodimus import RodimusConfig

from fla.models.rodimus.modeling_rodimus import RodimusForCausalLM, RodimusModel

paddleformers.transformers.AutoConfig.register(RodimusConfig.model_type, RodimusConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(RodimusConfig, RodimusModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(RodimusConfig, RodimusForCausalLM, exist_ok=True)


__all__ = ['RodimusConfig', 'RodimusForCausalLM', 'RodimusModel']