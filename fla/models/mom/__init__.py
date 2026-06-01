import paddleformers

from fla.models.mom.configuration_mom import MomConfig

from fla.models.mom.modeling_mom import MomForCausalLM, MomModel

paddleformers.transformers.AutoConfig.register(MomConfig.model_type, MomConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(MomConfig, MomModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(MomConfig, MomForCausalLM, exist_ok=True)

__all__ = ['MomConfig', 'MomForCausalLM', 'MomModel']
