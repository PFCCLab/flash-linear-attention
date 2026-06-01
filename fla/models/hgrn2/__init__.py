import paddleformers

from fla.models.hgrn2.configuration_hgrn2 import HGRN2Config

from fla.models.hgrn2.modeling_hgrn2 import HGRN2ForCausalLM, HGRN2Model

paddleformers.transformers.AutoConfig.register(HGRN2Config.model_type, HGRN2Config, exist_ok=True)
paddleformers.transformers.AutoModel.register(HGRN2Config, HGRN2Model, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM, exist_ok=True)


__all__ = ['HGRN2Config', 'HGRN2ForCausalLM', 'HGRN2Model']
