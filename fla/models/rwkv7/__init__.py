import paddleformers

from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config

from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model

paddleformers.transformers.AutoConfig.register(RWKV7Config.model_type, RWKV7Config, exist_ok=True)
paddleformers.transformers.AutoModel.register(RWKV7Config, RWKV7Model, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, exist_ok=True)


__all__ = ['RWKV7Config', 'RWKV7ForCausalLM', 'RWKV7Model']
