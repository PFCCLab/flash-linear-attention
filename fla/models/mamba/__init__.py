import paddleformers

from fla.models.mamba.configuration_mamba import MambaConfig
from fla.models.mamba.modeling_mamba import MambaForCausalLM, MambaModel

paddleformers.transformers.AutoConfig.register(MambaConfig.model_type, MambaConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(MambaConfig, MambaModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM, exist_ok=True)


__all__ = ['MambaConfig', 'MambaForCausalLM', 'MambaModel']
