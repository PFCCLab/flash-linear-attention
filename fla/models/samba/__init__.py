import paddleformers

from fla.models.samba.configuration_samba import SambaConfig
from fla.models.samba.modeling_samba import SambaForCausalLM, SambaModel

paddleformers.transformers.AutoConfig.register(SambaConfig.model_type, SambaConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(SambaConfig, SambaModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(SambaConfig, SambaForCausalLM, exist_ok=True)


__all__ = ['SambaConfig', 'SambaForCausalLM', 'SambaModel']