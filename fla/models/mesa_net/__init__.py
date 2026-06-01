import paddleformers

from fla.models.mesa_net.configuration_mesa_net import MesaNetConfig

from fla.models.mesa_net.modeling_mesa_net import MesaNetForCausalLM, MesaNetModel

paddleformers.transformers.AutoConfig.register(MesaNetConfig.model_type, MesaNetConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(MesaNetConfig, MesaNetModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(MesaNetConfig, MesaNetForCausalLM, exist_ok=True)

__all__ = ['MesaNetConfig', 'MesaNetForCausalLM', 'MesaNetModel']
