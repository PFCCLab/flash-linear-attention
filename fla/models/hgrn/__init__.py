import paddleformers

from fla.models.hgrn.configuration_hgrn import HGRNConfig

from fla.models.hgrn.modeling_hgrn import HGRNForCausalLM, HGRNModel

paddleformers.transformers.AutoConfig.register(HGRNConfig.model_type, HGRNConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(HGRNConfig, HGRNModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM, exist_ok=True)


__all__ = ['HGRNConfig', 'HGRNForCausalLM', 'HGRNModel']
