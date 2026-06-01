import paddleformers

from fla.models.bitnet.configuration_bitnet import BitNetConfig

from fla.models.bitnet.modeling_bitnet import BitNetForCausalLM, BitNetModel

paddleformers.transformers.AutoConfig.register(BitNetConfig.model_type, BitNetConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(BitNetConfig, BitNetModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)


__all__ = ['BitNetConfig', 'BitNetForCausalLM', 'BitNetModel']
