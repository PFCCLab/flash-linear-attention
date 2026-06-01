import paddleformers


from fla.models.forgetting_transformer.configuration_forgetting_transformer import ForgettingTransformerConfig
from fla.models.forgetting_transformer.modeling_forgetting_transformer import (
    ForgettingTransformerForCausalLM,
    ForgettingTransformerModel,
)

paddleformers.transformers.AutoConfig.register(ForgettingTransformerConfig.model_type, ForgettingTransformerConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(ForgettingTransformerConfig, ForgettingTransformerModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(ForgettingTransformerConfig, ForgettingTransformerForCausalLM, exist_ok=True)


__all__ = ['ForgettingTransformerConfig', 'ForgettingTransformerForCausalLM', 'ForgettingTransformerModel']
