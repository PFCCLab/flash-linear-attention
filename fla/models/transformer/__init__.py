import paddleformers

from fla.models.transformer.configuration_transformer import TransformerConfig

from fla.models.transformer.modeling_transformer import TransformerForCausalLM, TransformerModel

paddleformers.transformers.AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']