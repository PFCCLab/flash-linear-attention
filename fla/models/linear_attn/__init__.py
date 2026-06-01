import paddleformers

from fla.models.linear_attn.configuration_linear_attn import LinearAttentionConfig

from fla.models.linear_attn.modeling_linear_attn import LinearAttentionForCausalLM, LinearAttentionModel

paddleformers.transformers.AutoConfig.register(LinearAttentionConfig.model_type, LinearAttentionConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(LinearAttentionConfig, LinearAttentionModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(LinearAttentionConfig, LinearAttentionForCausalLM, exist_ok=True)

__all__ = ['LinearAttentionConfig', 'LinearAttentionForCausalLM', 'LinearAttentionModel']
