import paddleformers

from fla.models.path_attn.configuration_path_attention import PaTHAttentionConfig

from fla.models.path_attn.modeling_path_attention import PaTHAttentionForCausalLM, PaTHAttentionModel

paddleformers.transformers.AutoConfig.register(PaTHAttentionConfig.model_type, PaTHAttentionConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(PaTHAttentionConfig, PaTHAttentionModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(PaTHAttentionConfig, PaTHAttentionForCausalLM, exist_ok=True)


__all__ = ['PaTHAttentionConfig', 'PaTHAttentionForCausalLM', 'PaTHAttentionModel']
