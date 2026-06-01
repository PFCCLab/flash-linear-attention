import paddleformers

from fla.models.deltaformer.configuration_deltaformer import DeltaFormerConfig

from fla.models.deltaformer.modeling_deltaformer import DeltaFormerForCausalLM, DeltaFormerModel

paddleformers.transformers.AutoConfig.register(DeltaFormerConfig.model_type, DeltaFormerConfig, exist_ok=True)
paddleformers.transformers.AutoModel.register(DeltaFormerConfig, DeltaFormerModel, exist_ok=True)
paddleformers.transformers.AutoModelForCausalLM.register(DeltaFormerConfig, DeltaFormerForCausalLM, exist_ok=True)

__all__ = ['DeltaFormerConfig', 'DeltaFormerForCausalLM', 'DeltaFormerModel']
