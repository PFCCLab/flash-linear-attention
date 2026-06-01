
import paddle
import torch
from torch import nn

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################


def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


paddle.Tensor._max = _Tensor_max


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
    return head_mask


def _get_head_mask(
    self,
    head_mask: paddle.Tensor | None,
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def _post_init(self):
    if hasattr(self, "init_weights"):
        self.init_weights()
    elif hasattr(self, "_init_weights"):
        self._init_weights()


def enable_paddleformers_compat():
    import paddleformers

    pretrained_model = paddleformers.transformers.model_utils.PretrainedModel
    generation_mixin = paddleformers.generation.utils.GenerationMixin

    pretrained_model.get_head_mask = _get_head_mask
    pretrained_model.device = None
    pretrained_model.post_init = _post_init

    if not hasattr(generation_mixin, "_paddle_utils_original_generate"):
        generation_mixin._paddle_utils_original_generate = generation_mixin.generate

        def _generate(self, input_ids, *args, **kwargs):
            return paddle.concat((input_ids, self._paddle_utils_original_generate(input_ids, *args, **kwargs)[0]), axis=-1)
        generation_mixin.generate = _generate


def _Tensor_min(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


paddle.Tensor._min = _Tensor_min


# RMSNorm 兼容 - 等价于 torch.nn.RMSNorm
if hasattr(paddle.compat.nn, 'RMSNorm'):
    paddle.nn.RMSNorm = paddle.compat.nn.RMSNorm
else:
    class _NativeRMSNorm(paddle.nn.Layer):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = [normalized_shape]
            else:
                normalized_shape = list(normalized_shape)

            self.normalized_shape = normalized_shape
            self.eps = eps
            self.elementwise_affine = elementwise_affine

            if elementwise_affine:
                self.weight = paddle.create_parameter(
                    shape=normalized_shape,
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(1.0),
                )
            else:
                self.weight = None

        def forward(self, x):
            # 与 torch.nn.RMSNorm 一致：归一化最后1维
            variance = x.pow(2).mean(axis=-1, keepdim=True)
            out = x * paddle.rsqrt(variance + self.eps)

            if self.weight is not None:
                out = out * self.weight

            return out

    paddle.nn.RMSNorm = _NativeRMSNorm
