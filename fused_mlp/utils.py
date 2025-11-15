# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Simplified helper functions extracted from unsloth/kernels/utils.py
# Removes quantization, QAT, FP8, and multi-GPU support for minimal standalone version

import torch
from packaging.version import Version
from contextlib import nullcontext

if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")


def torch_gpu_device(device):
    return nullcontext()


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)


def get_lora_parameters(proj):
    """
    Return a 5-tuple of (weight, weight quant_state, lora A, lora B, and lora scale).
    If QAT is enabled, additionally fake quantize the base layer and lora weights.
    """
    base_layer = getattr(
        proj, "base_layer", proj
    )
    W = base_layer.weight

    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    if type(adapter) is str:
        adapter = (adapter,)
    adapter = adapter[0]

    A = proj.lora_A[adapter].weight
    B = proj.lora_B[adapter].weight
    s = proj.scaling[adapter]
    return W, W_quant, A, B, s


def _maybe_fake_quantize_activations(
    X: torch.Tensor, proj: torch.nn.Module
) -> torch.Tensor:
    """
    If QAT is enabled, fake quantize the input activations.
    Otherwise, just return the input activations as is.
    Weights are fake quantized separately in `get_lora_parameters`.
    """
    base_layer = getattr(proj, "base_layer", proj)
    activation_fake_quantizer = getattr(base_layer, "activation_fake_quantizer", None)
    if activation_fake_quantizer is not None:
        X = activation_fake_quantizer(X)
    return X


@torch.inference_mode
def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
    if quant_state is None:
        return W
    return W


def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)
    out = torch.matmul(X, W, out = out)
    if W_quant is not None:
        del W

    if A is not None:
        A, B = A.t(), B.t()
        XA = torch.matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)

    return out.view(batch, seq_len, -1) if reshape else out
