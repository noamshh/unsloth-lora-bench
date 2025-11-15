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

import torch
from .utils import (
    _maybe_fake_quantize_activations,
    fast_dequantize,
    get_lora_parameters,
    matmul_lora,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
)


class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        _forward_function,
        _backward_function,
        inplace = True,
    ):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return i

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _backward_function,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = (
            gateA.to(dtype),
            gateB.to(dtype),
            upA.to(dtype),
            upB.to(dtype),
            downA.to(dtype),
            downB.to(dtype),
        )

        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(),
            gateB.t(),
            upA.t(),
            upB.t(),
            downA.t(),
            downB.t(),
        )

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)

        # Down projection LoRA weights
        # d_downA = h.t() @ (dY @ downB.t())
        # d_downB = (downA.t() @ h.t()) @ dY
        # d_downA *= downS
        # d_downB *= downS
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

        # Up projection LoRA weights
        # d_upA   = X.t() @ (df @ upB.t())
        # d_upB   = (upA.t() @ X.t()) @ df
        # d_upA  *= upS
        # d_upB  *= upS
        d_upA.addmm_(X.t(), df @ upB.t(), alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha = upS, beta = 0)

        # Gate projection LoRA weights
        # d_gateA = X.t() @ (de @ gateB.t())
        # d_gateB = (gateA.t() @ X.t()) @ de
        # d_gateA *= gateS
        # d_gateB *= gateS
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
        del upW
        # dX += df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())
        dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        # dX += de @ gateW.t()
        dX.addmm_(de, gateW.t())
        del gateW
        # dX += de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_gateA.t(),
            d_gateB.t(),
            None,
            None,
            None,
            d_upA.t(),
            d_upB.t(),
            None,
            None,
            None,
            d_downA.t(),
            d_downB.t(),
            None,
            None,
            None,
            None,
        )  # _backward and _forward and inplace


from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel


def apply_lora_mlp_swiglu(self, X, inplace = True):
    X = _maybe_fake_quantize_activations(X, self.gate_proj)
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_fg_kernel,
        swiglu_DWf_DW_dfg_kernel,
        inplace,
    )
    return out
