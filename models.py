import torch
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig


def load_vanilla_lora(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=getattr(torch, cfg.model.dtype),
        device_map="auto"
    )
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    return model, tokenizer


def load_fused_kernel(cfg):
    from fused_mlp import apply_lora_mlp_swiglu
    assert cfg.lora.dropout == 0.0, "fused kernel requires dropout=0"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=getattr(torch, cfg.model.dtype),
        device_map="auto"
    )
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    mlp_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
            module.forward = types.MethodType(apply_lora_mlp_swiglu, module)
            mlp_count += 1
    print(f"fused kernel: patched {mlp_count} MLP layers", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    return model, tokenizer


def load_full_unsloth(cfg):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_len,
        dtype=getattr(torch, cfg.model.dtype),
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        random_state=cfg.training.seed,
    )
    return model, tokenizer
