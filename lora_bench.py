import torch
import json
from omegaconf import OmegaConf
from models import load_vanilla_lora, load_fused_kernel, load_full_unsloth
from trainer import train


def run_variant(cfg, variant_name, load_fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_fn(cfg)
    metrics = train(model, tokenizer, cfg, variant_name)
    result = {
        "total_time_ms": round(metrics["total_time_ms"], 2),
        "time_per_iter_ms": round(metrics["time_per_iter_ms"], 2),
        "forward_time_ms": round(metrics["forward_time_ms"], 2),
        "backward_time_ms": round(metrics["backward_time_ms"], 2),
        "total_stddev_ms": round(metrics["total_stddev_ms"], 2),
        "forward_stddev_ms": round(metrics["forward_stddev_ms"], 2),
        "backward_stddev_ms": round(metrics["backward_stddev_ms"], 2)
    }
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return result


def main():
    cfg = OmegaConf.load("config.yaml")
    seq_lens = cfg.experiments.seq_lens
    variants = {
        "vanilla_lora": load_vanilla_lora,
        "fused_kernel": load_fused_kernel,
        "full_unsloth": load_full_unsloth
    }
    all_results = {seq_len: {"config": {}, "results": {}} for seq_len in seq_lens}
    for variant_name, load_fn in variants.items():
        print(f"running experiments: {variant_name}", flush=True)
        for seq_len in seq_lens:
            print(f"seq_len={seq_len}", flush=True)
            cfg.model.max_seq_len = seq_len
            result = run_variant(cfg, variant_name, load_fn)
            all_results[seq_len]["config"] = {
                "model": cfg.model.name,
                "max_seq_len": cfg.model.max_seq_len,
                "iterations": cfg.training.iterations,
                "batch_size": cfg.training.batch_size,
                "lora_r": cfg.lora.r
            }
            all_results[seq_len]["results"][variant_name] = result

    for seq_len in seq_lens:
        vanilla_time = all_results[seq_len]["results"]["vanilla_lora"]["time_per_iter_ms"]
        fused_time = all_results[seq_len]["results"]["fused_kernel"]["time_per_iter_ms"]
        full_time = all_results[seq_len]["results"]["full_unsloth"]["time_per_iter_ms"]
        all_results[seq_len]["speedup"] = {
            "fused_vs_vanilla": round(vanilla_time / fused_time, 2),
            "full_vs_fused": round(fused_time / full_time, 2),
            "full_vs_vanilla": round(vanilla_time / full_time, 2)
        }
    with open("results.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
