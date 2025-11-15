import torch
import json
from torch.utils.data import Dataset, DataLoader


class SFTDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=False)
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        encoding["labels"] = encoding["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in encoding.items()}


def train(model, tokenizer, cfg, variant_name):
    torch.manual_seed(cfg.training.seed)
    dataset = SFTDataset(cfg.training.dataset_path, tokenizer, cfg.model.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    model.train()
    data_iterator = iter(dataloader)

    print(f"warmup {cfg.training.warmup_steps} iters", flush=True)
    for iteration in range(cfg.training.warmup_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    measured_iters = cfg.training.iterations - cfg.training.warmup_steps
    print(f"training {measured_iters} iters", flush=True)

    forward_times = []
    backward_times = []

    for iteration in range(measured_iters):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        outputs = model(**batch)
        loss = outputs.loss
        fwd_end.record()
        bwd_start.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        forward_times.append(fwd_start.elapsed_time(fwd_end))
        backward_times.append(bwd_start.elapsed_time(bwd_end))

        if (iteration + 1) % 10 == 0:
            print(f"iter {iteration + 1}/{measured_iters}", flush=True)

    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    total_times = forward_times + backward_times

    return {
        "total_time_ms": total_times.sum().item(),
        "time_per_iter_ms": total_times.mean().item(),
        "forward_time_ms": forward_times.mean().item(),
        "backward_time_ms": backward_times.mean().item(),
        "total_stddev_ms": total_times.std().item(),
        "forward_stddev_ms": forward_times.std().item(),
        "backward_stddev_ms": backward_times.std().item()
    }
