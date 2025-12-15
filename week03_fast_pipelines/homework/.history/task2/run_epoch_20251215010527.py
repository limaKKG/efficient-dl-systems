from enum import Enum
from typing import List
import json
import time
import torch
import time
import statistics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import (
    BrainDataset,
    BigBrainDataset,
    UltraBigBrainDataset,
    UltraBigBrainBatchSampler,
    UltraDuperBigBrainDataset,
    collate_fn,
    MAX_LENGTH,
    PAD_ID,
)
from transformer import PositionalEncoding, generate_square_subsequent_mask

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model() -> torch.nn.Module:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    d_model = 1024
    nhead = 8
    dim_ff = 4096

    decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=False
    )
    decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)

    class GPT2Like(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
            self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=MAX_LENGTH)
            self.decoder = decoder
            self.lm_head = torch.nn.Linear(d_model, vocab_size)

        def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor):
            x = self.embed(input_ids)  
            x = x.transpose(0, 1) 
            x = self.pos_enc(x)
            tgt_mask = attn_mask 
            if tgt_mask.dim() == 3:
                tgt_mask = tgt_mask[0]
            memory = torch.zeros(1, x.size(1), x.size(2), device=x.device)
            out = self.decoder(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=key_padding_mask,
            )
            out = self.lm_head(out)
            return out
    return GPT2Like()


def _make_loader(data_mode: DataMode, batch_size: int, k: int = 10):
    data_path = "wikitext-103-raw-v1"
    if data_mode == DataMode.BRAIN:
        ds = BrainDataset(data_path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, MAX_LENGTH))
        return loader
    if data_mode == DataMode.BIG_BRAIN:
        ds = BigBrainDataset(data_path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, None))
        return loader
    if data_mode == DataMode.ULTRA_BIG_BRAIN:
        ds = UltraBigBrainDataset(data_path)
        sampler = UltraBigBrainBatchSampler(ds.lengths, batch_size=batch_size, k=k)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda b: collate_fn(b, None))
        return loader
    if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        ds = UltraDuperBigBrainDataset(data_path)
        loader = DataLoader(ds, batch_size=1) 
        return loader


def _causal_mask(seq_len: int) -> torch.Tensor:
    return generate_square_subsequent_mask(seq_len)

def _collect_stats(times: list[float]) -> dict:
    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "median": statistics.median(times),
    }


def run_epoch(data_mode: DataMode, batch_size: int = 8, k: int = 10, device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = get_gpt2_model().to(device)
    loader = _make_loader(data_mode, batch_size=batch_size, k=k)
    warmup_steps = 5
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup_steps:
                break
            if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
                input_ids, attn_mask, targets = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                targets = targets.to(device)
                key_padding_mask = input_ids == PAD_ID
                model(input_ids, attn_mask, key_padding_mask)
            else:
                input_ids, attention_mask, targets = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                seq_len = input_ids.size(1)
        if seq_len == 0:
            continue
                attn_mask = _causal_mask(seq_len).to(device)
                key_padding_mask = attention_mask == 0
                model(input_ids, attn_mask, key_padding_mask)

    times: List[float] = []
    with torch.no_grad():
        for batch in loader:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            start = time.perf_counter()

            if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
                input_ids, attn_mask, targets = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                key_padding_mask = input_ids == PAD_ID
                model(input_ids, attn_mask, key_padding_mask)
            else:
                input_ids, attention_mask, targets = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                seq_len = input_ids.size(1)
                attn_mask = _causal_mask(seq_len).to(device)
                key_padding_mask = attention_mask == 0
                model(input_ids, attn_mask, key_padding_mask)

            torch.cuda.synchronize(device) if device.type == "cuda" else None
            end = time.perf_counter()
            times.append(end - start)

    stats = _collect_stats(times)
    print(f"{data_mode.name} | batch_size={batch_size} | k={k}: {stats}")
    return stats
