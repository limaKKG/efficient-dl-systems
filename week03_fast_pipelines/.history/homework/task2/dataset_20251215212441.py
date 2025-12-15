from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
import random

MAX_LENGTH = 640
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
PAD_ID = TOKENIZER.pad_token_id or 0

def _read_train_lines(data_path: str) -> List[str]:
    parts = ["train-00000-of-00002.txt", "train-00001-of-00002.txt"]
    lines: List[str] = []
    for fname in parts:
        with open(f"{data_path}/{fname}", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


def _tokenize_line(text: str, max_length: int) -> List[int]:
    tokens = TOKENIZER.tokenize(text)
    token_ids = TOKENIZER.convert_tokens_to_ids(tokens)
    return token_ids[:max_length]
    
class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.samples: List[torch.Tensor] = []
        lines = _read_train_lines(data_path)
        for line in lines:
            ids = _tokenize_line(line, max_length)
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [PAD_ID] * pad_len
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.samples[idx]
        return x, self.max_length 


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        lines = _read_train_lines(data_path)
        self.samples: List[List[int]] = [
            _tokenize_line(line, max_length) for line in lines
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ids = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        lines = _read_train_lines(data_path)
        self.samples: List[List[int]] = [
            _tokenize_line(line, max_length) for line in lines
        ]
        self.lengths: List[int] = [len(x) for x in self.samples]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ids = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids)


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.lines = _read_train_lines(data_path)

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        current_tokens: List[int] = []
        current_segments: List[int] = []
        seg_id = 0

        def emit_chunk():
            nonlocal current_tokens, current_segments
            if not current_tokens:
                return None
            L = len(current_tokens)
            pad_len = self.max_length - L
            tokens = current_tokens + [PAD_ID] * pad_len
            segments = current_segments + [-1] * pad_len
            attn_mask = torch.zeros(self.max_length, self.max_length)
            attn_mask += torch.triu(torch.ones(self.max_length, self.max_length) * float("-inf"), diagonal=1)
            for i in range(self.max_length):
                for j in range(self.max_length):
                    if segments[i] != segments[j]:
                        attn_mask[i, j] = float("-inf")
            input_ids = torch.tensor(tokens, dtype=torch.long)
            target_ids = input_ids.clone()
            current_tokens = []
            current_segments = []
            return input_ids, attn_mask, target_ids

        for line in self.lines:
            ids = _tokenize_line(line, self.max_length)
            if not ids:
                continue
            while ids:
                space = self.max_length - len(current_tokens)
                take = min(space, len(ids))
                current_tokens.extend(ids[:take])
                current_segments.extend([seg_id] * take)
                ids = ids[take:]
                if len(current_tokens) == self.max_length:
                    chunk = emit_chunk()
                    if chunk:
                        yield chunk
            seg_id += 1

        chunk = emit_chunk()
        if chunk:
            yield chunk

def collate_fn(
    batch: list[tuple[str, torch.Tensor]], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    if max_length is None:
        max_len = min(max(len(x[0]) for x in batch), MAX_LENGTH)
    else:
        max_len = max_length
    max_len = max(1, max_len)  

    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)

    for i, (tokens, _) in enumerate(batch):
        seq = tokens[:max_len]
        input_ids[i, : len(seq)] = seq
        targets[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask, targets


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, dataset_lengths: List[int], batch_size: int, k: int = 10):
        self.batch_size = batch_size
        self.k = k
        self.bins: Dict[int, List[int]] = {}
        for idx, length in enumerate(dataset_lengths):
            bin_id = length // k
            self.bins.setdefault(bin_id, []).append(idx)
        self._num_batches = sum((len(v) + batch_size - 1) // batch_size for v in self.bins.values())

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        bin_ids = list(self.bins.keys())
        random.shuffle(bin_ids)
        for b in bin_ids:
            idxs = self.bins[b][:]
            random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                yield idxs[i : i + self.batch_size]
