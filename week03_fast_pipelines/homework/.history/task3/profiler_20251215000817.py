import json
import time
import torch
import os
from collections import defaultdict


class Profile:
    def __init__(self, model, name="model", schedule=None):
        self.name_map = self._build_name_map(model, name)
        self.events = []
        # schedule: dict with keys wait, warmup, active
        self.schedule = schedule or {"wait": 0, "warmup": 0, "active": 1}
        self.global_step = 0
        self.current_phase = "wait"
        self.start_time_ns = None

        self.fwd_start = {}
        self.bwd_start = {}
        self.handles = []
        self.model = model
    
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        if not self._is_active():
            return
        t = time.perf_counter_ns()
        self.fwd_start[module] = t
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(f"FWD:{self.name_map[module]}")

    def _forward_post_hook(self, module, inputs, outputs):
        if not self._is_active():
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            return
        end = time.perf_counter_ns()
        start = self.fwd_start.pop(module, end)
        dur_us = (end - start) / 1e3
        ts_us = (start - self.start_time_ns) / 1e3
        self.events.append(
            {
                "name": self.name_map[module],
                "cat": "forward",
                "ph": "X",
                "ts": ts_us,
                "dur": dur_us,
                "pid": 0,
                "tid": 0,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

    def _backward_pre_hook(self, module, grad_output):
        if not self._is_active():
            return
        t = time.perf_counter_ns()
        self.bwd_start[module] = t
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(f"BWD:{self.name_map[module]}")

    def _backward_post_hook(self, module, grad_input, grad_output):
        if not self._is_active():
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            return
        end = time.perf_counter_ns()
        start = self.bwd_start.pop(module, end)
        dur_us = (end - start) / 1e3
        ts_us = (start - self.start_time_ns) / 1e3
        self.events.append(
            {
                "name": self.name_map[module],
                "cat": "backward",
                "ph": "X",
                "ts": ts_us,
                "dur": dur_us,
                "pid": 0,
                "tid": 1,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

    def __enter__(self):
        self.start_time_ns = time.perf_counter_ns()
        for module in self.name_map:
            self.handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self.handles.append(module.register_forward_hook(self._forward_post_hook))
            if hasattr(module, "register_full_backward_pre_hook"):
                self.handles.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
            self.handles.append(module.register_full_backward_hook(self._backward_post_hook))
        return self
 
    def __exit__(self, type, value, traceback):
        for h in self.handles:
            h.remove()
        self.handles = []

    def step(self):
        self.global_step += 1
        w, wu, act = self.schedule.get("wait", 0), self.schedule.get("warmup", 0), self.schedule.get("active", 0)
        if self.global_step <= w:
            self.current_phase = "wait"
        elif self.global_step <= w + wu:
            self.current_phase = "warmup"
        elif self.global_step <= w + wu + act:
            self.current_phase = "active"
        else:
            self.current_phase = "done"

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        trace = {"traceEvents": self.events}
        with open(path, "w") as f:
            json.dump(trace, f)
        print(f"Saved trace to {path}")

    def _is_active(self):
        return self.current_phase == "active"
