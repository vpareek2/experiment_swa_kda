"""Bounded CUDA-event and aggregate-operator profile for the speed supervisor."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch

class SpeedProfile:
    def __init__(self, model, max_bytes: int, rows: int):
        self.model, self.max_bytes, self.rows = model, max_bytes, rows
        self.active = False; self.records = defaultdict(list); self.handles = []
        self._patch_fla()
        for name, module in model.named_modules():
            label = self._label(name, module)
            if label:
                self.handles += [module.register_forward_pre_hook(lambda m, a, label=label: self._start(label)),
                                 module.register_forward_hook(lambda m, a, o, label=label: self._end(label))]
                if label.endswith('kda_layer'):
                    self.handles += [module.register_full_backward_pre_hook(lambda m, g, label=label: self._start(label + '_backward')),
                                     module.register_full_backward_hook(lambda m, gi, go, label=label: self._end(label + '_backward'))]
    def _label(self, name, module):
        if module.__class__.__name__ == 'KimiDeltaAttention': return f'{name}.kda_layer'
        for suffix, label in (('.q_proj','q_projection'),('.k_proj','k_projection'),('.v_proj','v_projection'),
                              ('.q_conv1d','q_short_convolution'),('.k_conv1d','k_short_convolution'),('.v_conv1d','v_short_convolution'),
                              ('.g_proj','output_gate_projection'),('.o_proj','output_projection'),('.o_norm','output_norm')):
            if name.endswith(suffix): return label
        return None
    def _event(self): return torch.cuda.Event(enable_timing=True)
    def _start(self, label):
        if self.active:
            event=self._event(); event.record(); self.records[label].append([event, None])
    def _end(self, label):
        if self.active and self.records[label] and self.records[label][-1][1] is None:
            event=self._event(); event.record(); self.records[label][-1][1]=event
    def _patch_fla(self):
        import importlib
        kda_module = importlib.import_module("nanochat.mixers.kda")
        self.kda_module, self.original_fla = kda_module, kda_module._run_fla_kda
        def wrapped(*args, **kwargs):
            self._start('fla_kda_forward')
            try: return self.original_fla(*args, **kwargs)
            finally: self._end('fla_kda_forward')
        kda_module._run_fla_kda = wrapped
    def begin(self):
        self.active = True
        self._start('training_update')
    def mark(self, label, begin):
        (self._start if begin else self._end)(label)
    def finish(self):
        self._end('training_update'); torch.cuda.synchronize(); self.active=False
        regions={}
        for label, pairs in self.records.items():
            values=[a.elapsed_time(b) for a,b in pairs if b is not None]
            if values: regions[label]={'milliseconds': sum(values), 'calls': len(values)}
        # These are protected named operator regions, not an unbounded CUPTI
        # event stream. Their fixed taxonomy makes the artifact comparable
        # across candidates without producing a Chrome trace or millions of
        # per-kernel records.
        operator_regions = [
            {'name': name, **value} for name, value in sorted(regions.items(), key=lambda item: item[1]['milliseconds'], reverse=True)
        ][:self.rows]
        return {'schema_version': 1, 'profile_mode': 'mandatory_cuda_event_operator_regions',
                'regions': regions, 'operator_regions': operator_regions}
    def write(self, path: str):
        result=self.finish(); encoded=json.dumps(result, sort_keys=True).encode()
        if len(encoded) > self.max_bytes: raise RuntimeError(f'speed profile exceeds {self.max_bytes} byte cap')
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(encoded + b'\n')
        return result
    def close(self):
        for handle in self.handles: handle.remove()
        self.kda_module._run_fla_kda = self.original_fla
