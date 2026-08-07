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
        self.active = False; self.records = defaultdict(list); self.handles = []; self.prof = None
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
        import nanochat.mixers.kda as kda_module
        self.kda_module, self.original_fla = kda_module, kda_module._run_fla_kda
        def wrapped(*args, **kwargs):
            self._start('fla_kda_forward')
            try: return self.original_fla(*args, **kwargs)
            finally: self._end('fla_kda_forward')
        kda_module._run_fla_kda = wrapped
    def begin(self):
        self.active = True
        self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False)
        self.prof.start(); self._start('training_update')
    def mark(self, label, begin):
        (self._start if begin else self._end)(label)
    def finish(self):
        self._end('training_update'); torch.cuda.synchronize(); self.active=False; self.prof.stop()
        regions={}
        for label, pairs in self.records.items():
            values=[a.elapsed_time(b) for a,b in pairs if b is not None]
            if values: regions[label]={'milliseconds': sum(values), 'calls': len(values)}
        entries=[]
        for event in self.prof.key_averages():
            self_cuda=float(getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0.0)))
            total_cuda=float(getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0.0)))
            if self_cuda or total_cuda:
                entries.append({'name': event.key, 'calls': int(event.count), 'self_cuda_us': self_cuda, 'total_cuda_us': total_cuda})
        entries.sort(key=lambda x: x['self_cuda_us'], reverse=True)
        return {'schema_version': 1, 'profile_mode': 'mandatory_cuda_events_and_aggregate_cuda_operators',
                'regions': regions, 'operators': entries[:self.rows], 'operator_event_count': sum(x['calls'] for x in entries)}
    def write(self, path: str):
        result=self.finish(); encoded=json.dumps(result, sort_keys=True).encode()
        if len(encoded) > self.max_bytes: raise RuntimeError(f'speed profile exceeds {self.max_bytes} byte cap')
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(encoded + b'\n')
        return result
    def close(self):
        for handle in self.handles: handle.remove()
        self.kda_module._run_fla_kda = self.original_fla
