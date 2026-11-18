"""Pinned local multimodal inference. No automatic device placement/offloading."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from .json_protocol import parse_json
from .constrained_decoding import GrammarFactory, schema_sha256


class LocalVLM:
    def __init__(self, model_path: str, max_pixels: int = 262144):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required. CPU model fallback is disabled.')
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=65536, max_pixels=max_pixels,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa', local_files_only=True,
        ).eval()
        for name, value in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if value.device.type != 'cuda':
                raise RuntimeError(f'Non-CUDA model tensor: {name} on {value.device}')
        self.device_audit = {
            'all_parameters_cuda': True, 'all_buffers_cuda': True,
            'dtype': 'bfloat16', 'hf_device_map': self.model.hf_device_map,
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
        }
        self._grammar_factory = None

    def grammar_factory(self):
        # Only the serial caller or the single batching worker invokes this.
        if self._grammar_factory is None:
            self._grammar_factory = GrammarFactory(self.processor.tokenizer)
        return self._grammar_factory

    @torch.inference_mode()
    def generate(self, prompt: str, images: list[str | Path], max_new_tokens: int,
                 json_schema: dict | None = None) -> dict:
        content = [{'type': 'image', 'image': Image.open(path).convert('RGB')} for path in images]
        content.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content}]
        wall_start = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt',
        ).to('cuda:0')
        generation_options = {}
        if json_schema is not None:
            eos = self.model.generation_config.eos_token_id
            eos = eos if eos is not None else self.processor.tokenizer.eos_token_id
            eos_ids = [eos] if isinstance(eos,int) else list(eos)
            generation_options['prefix_allowed_tokens_fn'] = self.grammar_factory().for_batch(
                [json_schema],inputs['input_ids'].shape[1],[max_new_tokens],eos_ids)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        start.record()
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     use_cache=True, **generation_options)
        end.record()
        torch.cuda.synchronize()
        new_ids = output[:, inputs['input_ids'].shape[1]:]
        text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        result = {
            'raw_text': text, 'input_tokens': int(inputs['input_ids'].shape[1]),
            'output_tokens': int(new_ids.shape[1]),
            'gpu_seconds': start.elapsed_time(end) / 1000,
            'wall_seconds': time.perf_counter() - wall_start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'image_grid_thw': inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else [],
            'decode': 'greedy', 'max_new_tokens': max_new_tokens,
            'at_generation_limit': new_ids.shape[1] == max_new_tokens,
            'json_schema_sha256': schema_sha256(json_schema) if json_schema is not None else None,
            'structured_decoding': self.grammar_factory().provenance['protocol'] if json_schema is not None else None,
        }
        try:
            result['parsed'] = parse_json(text)
            result['parse_error'] = None
        except ValueError as error:
            result['parsed'] = None
            result['parse_error'] = str(error)
        return result
