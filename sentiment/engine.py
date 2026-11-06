"""Standalone local inference. Heavy dependencies load only when needed."""

import json
import math
import threading
from pathlib import Path

from sentiment.text import aggregate_words, normalize_review

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'outputs' / 'checkpoints' / 'bert_seed_13'
MAX_CHARACTERS = 20_000


class SentimentEngine:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        checkpoint = Path(checkpoint)
        for filename in ('model.safetensors', 'config.json', 'tokenizer.json'):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(f'Missing {filename} in {checkpoint}. '
                                        'Restore the complete bert_seed_13 checkpoint folder.')
        self.calibration = json.loads(Path(__file__).with_name('calibration.json').read_text())
        selection = checkpoint / 'selection.json'
        if not selection.is_file() or json.loads(selection.read_text()).get('seed') != 13:
            raise ValueError('This app’s calibration requires the seed-13 checkpoint and selection.json.')

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError('Word explanations need the saved fast tokenizer.')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, attn_implementation='eager').to(self.device)
        self.model.eval()
        if self.model.config.id2label != {0: 'negative', 1: 'positive'}:
            raise ValueError('The checkpoint must map class 0 to negative and class 1 to positive.')
        self.max_length = self.calibration['max_length']
        self.temperature = self.calibration['temperature']
        # Captum installs temporary hooks on the shared model. Serialize ALL forwards.
        self.lock = threading.RLock()

    def _encode(self, review):
        if len(review) > MAX_CHARACTERS:
            raise ValueError(f'Please keep reviews under {MAX_CHARACTERS:,} characters.')
        text = normalize_review(review)
        if not text:
            raise ValueError('Write a movie review first.')
        full = self.tokenizer(text, truncation=False, verbose=False)['input_ids']
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_offsets_mapping=True, return_special_tokens_mask=True,
                                 return_tensors='pt')
        offsets = encoded.pop('offset_mapping')[0].tolist()
        special = encoded.pop('special_tokens_mask')[0].tolist()
        content = [end for (start, end), sp in zip(offsets, special) if not sp and end > start]
        if not content:
            raise ValueError('Please include some review text to analyze.')
        truncated = len(full) > self.max_length
        return text, {k: v.to(self.device) for k, v in encoded.items()}, offsets, special, {
            'truncated': truncated,
            'analyzed_end': max(content) if truncated else len(text),
            'tokens_used': len(offsets), 'total_tokens': len(full),
        }

    def predict(self, review):
        with self.lock:
            text, encoded, _, _, coverage = self._encode(review)
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = (logits / self.temperature).softmax(-1)[0].cpu().tolist()
            target = max(range(2), key=probabilities.__getitem__)
            return {'text': text, 'label': ['negative', 'positive'][target], 'target': target,
                    'confidence': probabilities[target],
                    'probabilities': dict(zip(('negative', 'positive'), probabilities)),
                    'checkpoint': 'bert_seed_13', 'temperature': self.temperature, **coverage}

    def explain(self, review, initial_steps=64, max_steps=256):
        """Explain an ORIGINAL review input, with the same normalization as predict."""
        from captum.attr import LayerIntegratedGradients

        if not 2 <= initial_steps <= max_steps <= 512:
            raise ValueError('Integration steps must satisfy 2 <= initial <= maximum <= 512.')
        with self.lock, self.torch.enable_grad():
            text, encoded, offsets, special, coverage = self._encode(review)
            ids, mask = encoded['input_ids'], encoded['attention_mask']
            baseline = ids.clone()
            for index, is_special in enumerate(special):
                if not is_special:
                    baseline[0, index] = self.tokenizer.pad_token_id

            def forward(input_ids, attention_mask):
                return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            with self.torch.no_grad():
                original = forward(ids, mask)
                target = int(original.argmax(-1).item())
                logit_difference = abs(float((original[0, target] - forward(baseline, mask)[0, target]).item()))
            method = LayerIntegratedGradients(forward, self.model.bert.embeddings)
            steps = initial_steps
            tolerance = max(.05, .05 * logit_difference)
            while True:
                attr, delta = method.attribute(
                    inputs=ids, baselines=baseline, additional_forward_args=(mask,), target=target,
                    n_steps=steps, internal_batch_size=4, return_convergence_delta=True)
                residual = abs(float(delta.detach().cpu().item()))
                if not math.isfinite(residual) or not bool(self.torch.isfinite(attr).all()):
                    raise RuntimeError('The explanation produced non-finite scores. Try another review.')
                if residual <= tolerance or steps >= max_steps:
                    break
                steps = min(steps * 2, max_steps)
            scores = attr.sum(-1).detach().cpu().tolist()[0]
            groups = aggregate_words(text, offsets, special, scores)
            self.model.zero_grad(set_to_none=True)
            return {'groups': groups, 'target': target, 'steps': steps,
                    'absolute_residual': residual, 'tolerance': tolerance,
                    'converged': residual <= tolerance, 'reference': 'attended PAD', **coverage}
