from pathlib import Path

import pytest

from sentiment.engine import SentimentEngine


def test_missing_checkpoint_has_actionable_error():
    with pytest.raises(FileNotFoundError, match='model.safetensors'):
        SentimentEngine(Path('outputs/checkpoints/does_not_exist'))


@pytest.mark.integration
def test_real_checkpoint_prediction_and_explanation():
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('captum')
    engine = SentimentEngine(Path('outputs/checkpoints/bert_seed_13'))
    positive = engine.predict('A wonderful film with beautiful acting. I loved every minute.')
    negative = engine.predict('An awful, boring movie. The acting was terrible and I hated it.')
    assert positive['label'] == 'positive'
    assert negative['label'] == 'negative'
    assert sum(positive['probabilities'].values()) == pytest.approx(1, abs=1e-6)
    explanation = engine.explain(positive['text'], initial_steps=16, max_steps=32)
    assert explanation['groups']
    assert all(g['end'] <= explanation['analyzed_end'] for g in explanation['groups'])
    assert explanation['absolute_residual'] >= 0
    assert engine.predict(positive['text'])['confidence'] == pytest.approx(positive['confidence'])
