from unittest.mock import patch
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from sentiment.text import normalize_review

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


@pytest.fixture(autouse=True)
def isolate_model_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_initial_screen_and_empty_submission():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.text_area(key='review').value == ''
    app.button(key='analyze').click().run()
    assert not app.exception
    assert any('review' in warning.value.lower() for warning in app.warning)


def test_example_and_clear():
    app = AppTest.from_file(APP).run()
    app.button(key='example_positive').click().run()
    assert 'wonderful' in app.text_area(key='review').value
    app.button(key='clear').click().run()
    assert app.text_area(key='review').value == ''


class FakeEngine:
    """UI contract fixture; actual inference has its own integration test."""
    def predict(self, text):
        text = normalize_review(text)
        return dict(text=text, label='positive', target=1, confidence=.9,
                    probabilities={'negative': .1, 'positive': .9}, checkpoint='bert_seed_13',
                    temperature=1.7926, truncated=False, analyzed_end=len(text),
                    tokens_used=8, total_tokens=8)

    def explain(self, text):
        return dict(groups=[dict(word=text[:1], start=0, end=1, score=.2)],
                    target=1, steps=64, absolute_residual=.01, tolerance=.05,
                    converged=True, reference='attended PAD', analyzed_end=len(text))


def test_analysis_explanation_and_edit_invalidate_old_result():
    with patch('sentiment.engine.SentimentEngine', FakeEngine):
        app = AppTest.from_file(APP).run()
        app.text_area(key='review').input('A wonderful film.').run()
        app.button(key='analyze').click().run()
        assert not app.exception
        assert app.session_state['prediction']['label'] == 'positive'
        app.button(key='explain').click().run()
        assert not app.exception
        assert app.session_state['explanation']['converged']
        assert any('highlight-review' in item.value for item in app.markdown)
        app.text_area(key='review').input('A different review.').run()
        assert app.session_state['prediction'] is None
        assert app.session_state['explanation'] is None


def test_missing_model_library_produces_setup_message():
    with patch('sentiment.engine.SentimentEngine', side_effect=ModuleNotFoundError("No module named 'torch'")):
        app = AppTest.from_file(APP).run()
        app.text_area(key='review').input('A movie worth seeing.').run()
        app.button(key='analyze').click().run()
        assert not app.exception
        assert app.error
        assert any('requirements.txt' in item.value for item in app.code)


def test_explanation_receives_original_review_to_avoid_double_entity_decoding():
    raw_review = 'A &amp;lt;wonderful&amp;gt; film.'
    received = []

    class RecordingEngine(FakeEngine):
        def explain(self, text):
            received.append(text)
            return super().explain(normalize_review(text))

    with patch('sentiment.engine.SentimentEngine', RecordingEngine):
        app = AppTest.from_file(APP).run()
        app.text_area(key='review').input(raw_review).run()
        app.button(key='analyze').click().run()
        app.button(key='explain').click().run()
        assert not app.exception
        assert received == [raw_review]
