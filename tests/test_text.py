import html
import json
from pathlib import Path

import pytest

from sentiment.text import aggregate_words, highlight_review, normalize_review


def test_normalization_matches_training():
    assert normalize_review(' Good<br /> movie &amp;  great acting! ') == 'Good movie & great acting!'


def test_subwords_and_contractions_preserve_offsets_and_scores():
    text = "Unbelievably good! I can't wait."
    groups = aggregate_words(text, [(0, 0), (0, 2), (2, 12), (13, 17), (17, 18),
                                    (19, 20), (21, 24), (24, 25), (25, 26), (0, 0)],
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, .2, .3, .4, -.1, .1, .2, -.1, .1, 0])
    assert groups[0]['word'] == 'Unbelievably'
    assert groups[0]['score'] == pytest.approx(.5)
    assert groups[-1]['word'] == "can't"
    assert groups[-1]['token_positions'] == [6, 7, 8]
    assert sum(len(g['token_positions']) for g in groups) == 8


def test_highlights_escape_html_and_preserve_unanalyzed_tail():
    text = 'Good <script>alert(1)</script> ending.'
    groups = [{'word': 'Good', 'start': 0, 'end': 4, 'score': .4}]
    result = highlight_review(text, groups, analyzed_end=4)
    assert '<script>' not in result
    assert html.escape(text[4:]) in result
    assert 'unanalyzed' in result
    assert 'Supports' in result


def test_negative_evidence_is_named_opposing():
    result = highlight_review('dull', [{'start': 0, 'end': 4, 'score': -.3}], 4)
    assert 'Opposes' in result
    assert 'evidence-against' in result


def test_empty_highlights_still_preserve_review():
    assert 'Some &lt;text&gt;' in highlight_review('Some <text>', [], 11)


def test_unknown_offset_fails_instead_of_dropping_tokens():
    with pytest.raises(ValueError, match='align'):
        aggregate_words(' hello', [(0, 1)], [0], [.4])


def test_truncation_inside_word_does_not_highlight_or_duplicate_unseen_suffix():
    text = 'Unbelievably good'
    groups = aggregate_words(text, [(0, 0), (0, 2), (0, 0)], [1, 0, 1], [0, .4, 0])
    assert groups[0]['word'] == 'Un'
    assert groups[0]['end'] == 2
    rendered = highlight_review(text, groups, analyzed_end=2)
    assert rendered.count('believably') == 1
    assert '>Un</mark>' in rendered


def test_calibration_identifies_rounded_notebook_source():
    calibration = json.loads(Path('sentiment/calibration.json').read_text())
    assert calibration['seed'] == 13
    assert calibration['temperature'] == pytest.approx(1.7926)
    assert calibration['max_length'] == 256
    assert calibration['temperature_precision'] == 'rounded to 4 decimal places'
