"""Training-compatible normalization and safe, offset-based visualization."""

import html
import re
from bisect import bisect_right

WORD_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def normalize_review(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def aggregate_words(text, offsets, special_mask, scores):
    """Sum WordPiece scores into original word/contraction/punctuation spans."""
    if not (len(offsets) == len(special_mask) == len(scores)):
        raise ValueError('Token offsets, special masks, and scores must align.')
    spans = list(WORD_RE.finditer(text))
    starts = [span.start() for span in spans]
    groups = [dict(word=s.group(), start=s.start(), end=s.end(),
                   token_positions=[], score=0.0) for s in spans]
    for pos, ((start, end), special, score) in enumerate(zip(offsets, special_mask, scores)):
        if special or end <= start:
            continue
        index = bisect_right(starts, start) - 1
        if index < 0 or not (spans[index].start() <= start < spans[index].end()):
            raise ValueError(f'Cannot align token at character offsets {(start, end)}.')
        groups[index]['token_positions'].append(pos)
        groups[index]['score'] += float(score)
    aligned = [group for group in groups if group['token_positions']]
    # A reading window can stop between WordPieces inside the final word.
    # Display only the characters actually represented by included tokens.
    analyzed_end = max((end for (start, end), sp in zip(offsets, special_mask)
                        if not sp and end > start), default=0)
    for group in aligned:
        group['end'] = min(group['end'], analyzed_end)
        group['word'] = text[group['start']:group['end']]
    return aligned


def highlight_review(text, groups, analyzed_end):
    """Escape every user character; retain the entire unprocessed tail."""
    scale = max((abs(g['score']) for g in groups), default=0) or 1
    parts, cursor = [], 0
    for group in groups:
        start, end, score = group['start'], group['end'], group['score']
        parts.append(html.escape(text[cursor:start]))
        direction = 'for' if score >= 0 else 'against'
        meaning = 'Supports' if score >= 0 else 'Opposes'
        opacity = .08 + .32 * abs(score) / scale
        parts.append(
            f'<mark class="evidence-{direction}" style="--strength:{opacity:.3f}" '
            f'title="{meaning} the prediction: {score:+.4f}">'
            f'{html.escape(text[start:end])}</mark>')
        cursor = end
    parts.append(html.escape(text[cursor:analyzed_end]))
    if analyzed_end < len(text):
        parts.append('<span class="unanalyzed" title="Outside the model’s reading window">'
                     + html.escape(text[analyzed_end:]) + '</span>')
    return '<div class="highlight-review">' + ''.join(parts) + '</div>'
