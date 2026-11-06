"""Run with: python -m streamlit run app.py"""

import html
import json
import logging
from pathlib import Path

import streamlit as st

from sentiment.engine import MAX_CHARACTERS, SentimentEngine
from sentiment.text import highlight_review, normalize_review

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Sentiment Studio · Read between the words', page_icon='✳',
                   layout='wide', initial_sidebar_state='collapsed')
st.markdown('<style>' + (ROOT / 'assets/style.css').read_text(encoding='utf-8') + '</style>',
            unsafe_allow_html=True)

EXAMPLES = {
    'positive': 'A wonderful film that stayed with me long after the credits. The performances '
                'are heartfelt, the photography is beautiful, and every scene feels carefully considered. '
                'I loved every minute of it.',
    'negative': 'A painfully dull movie with a predictable story. The dialogue felt forced, '
                'the characters were forgettable, and the ending was a complete disappointment. '
                'I wish I had spent my evening watching something else.',
    'mixed': 'The acting was exquisite and the opening was full of promise. But the story '
             'became painfully dull and predictable, and the beautiful photography could not save it.',
}


@st.cache_resource(show_spinner=False)
def get_engine():
    return SentimentEngine()


def invalidate_result():
    st.session_state.prediction = None
    st.session_state.explanation = None
    st.session_state.setup_error = None


def set_review(value):
    st.session_state.review = value
    invalidate_result()


for name, default in {'review': '', 'prediction': None, 'explanation': None, 'setup_error': None}.items():
    if name not in st.session_state:
        st.session_state[name] = default

st.markdown('''
<nav class="masthead" aria-label="Brand">
  <div class="brand"><span class="brand-symbol" aria-hidden="true">✳</span> sentiment<span class="brand-light">studio</span></div>
  <div class="masthead-note">WORDS IN. <span>UNDERSTANDING OUT.</span></div>
</nav>
<header class="hero">
  <div><div class="eyebrow"><span class="tiny-dot"></span> AN EXPLORATION IN LANGUAGE & FEELING</div>
    <h1>Every review<br>has a <em>feeling.</em></h1></div>
  <div class="hero-aside"><span class="aside-rule"></span>
    <p>Go beyond positive or negative.<br>Find the words behind the feeling.</p>
    <span class="edition">MOVIE REVIEWS &nbsp; / &nbsp; EXPLAINABLE AI</span></div>
</header>
''', unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap='large')
with left:
    with st.container(border=True, key='editor_panel'):
        st.markdown('<div class="panel-kicker"><span>01 / YOUR WORDS</span><span>MOVIE REVIEW</span></div>'
                    '<h2 class="panel-title">What did you think?</h2>', unsafe_allow_html=True)
        st.caption('Write a review, or start with a little inspiration.')
        examples = st.columns(3, gap='small')
        for col, (key, label) in zip(examples, [('positive', 'Loved it ↗'), ('negative', 'Not a fan ↘'), ('mixed', 'Mixed feelings ↔')]):
            col.button(label, key=f'example_{key}', on_click=set_review,
                       args=(EXAMPLES[key],), use_container_width=True)
        review = st.text_area('Your movie review', key='review', height=195,
                              max_chars=MAX_CHARACTERS, label_visibility='collapsed',
                              placeholder='The kind of film you keep thinking about on the way home…',
                              on_change=invalidate_result)
        action, clear = st.columns([3, 1], gap='small')
        submitted = action.button('Analyze review  ↗', key='analyze', type='primary', use_container_width=True)
        clear.button('Clear', key='clear', on_click=set_review, args=('',), use_container_width=True)
        st.markdown('<div class="editor-note">Your words stay on this app’s server. No external AI API.</div>',
                    unsafe_allow_html=True)
        if submitted:
            if not normalize_review(review):
                st.warning('Write a movie review first, or choose an example above.')
            else:
                invalidate_result()
                try:
                    with st.spinner('Reading your review… The first analysis loads the model.'):
                        st.session_state.prediction = get_engine().predict(review)
                except ImportError as exc:
                    st.session_state.setup_error = 'dependencies'
                    logging.getLogger(__name__).info('Model dependency unavailable: %s', exc)
                except (OSError, ValueError, RuntimeError) as exc:
                    st.session_state.setup_error = str(exc)

with right:
    with st.container(border=True, key='result_panel'):
        st.markdown('<div class="panel-kicker"><span>02 / THE FEELING</span><span>MODEL PREDICTION</span></div>',
                    unsafe_allow_html=True)
        prediction = st.session_state.prediction
        error = st.session_state.setup_error
        if error:
            st.markdown('<h2 class="panel-title">One small setup step.</h2>', unsafe_allow_html=True)
            if error == 'dependencies':
                st.error('The model libraries are not installed in this Python environment yet.')
                st.code('python -m pip install -r requirements.txt', language='powershell')
                st.caption('Install the dependencies, restart Streamlit, then analyze your review again.')
            else:
                st.error('The review could not be analyzed.')
                st.caption(error)
        elif prediction:
            label = prediction['label']
            positive = prediction['probabilities']['positive'] * 100
            negative = prediction['probabilities']['negative'] * 100
            st.markdown(f'''
<section class="prediction reveal {label}" aria-label="Sentiment result">
  <span class="result-badge">{'↗' if label == 'positive' else '↘'} &nbsp; {label.upper()} SENTIMENT</span>
  <h2 class="verdict">{'A good impression.' if label == 'positive' else 'Not quite a hit.'}</h2>
  <div class="confidence">{prediction['confidence'] * 100:.1f}<span>%</span></div>
  <p class="confidence-label">model confidence in this prediction</p>
  <div class="prob-labels"><span>Negative <b>{negative:.1f}%</b></span><span>Positive <b>{positive:.1f}%</b></span></div>
  <div class="probability-track" role="img" aria-label="Negative {negative:.1f} percent; positive {positive:.1f} percent">
    <span class="negative-fill" style="width:{negative:.4f}%"></span><span class="positive-fill" style="width:{positive:.4f}%"></span>
  </div>
</section>''', unsafe_allow_html=True)
            if prediction['truncated']:
                st.info('This review is longer than the model’s reading window. The prediction uses '
                        'the first 256 tokens (pieces of words); the remaining text is not analyzed.')
            st.button('See the words behind it  ↗', key='explain', use_container_width=True,
                      disabled=st.session_state.explanation is not None)
            st.caption('Word explanations take a little longer, especially on a CPU.')
        else:
            st.markdown('''
<section class="empty-result">
  <div class="feeling-orbit" aria-hidden="true"><span class="orbit-line"></span><span class="orbit-glyph">“</span><i></i></div>
  <h2>A little perspective<br>on your words.</h2>
  <p>Your sentiment, confidence, and the words<br>that made a difference will appear here.</p>
  <span class="awaiting"><span class="tiny-dot"></span> WAITING FOR YOUR REVIEW</span>
</section>''', unsafe_allow_html=True)

if st.session_state.prediction and st.session_state.get('explain', False) and st.session_state.explanation is None:
    try:
        with st.spinner('Tracing the words behind the prediction…'):
            # Both operations receive the original input, so HTML entities are
            # decoded exactly once and attribution offsets match prediction text.
            st.session_state.explanation = get_engine().explain(st.session_state.review)
        st.rerun()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        st.error('The prediction is ready, but its word explanation could not be calculated.')
        st.caption(str(exc))

if st.session_state.explanation is not None:
    exp = st.session_state.explanation
    prediction = st.session_state.prediction
    with st.container(border=True, key='explanation_panel'):
        st.markdown('<div class="panel-kicker"><span>03 / BETWEEN THE WORDS</span><span>WORD CONTRIBUTIONS</span></div>'
                    '<h2 class="panel-title">A closer reading.</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="explanation-intro">Colors show how words contributed to the '
                    f'<strong>{html.escape(prediction["label"])}</strong> prediction.</p>'
                    '<div class="legend"><span><i class="swatch support"></i> Supports prediction</span>'
                    '<span><i class="swatch oppose"></i> Opposes prediction</span>'
                    '<span>Darker = stronger contribution</span></div>', unsafe_allow_html=True)
        st.markdown(highlight_review(prediction['text'], exp['groups'], exp['analyzed_end']), unsafe_allow_html=True)
        if prediction['truncated']:
            st.caption('Faded text is outside the model’s reading window and did not influence this prediction.')
        if not exp['converged']:
            st.warning('These word scores have not reached the numerical accuracy target. Treat the highlights as approximate.')
        st.caption('Highlights describe this model’s response. They do not prove why a person liked or disliked a film.')

if st.session_state.prediction:
    with st.expander('About this analysis & download'):
        st.write('This BERT model was trained on IMDb movie reviews. It always chooses positive or '
                 'negative, even when a review is mixed. Confidence is an estimate, not a guarantee.')
        st.write('The model reads up to 256 tokens, including special tokens. Confidence uses a '
                 'validation-fitted temperature of 1.7926, recovered from the notebook’s rounded output.')
        if st.session_state.explanation:
            exp = st.session_state.explanation
            st.write(f'Integrated Gradients targets the predicted-class logit using an attended PAD '
                     f'reference. Integration steps: {exp["steps"]}. Absolute completeness residual: '
                     f'{exp["absolute_residual"]:.4f}; target: {exp["tolerance"]:.4f}. '
                     'Numerical convergence does not establish semantic faithfulness.')
        st.download_button('Download analysis ↓',
                           data=json.dumps({'prediction': st.session_state.prediction,
                                            'explanation': st.session_state.explanation}, indent=2),
                           file_name='sentiment-analysis.json', mime='application/json')

st.markdown('''
<section class="reading-guide" aria-label="How it works">
  <div class="guide-title"><span class="eyebrow">THE IDEA IS SIMPLE</span><h2>Less guessing.<br>More understanding.</h2></div>
  <div class="guide-step"><span>01</span><h3>Put it into words.</h3><p>A rave, a rant, or something in between. Start with an English movie review.</p></div>
  <div class="guide-step"><span>02</span><h3>Find the sentiment.</h3><p>A model trained on movie reviews reads the text and estimates its overall feeling.</p></div>
  <div class="guide-step"><span>03</span><h3>Look a little closer.</h3><p>Explore which words supported the prediction, and which pulled the other way.</p></div>
</section>
<footer class="footer"><span>✳ &nbsp; SENTIMENT STUDIO</span><span>A SMALL WINDOW INTO HOW LANGUAGE MODELS READ.</span><span>BUILT FOR CURIOSITY.</span></footer>
''', unsafe_allow_html=True)
