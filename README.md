# Sentiment Studio

**A movie review is more than a thumbs up or down.** This project pairs a BERT sentiment classifier with word level attribution, so you can inspect which parts of a review supported its prediction and which pulled against it.

`IMDb reviews` &nbsp;→&nbsp; `BERT prediction` &nbsp;→&nbsp; `word contributions`

[Run the app](#run-the-app) · [Read the research notebook](#the-research) · [Understand the limits](#read-the-result-carefully)

---

## The experience

Paste an English movie review, or choose one of the three built in examples. The app returns a positive or negative prediction and a calibrated confidence estimate. Ask for an explanation to see the review marked up: green words support the chosen class; terracotta words oppose it. Longer reviews keep their unprocessed tail visible, so the reading window is clear.

The interface runs in Streamlit. It uses local model files and sends no review to an external AI API. A review does travel from your browser to the Streamlit server you are using.

| Part | What it does |
| :--- | :--- |
| `app.py` and `assets/style.css` | Review entry, result view, and responsive styling |
| `sentiment/engine.py` | Local prediction and Integrated Gradients |
| `sentiment/text.py` | Review normalization, token alignment, and safe highlighting |
| `sentiment/calibration.json` | Validation fitted temperature for the seed 13 model |
| `tests/` | Text, UI, and optional real model checks |
| Research notebook | Three seed training and evaluation pipeline |

## Run the app

Use Python 3.11 or 3.12. From this folder, create an environment and install the app dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at the local address Streamlit prints, usually `http://localhost:8501`. CPU inference works; a CUDA device is used when available. The first prediction loads the model. Explanations take longer, especially on CPU.

### The local checkpoint

Model weights are deliberately absent from this repository. To use the app, place the **complete seed 13 checkpoint** at:

```text
outputs/checkpoints/bert_seed_13/
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  selection.json
```

The [research notebook](<Explainable_Sentiment_Analysis_BERT_Research%20(1)%20(2).ipynb>) trains and saves it. Run the notebook from this project directory so its `outputs/` path matches the app. Its installation cell covers the additional research packages; training all three seeds needs a suitable GPU, substantial disk space, and downloads of the base model and datasets. If you already have the checkpoint from a previous run, copy it into the path above. The app will explain what is missing if the folder is incomplete.

`outputs/` is ignored by Git because it contains generated tables, figures, and model weights. The three existing weight files are about 438 MB **each**. Browser profiles, virtual environments, caches, and local secrets are also excluded.

## The research

The notebook uses IMDb reviews to compare fine tuned BERT with lexical baselines, calibrates predicted probabilities on validation data, and checks explanation behavior with matched budget perturbations. It records three training seeds (`13`, `42`, `123`); the app intentionally uses seed `13`, the notebook's chosen explanation model. Notebook execution outputs have been cleared for a clean source release. Running it will regenerate results locally.

The app reads the first **256 tokens**, including special tokens. It normalizes HTML entities, line breaks, and whitespace as the notebook does. Confidence uses temperature `1.7926`, recovered from a rounded notebook result; the original full precision value was not exported. Word colors are Integrated Gradients contributions to the predicted class **logit**, using attended PAD content tokens as a reference.

### Read the result carefully

- The classifier always picks positive or negative. A mixed review still gets one label.
- Confidence is a model estimate. It does not guarantee that the label is right.
- Highlights describe this model's response to the text. They do not show what caused a human reviewer to like a film.
- The app checks numerical completeness of each explanation and warns when its target is not reached. Numerical convergence does not prove an explanation is faithful in everyday language.
- IMDb movie reviews are a narrow domain; these results do not establish performance on other topics or languages.

## Check the code

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Most tests exercise text handling and the Streamlit flow. The integration test loads the real seed 13 checkpoint and runs an explanation; it needs the optional model libraries and checkpoint in place. A successful UI test alone does not establish classifier accuracy.

The source notebook and app are here to inspect and run locally. Model files, generated results, and dataset copies belong in your own working directory, outside Git history.

## Contributing and license

The [contribution guide](CONTRIBUTING.md) covers tests that run without a checkpoint. Project code and documentation are under the [MIT License](LICENSE). This does not license IMDb data, the BERT base model, or any checkpoint you train.
