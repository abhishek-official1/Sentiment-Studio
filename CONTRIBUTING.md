# Contributing to Sentiment Studio

The app needs a local seed-13 checkpoint for real inference, but text and interface tests can run without one.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Keep generated checkpoints, IMDb copies, browser profiles, and notebook outputs out of Git. Use short invented reviews in tests. If you change token alignment, calibration, or attribution, explain the expected behavior with a hand-worked example or the source notebook cell, and test the change without assuming that a plausible-looking highlight is correct.

Open an issue for a shareable bug report. Use [SECURITY.md](SECURITY.md) for security-sensitive findings.
