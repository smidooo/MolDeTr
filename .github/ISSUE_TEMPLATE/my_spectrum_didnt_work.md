---
name: My spectrum didn't work
about: Predictions look wrong, or your input was rejected
title: "[spectrum] "
labels: question
---

MolDeTr expects a very specific input, and most "wrong prediction" reports come down to the input
format or an out-of-scope spectrum. Please skim [`docs/INPUT_FORMAT.md`](../docs/INPUT_FORMAT.md) and
[`docs/SCOPE.md`](../docs/SCOPE.md) first, then fill this in.

**Your input**
- Number of points (shape):
- dtype (real or complex):
- Digital resolution (points/Hz), if known:
- Field strength (MHz):
- Is there a solvent/water peak in the window? (MolDeTr does **not** do water suppression)

**Validator output**
Run this and paste the result — it reports the exact problem:
```python
import numpy as np
from moldetr.validation import validate_spectrum
a = np.load("your_window.npz", allow_pickle=True)["spectrum_padded"]  # or your array key
validate_spectrum(a, points_per_hz=5.12)
```

**What you saw vs expected**
The predicted δ / max J / proton count, and what you expected for that spectrum.

**Checklist**
- [ ] My window is 6144 points at 5.12 points/Hz (a 1200 Hz window) — see INPUT_FORMAT.md
- [ ] Every coupling partner of an in-window peak is also inside the window
- [ ] My spectrum is in scope (1-D ¹H, single compound, no water/solvent peak) — see SCOPE.md
