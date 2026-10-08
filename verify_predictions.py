"""Verify all exported samples against saved Kaggle predictions."""
from pathlib import Path
import numpy as np
from inference import load_model, predict

root = Path(__file__).resolve().parent
with np.load(root / "data/demo_samples.npz", allow_pickle=False) as data:
    model = load_model(root / "models/best_model.pt", data["classes"])
    actual = predict(model, data["X"])
    np.testing.assert_allclose(actual, data["probabilities"], rtol=1e-4, atol=1e-5)
    assert np.array_equal(
        actual >= data["thresholds"], data["probabilities"] >= data["thresholds"]
    )
    print(f"PASS: {len(actual)} samples; maximum difference "
          f"{np.abs(actual - data['probabilities']).max():.2e}; labels agree.")
