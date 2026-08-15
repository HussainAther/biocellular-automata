import numpy as np
import tempfile
import os

from utils.io import save_history_npz, load_history_npz


def test_npz_save_load_roundtrip():
    history = np.random.randint(0, 2, size=(10, 20), dtype=int)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.npz")
        save_history_npz(history, path)
        loaded = load_history_npz(path)

    assert np.array_equal(history, loaded)

