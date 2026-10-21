import pytest

import data_loader


def test_missing_poems_file_raises_with_download_hint(tmp_path, monkeypatch):
    # Only the committed files are needed up to the poems step.
    lang_dir = tmp_path / "russian"
    lang_dir.mkdir()
    real = data_loader.DATA_DIR / "russian"
    for name in ("dataset_metadata.json", "labels.npy"):
        (lang_dir / name).write_bytes((real / name).read_bytes())
    import numpy as np
    np.save(lang_dir / "embeddings_gemini.npy", np.zeros((5800, 2), dtype=np.float32))
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="download_data.py"):
        data_loader.load_dataset("russian", include_poems=True)

    data = data_loader.load_dataset("russian")
    assert "poems" not in data
