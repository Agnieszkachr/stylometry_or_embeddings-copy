import hashlib
import urllib.request

import pytest

import download_data


CONTENT = b"fresh payload"


def _manifest():
    md5 = hashlib.md5(CONTENT).hexdigest()
    return {
        "data/russian/a.bin": {"zenodo_name": "russian_a.bin", "size_mb": 1,
                               "md5": md5, "required": True, "description": ""},
        "data/italian/b.bin": {"zenodo_name": "italian_b.bin", "size_mb": 1,
                               "md5": md5, "required": True, "description": ""},
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run download_data.main() against tmp_path with a fake manifest."""
    monkeypatch.setattr(download_data, "__file__", str(tmp_path / "download_data.py"))
    monkeypatch.setattr(download_data, "FILES", _manifest())
    fetched = []

    def fake_urlretrieve(url, dest):
        fetched.append(url)
        with open(dest, "wb") as f:
            f.write(CONTENT)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["download_data.py", *argv])
        return download_data.main()

    return tmp_path, fetched, run


def test_force_replaces_corrupt_existing_file(sandbox):
    root, fetched, run = sandbox
    for rel in download_data.FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"corrupt")

    assert run() == 1          # verification flags the corrupt files ...
    assert fetched == []       # ... without re-fetching them

    assert run("--force") == 0
    assert len(fetched) == 2
    for rel in download_data.FILES:
        assert (root / rel).read_bytes() == CONTENT


def test_interrupted_download_leaves_no_file(sandbox, monkeypatch):
    root, _, run = sandbox

    def broken_urlretrieve(url, dest):
        with open(dest, "wb") as f:
            f.write(b"trunc")
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlretrieve", broken_urlretrieve)
    assert run("--russian") == 1
    data_dir = root / "data" / "russian"
    assert not (data_dir / "a.bin").exists()
    assert list(data_dir.iterdir()) == []


def test_failed_force_download_keeps_existing_file(sandbox, monkeypatch):
    root, _, run = sandbox
    target = root / "data" / "russian" / "a.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(CONTENT)

    def broken_urlretrieve(url, dest):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlretrieve", broken_urlretrieve)
    assert run("--russian", "--force") == 1
    assert target.read_bytes() == CONTENT


def test_both_corpus_flags_select_both_corpora(sandbox):
    root, fetched, run = sandbox
    assert run("--russian", "--italian") == 0
    assert len(fetched) == 2
    assert (root / "data" / "russian" / "a.bin").exists()
    assert (root / "data" / "italian" / "b.bin").exists()


def test_minimal_and_all_are_mutually_exclusive(sandbox):
    _, _, run = sandbox
    with pytest.raises(SystemExit):
        run("--minimal", "--all")
