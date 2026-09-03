from pathlib import Path

from forge.artifact_hash import dir_size_bytes, hash_directory, hash_file


def test_hash_file_is_stable(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert hash_file(f) == hash_file(f)


def test_hash_file_changes_with_content(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h1 = hash_file(f)
    f.write_text("hello world")
    h2 = hash_file(f)
    assert h1 != h2


def test_hash_directory_is_order_independent(tmp_path: Path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "a.txt").write_text("one")
    (d1 / "b.txt").write_text("two")
    (d2 / "b.txt").write_text("two")
    (d2 / "a.txt").write_text("one")
    assert hash_directory(d1) == hash_directory(d2)


def test_hash_directory_changes_when_a_file_changes(tmp_path: Path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("one")
    h1 = hash_directory(d)
    (d / "a.txt").write_text("changed")
    h2 = hash_directory(d)
    assert h1 != h2


def test_dir_size_bytes_sums_file_sizes(tmp_path: Path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("12345")
    (d / "b.txt").write_text("1234567890")
    assert dir_size_bytes(d) == 15
