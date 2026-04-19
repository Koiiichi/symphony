import subprocess

from core.rollback import create_checkpoint, discard_checkpoint, restore_checkpoint


def _run(cwd, *args):
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path):
    _run(tmp_path, "git", "init")
    _run(tmp_path, "git", "config", "user.email", "test@example.com")
    _run(tmp_path, "git", "config", "user.name", "Test User")
    (tmp_path / "a.txt").write_text("base\n")
    _run(tmp_path, "git", "add", "a.txt")
    _run(tmp_path, "git", "commit", "-m", "init")


def test_create_checkpoint_skips_dirty_tree(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")

    token = create_checkpoint(tmp_path, "pre-pass")

    assert token is None
    assert (tmp_path / "a.txt").read_text() == "dirty\n"


def test_restore_head_checkpoint_reverts_project_subtree(tmp_path):
    _init_repo(tmp_path)
    token = create_checkpoint(tmp_path, "pre-pass")
    assert token and token.startswith("HEAD:")

    (tmp_path / "a.txt").write_text("changed\n")
    (tmp_path / "new.txt").write_text("new\n")

    restored = restore_checkpoint(tmp_path, token)
    assert restored is True
    assert (tmp_path / "a.txt").read_text() == "base\n"
    assert not (tmp_path / "new.txt").exists()


def test_discard_head_checkpoint_is_noop(tmp_path):
    _init_repo(tmp_path)
    token = create_checkpoint(tmp_path, "pre-pass")
    assert token and token.startswith("HEAD:")

    assert discard_checkpoint(tmp_path, token) is True
