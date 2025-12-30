import os

from visdet.engine import dist


def test_get_rank_world_size_env_fallback(monkeypatch):
    # Ensure process group isn't initialized in this unit test
    assert not dist.is_distributed()

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert dist.get_rank() == 3
    assert dist.get_world_size() == 8
    assert dist.get_dist_info() == (3, 8)


def test_infer_launcher_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert dist.infer_launcher() == "pytorch"

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    assert dist.infer_launcher() == "slurm"

    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    assert dist.infer_launcher() == "mpi"

    monkeypatch.delenv("OMPI_COMM_WORLD_LOCAL_RANK", raising=False)
    assert dist.infer_launcher() == "none"


def test_master_only_decorator(monkeypatch):
    monkeypatch.setenv("RANK", "1")

    called = {"value": False}

    @dist.master_only
    def _fn():
        called["value"] = True

    _fn()
    assert called["value"] is False

    monkeypatch.setenv("RANK", "0")
    _fn()
    assert called["value"] is True
