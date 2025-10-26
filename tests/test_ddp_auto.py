"""Unit tests for automatic DDP training functionality.

Tests cover GPU detection, environment setup, and multi-process spawning logic.
Tests use mocks to avoid requiring actual multi-GPU hardware.
"""

import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from visdet.ddp.auto_ddp import (
    _find_free_port,
    _worker_fn,
    auto_ddp_train,
    setup_distributed_env,
)


class TestSetupDistributedEnv(unittest.TestCase):
    """Tests for setup_distributed_env function."""

    def setUp(self) -> None:
        """Save original environment variables."""
        self.original_env = {
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT"),
        }

    def tearDown(self) -> None:
        """Restore original environment variables."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_setup_distributed_env_sets_variables(self) -> None:
        """Test that setup_distributed_env sets all required environment variables."""
        rank = 1
        world_size = 4
        port = 29500

        setup_distributed_env(rank, world_size, port)

        assert os.environ["RANK"] == "1"
        assert os.environ["LOCAL_RANK"] == "1"
        assert os.environ["WORLD_SIZE"] == "4"
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "29500"

    def test_setup_distributed_env_custom_port(self) -> None:
        """Test setup_distributed_env with custom port."""
        rank = 0
        world_size = 2
        port = 30000

        setup_distributed_env(rank, world_size, port)

        assert os.environ["MASTER_PORT"] == "30000"

    def test_setup_distributed_env_multiple_calls(self) -> None:
        """Test that setup_distributed_env correctly updates variables on multiple calls."""
        # First call
        setup_distributed_env(0, 2, 29500)
        assert os.environ["RANK"] == "0"

        # Second call
        setup_distributed_env(1, 2, 29501)
        assert os.environ["RANK"] == "1"
        assert os.environ["MASTER_PORT"] == "29501"


class TestFindFreePort(unittest.TestCase):
    """Tests for _find_free_port function."""

    def test_find_free_port_returns_integer(self) -> None:
        """Test that _find_free_port returns an integer."""
        port = _find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_find_free_port_returns_different_ports(self) -> None:
        """Test that multiple calls to _find_free_port may return different ports."""
        port1 = _find_free_port()
        port2 = _find_free_port()
        # Ports should be different or same (both are valid)
        assert isinstance(port1, int) and isinstance(port2, int)
        assert port1 > 0 and port2 > 0

    def test_find_free_port_is_valid_range(self) -> None:
        """Test that returned port is in valid range."""
        port = _find_free_port()
        # Port should be in valid range (1024-65535 for non-privileged)
        assert 1024 < port < 65536


class TestAutoAutoddpTrain(unittest.TestCase):
    """Tests for auto_ddp_train function."""

    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_single_gpu_training(self, mock_device_count: MagicMock) -> None:
        """Test that single GPU training calls train_fn directly."""
        mock_device_count.return_value = 1
        train_fn = MagicMock()

        auto_ddp_train(train_fn, "arg1", kwarg1="value1")

        # Train function should be called directly
        train_fn.assert_called_once_with("arg1", kwarg1="value1")

    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_zero_gpu_training(self, mock_device_count: MagicMock) -> None:
        """Test that zero GPUs is treated same as single GPU (no DDP)."""
        mock_device_count.return_value = 0
        train_fn = MagicMock()

        auto_ddp_train(train_fn)

        # Train function should be called directly
        train_fn.assert_called_once()

    @patch("visdet.ddp.auto_ddp.mp.spawn")
    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_multi_gpu_training_spawns_processes(self, mock_device_count: MagicMock, mock_spawn: MagicMock) -> None:
        """Test that multi-GPU training spawns worker processes."""
        mock_device_count.return_value = 2
        train_fn = MagicMock()

        auto_ddp_train(train_fn, "arg1", kwarg1="value1")

        # mp.spawn should be called with correct nprocs
        mock_spawn.assert_called_once()
        call_args = mock_spawn.call_args
        assert call_args.kwargs["nprocs"] == 2

    @patch("visdet.ddp.auto_ddp.mp.spawn")
    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_multi_gpu_training_passes_arguments(self, mock_device_count: MagicMock, mock_spawn: MagicMock) -> None:
        """Test that arguments are correctly passed to worker function."""
        mock_device_count.return_value = 4
        train_fn = MagicMock()
        test_args = ("arg1", "arg2")
        test_kwargs = {"kwarg1": "value1", "kwarg2": "value2"}

        auto_ddp_train(train_fn, *test_args, **test_kwargs)

        # Verify spawn was called with correct nprocs
        mock_spawn.assert_called_once()
        call_kwargs = mock_spawn.call_args.kwargs
        assert call_kwargs["nprocs"] == 4

        # Verify the first positional arg to spawn is the worker function
        call_args = mock_spawn.call_args.args
        assert len(call_args) > 0  # Worker function is first arg
        # The args keyword argument contains (world_size, port, train_fn, train_args, train_kwargs)
        args_kwarg = call_kwargs.get("args", ())
        if args_kwarg:
            assert args_kwarg[2] == train_fn  # train_fn should be 3rd element
            assert args_kwarg[3] == test_args
            assert args_kwarg[4] == test_kwargs

    @patch("visdet.ddp.auto_ddp.mp.get_start_method")
    @patch("visdet.ddp.auto_ddp.mp.set_start_method")
    @patch("visdet.ddp.auto_ddp.mp.spawn")
    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_multiprocessing_start_method(
        self,
        mock_device_count: MagicMock,
        mock_spawn: MagicMock,
        mock_set_start_method: MagicMock,
        mock_get_start_method: MagicMock,
    ) -> None:
        """Test that spawn start method is set for multi-GPU training."""
        mock_device_count.return_value = 2
        mock_get_start_method.return_value = None  # Not set yet
        train_fn = MagicMock()

        auto_ddp_train(train_fn)

        # set_start_method should be called with 'spawn'
        mock_set_start_method.assert_called_once_with("spawn")


class TestWorkerFn(unittest.TestCase):
    """Tests for _worker_fn function."""

    @patch("visdet.ddp.auto_ddp.torch.distributed.destroy_process_group")
    @patch("visdet.ddp.auto_ddp.torch.distributed.is_initialized")
    @patch("visdet.ddp.auto_ddp.torch.distributed.init_process_group")
    @patch("visdet.ddp.auto_ddp.setup_distributed_env")
    def test_worker_fn_calls_train_function(
        self,
        mock_setup_env: MagicMock,
        mock_init_group: MagicMock,
        mock_is_initialized: MagicMock,
        mock_destroy_group: MagicMock,
    ) -> None:
        """Test that worker function calls the training function."""
        mock_is_initialized.return_value = True
        train_fn = MagicMock()

        _worker_fn(
            rank=0,
            world_size=2,
            port=29500,
            train_fn=train_fn,
            train_args=("arg1",),
            train_kwargs={"kwarg1": "value1"},
        )

        # Training function should be called
        train_fn.assert_called_once_with("arg1", kwarg1="value1")

    @patch("visdet.ddp.auto_ddp.torch.distributed.destroy_process_group")
    @patch("visdet.ddp.auto_ddp.torch.distributed.is_initialized")
    @patch("visdet.ddp.auto_ddp.torch.distributed.init_process_group")
    @patch("visdet.ddp.auto_ddp.setup_distributed_env")
    def test_worker_fn_initializes_process_group(
        self,
        mock_setup_env: MagicMock,
        mock_init_group: MagicMock,
        mock_is_initialized: MagicMock,
        mock_destroy_group: MagicMock,
    ) -> None:
        """Test that worker function initializes the process group."""
        mock_is_initialized.return_value = True
        train_fn = MagicMock()

        _worker_fn(
            rank=1,
            world_size=4,
            port=29501,
            train_fn=train_fn,
            train_args=(),
            train_kwargs={},
        )

        # init_process_group should be called with correct parameters
        mock_init_group.assert_called_once()
        call_kwargs = mock_init_group.call_args.kwargs
        assert call_kwargs["backend"] == "nccl"
        assert call_kwargs["rank"] == 1
        assert call_kwargs["world_size"] == 4

    @patch("visdet.ddp.auto_ddp.torch.distributed.destroy_process_group")
    @patch("visdet.ddp.auto_ddp.torch.distributed.is_initialized")
    @patch("visdet.ddp.auto_ddp.torch.distributed.init_process_group")
    @patch("visdet.ddp.auto_ddp.setup_distributed_env")
    def test_worker_fn_destroys_process_group(
        self,
        mock_setup_env: MagicMock,
        mock_init_group: MagicMock,
        mock_is_initialized: MagicMock,
        mock_destroy_group: MagicMock,
    ) -> None:
        """Test that worker function destroys process group in finally block."""
        mock_is_initialized.return_value = True
        train_fn = MagicMock()

        _worker_fn(
            rank=0,
            world_size=2,
            port=29500,
            train_fn=train_fn,
            train_args=(),
            train_kwargs={},
        )

        # destroy_process_group should be called in finally block
        mock_destroy_group.assert_called_once()

    @patch("visdet.ddp.auto_ddp.torch.distributed.destroy_process_group")
    @patch("visdet.ddp.auto_ddp.torch.distributed.is_initialized")
    @patch("visdet.ddp.auto_ddp.torch.distributed.init_process_group")
    @patch("visdet.ddp.auto_ddp.setup_distributed_env")
    def test_worker_fn_exception_handling(
        self,
        mock_setup_env: MagicMock,
        mock_init_group: MagicMock,
        mock_is_initialized: MagicMock,
        mock_destroy_group: MagicMock,
    ) -> None:
        """Test that worker function handles exceptions and cleans up."""
        mock_is_initialized.return_value = True
        train_fn = MagicMock(side_effect=RuntimeError("Training failed"))

        with self.assertRaises(RuntimeError):
            _worker_fn(
                rank=0,
                world_size=2,
                port=29500,
                train_fn=train_fn,
                train_args=(),
                train_kwargs={},
            )

        # Process group should still be destroyed even on exception
        mock_destroy_group.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for DDP functionality."""

    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_single_gpu_end_to_end(self, mock_device_count: MagicMock) -> None:
        """Test single GPU training end-to-end."""
        mock_device_count.return_value = 1
        results = []

        def mock_train():
            results.append("trained")

        auto_ddp_train(mock_train)

        assert results == ["trained"]

    @patch("visdet.ddp.auto_ddp.mp.spawn")
    @patch("visdet.ddp.auto_ddp.torch.cuda.device_count")
    def test_multi_gpu_end_to_end(self, mock_device_count: MagicMock, mock_spawn: MagicMock) -> None:
        """Test multi-GPU training end-to-end."""
        mock_device_count.return_value = 2
        train_fn = MagicMock()

        auto_ddp_train(train_fn)

        assert mock_spawn.called


if __name__ == "__main__":
    unittest.main()
