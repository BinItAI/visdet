import pytest
import torch

from visdet.apis import inference as inference_api


def test_parse_inference_devices_single_device():
    assert inference_api._parse_inference_devices("cpu") == ("cpu", None)
    assert inference_api._parse_inference_devices("cuda:0") == ("cuda:0", None)


def test_parse_inference_devices_auto_cuda():
    primary, device_ids = inference_api._parse_inference_devices("cuda")

    if not torch.cuda.is_available():
        assert (primary, device_ids) == ("cpu", None)
        return

    assert primary == "cuda:0"
    if torch.cuda.device_count() > 1:
        assert device_ids == list(range(torch.cuda.device_count()))
    else:
        assert device_ids is None


def test_parse_inference_devices_multi_device():
    assert inference_api._parse_inference_devices("cuda:0,1") == ("cuda:0", [0, 1])
    assert inference_api._parse_inference_devices(["cuda:0", "cuda:1"]) == ("cuda:0", [0, 1])
    assert inference_api._parse_inference_devices([0, 2, 3]) == ("cuda:0", [0, 2, 3])


def test_parse_inference_devices_rejects_non_cuda_sequences():
    with pytest.raises(ValueError):
        inference_api._parse_inference_devices(["cpu", "cuda:0"])  # type: ignore[arg-type]
