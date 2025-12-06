def test_torch_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA is not available."
