import torch

def stfpm_loss(teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
    return torch.sum((teacher_features - student_features) ** 2, 1).mean()

def stfpm_cosine_loss(teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
    # shape: B, C, H, W
    cosine_loss = torch.nn.CosineSimilarity()
    return torch.mean(
        1 - cosine_loss(
            student_features.reshape(student_features.shape[0], -1),
            teacher_features.reshape(teacher_features.shape[0], -1)
        )
    )