import torch


# nDCG2
# Σ_{j=1}^{k} (2^{rel_j} - 1) / log_2(j + 1)
def dcg2(inputs: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    inputs, _ = inputs.topk(k=k)
    if target in inputs:
        idx = torch.where(target == inputs)[0][0]
        return 1 / torch.log2(idx + 2)
    return torch.tensor(0.0)
