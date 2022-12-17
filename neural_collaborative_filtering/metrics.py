import torch

def hit_rate(inputs: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    _, input_idxs = inputs.topk(k=k)
    if target in input_idxs:
        return torch.tensor(1.0)
    return torch.tensor(0.0)


# nDCG2
# Σ_{j=1}^{k} (2^{rel_j} - 1) / log_2(j + 1)
def dcg2(inputs: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    _, input_idxs = inputs.topk(k=k)
    if target in input_idxs:
        idx = torch.where(target == input_idxs)[0][0]
        return 1 / torch.log2((idx + 2).float())
    return torch.tensor(0.0)
