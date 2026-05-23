import torch
import logging

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    METRICS_DEVICE = torch.device("cuda")
else:
    METRICS_DEVICE = torch.device("cpu")

TOP_K_VALS = [1, 5, 10, 25, 50, 100, 250, 500, 1000]


def compute_spectral_metrics(tensor):
    """
    Compute spectral metrics on collected activations.
    We compute: RankMe, Participation Ratio (PR), AlphaReQ, and Top-K Variance. 
    All of these are based on the eigenvalues of the covariance matrix of the activations.
    Computation on GPU for efficiency.
    """
    N, D = tensor.shape
    if N < 2:
        return {"rankme": 0.0, "pr": 0.0, "alpha_req": 0.0, **{f"top_{k}_var": 0.0 for k in TOP_K_VALS}}

    try:
        centered = tensor.to(METRICS_DEVICE).to(torch.float64)
        centered = centered - centered.mean(dim=0, keepdim=True)

        if N <= D:
            mat = torch.mm(centered, centered.t())
        else:
            mat = torch.mm(centered.t(), centered)

        eigvals = torch.linalg.eigvalsh(mat)
        eigvals = torch.flip(eigvals, dims=[0])
        eigvals = torch.clamp(eigvals, min=0.0)

        total = eigvals.sum() + 1e-12
        p = eigvals / total
        p_safe = p[p > 1e-12]
        rankme = torch.exp(-(p_safe * torch.log(p_safe)).sum()).item()

        pr = (total ** 2 / ((eigvals ** 2).sum() + 1e-12)).item()

        s = eigvals.sqrt()
        valid = s > 1e-6
        if valid.sum() > 2:
            s_v = s[valid]
            ranks = torch.arange(1, len(s_v) + 1, dtype=torch.float64, device=METRICS_DEVICE)
            x = torch.log(ranks)
            y = torch.log(s_v)
            xm, ym = x.mean(), y.mean()
            slope = ((x - xm) * (y - ym)).sum() / (((x - xm) ** 2).sum() + 1e-12)
            alpha_req = -slope.item()
        else:
            alpha_req = 0.0

        top_k_vars = {k: (eigvals[:min(k, len(eigvals))].sum() / total).item() for k in TOP_K_VALS}

    except Exception as e:
        logger.warning(f"Spectral computation failed: {e}")
        return {"rankme": 1.0, "pr": 1.0, "alpha_req": 0.0, **{f"top_{k}_var": 0.0 for k in TOP_K_VALS}}

    return {
        "rankme": round(rankme, 4),
        "pr": round(pr, 4),
        "alpha_req": round(alpha_req, 6),
        **{f"top_{k}_var": round(top_k_vars[k], 4) for k in TOP_K_VALS},
    }
