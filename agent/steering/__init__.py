"""Level-3 activation steering (Spec B).

Pure-stdlib vector math + composition live here; torch is imported ONLY inside
hooks.py (the real path) and lazily, so importing this package never requires torch.
"""
from agent.steering.vectors import (
    Vector, add, cosine, diff_of_means, dot, mean_vectors, mock_vector, norm,
    normalize, scale, sub,
)

__all__ = [
    "Vector", "add", "cosine", "diff_of_means", "dot", "mean_vectors",
    "mock_vector", "norm", "normalize", "scale", "sub",
]
