import numpy as np
import scipy.sparse as sp

def boundary_matrix(st, dim):
    """Signed incidence matrix ∂_{dim}."""
    sims_d   = [s for s in st.get_simplices() if len(s[0]) == dim + 1]
    sims_dm1 = [s for s in st.get_simplices() if len(s[0]) == dim]
    col = {tuple(sorted(s[0])): i for i, s in enumerate(sims_d)}
    row = {tuple(sorted(s[0])): i for i, s in enumerate(sims_dm1)}
    data, rows, cols = [], [], []
    for sig, _ in sims_d:
        sig = tuple(sorted(sig))
        c   = col[sig]
        for i in range(len(sig)):
            face = tuple(sorted(sig[:i] + sig[i+1:]))
            rows.append(row[face])
            cols.append(c)
            data.append((-1)**i)
    m, n = len(sims_dm1), len(sims_d)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=float)

def up_laplacian(st, k):
    """Compute the up‑persistent Laplacian Δ_k^up."""
    B = boundary_matrix(st, k + 1)
    return B @ B.T
