import numpy as np, matplotlib.pyplot as plt, gudhi as gd, pandas as pd, argparse, os
from .utils import up_laplacian

parser = argparse.ArgumentParser(description="Reproduce Fig. 4")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)
pts = np.random.rand(20, 2)

vr = gd.RipsComplex(points=pts, max_edge_length=1.5)
st_full = vr.create_simplex_tree(max_dimension=2)

two_sims = sorted([(s, f) for s, f in st_full.get_simplices() if len(s) == 3],
                  key=lambda x: x[1])

st_base = gd.SimplexTree()
for s, f in st_full.get_simplices():
    if len(s) < 3:
        st_base.insert(s, f)
st_base.initialize_filtration()

def eig_vals(tree):
    return np.linalg.eigvalsh(up_laplacian(tree, 1).toarray())

eig_old = eig_vals(st_base)
records = []

for step, (sigma, filt) in enumerate(two_sims[:50], 1):
    st_base.insert(sigma, filt)
    st_base.initialize_filtration()

    eig_new = eig_vals(st_base)
    eig_old_p = np.append(eig_old, 0)  # pad
    delta = np.abs(eig_new - eig_old_p)
    norm = np.sqrt(3)  # ℓ2‑norm of boundary of a 2‑simplex

    for d in delta:
        records.append((step, norm, d))

    eig_old = eig_new

# save data
os.makedirs("data", exist_ok=True)
np.save("data/points.npy", pts)
pd.DataFrame(records, columns=["step", "norm", "delta"]).to_csv(
    "data/eig_drift.csv", index=False
)

norms  = [r[1] for r in records]
deltas = [r[2] for r in records]
x = np.linspace(0, max(norms) * 1.05, 200)

plt.figure(figsize=(5,4))
plt.scatter(norms, deltas, s=25, edgecolors='k', alpha=0.6)
plt.plot(x, 2*x, 'r--', lw=1.4, label=r'$y = 2x$')
plt.xlabel(r'$\|\partial\sigma\|_2$')
plt.ylabel(r'$\Delta\lambda_j$')
plt.title('Eigenvalue drift vs. boundary norm (20‑vertex VR)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig_toy.png", dpi=300)
plt.savefig("fig_toy.pdf")
print("max Δλ / (2‖∂σ‖₂) =", max(deltas)/(2*norm))
