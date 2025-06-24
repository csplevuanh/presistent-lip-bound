# Lipschitz Bound Demo for Persistent Laplacian

This repository reproduces Figure 4 in the paper  
**“Lipschitz Bounds for Persistent Laplacian Eigenvalues under One‑Simplex Insertions.”**

```bash
# quick run (Python ≥ 3.9)
pip install -r requirements.txt
python -m src.experiment          # saves fig_toy.png / fig_toy.pdf
```

The script builds a 20‑vertex Vietoris–Rips filtration in ℝ², inserts the
first 50 two‑simplices one‑by‑one, and plots every eigen‑drift
Δλ<sub>j</sub> against the bound 2‖∂σ‖₂.

<p align="center">
  <img src="fig_toy.png" width="420">
</p>

## Layout

| path | purpose |
|------|---------|
| **src/experiment.py** | main driver: generate data & plot |
| **src/utils.py**      | sparse boundary matrix + Laplacian helpers |
| **data/**             | will hold points *.npy* and eig\_drift *.csv* after the run |
