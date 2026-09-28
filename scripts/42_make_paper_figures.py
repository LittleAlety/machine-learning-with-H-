#!/usr/bin/env python3
"""Generate publication figures for the v2 comparison paper from real project data."""
import json, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures_v2")
os.makedirs(OUT, exist_ok=True)

# Palette
TERRA = "#b05c34"; TERRA_D = "#7c351c"; GOLD = "#c2924a"
GREEN = "#7d8f6a"; SLATE = "#6b7a86"; INK = "#3b2f26"; GREY = "#b9ac99"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c9bba6", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": "#7a6a58", "ytick.color": "#7a6a58",
    "axes.grid": True, "grid.color": "#efe7da", "grid.linewidth": 0.8,
    "figure.dpi": 200, "savefig.bbox": "tight"
})

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def load_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def n_elements(comp):
    import re
    return len(re.findall(r"[A-Z][a-z]?", comp))

# ---------- Fig 1: parity ----------
preds = load_json(os.path.join(ROOT, "web", "assets", "predictions_all.json"))
yt = np.array([r["dG_true"] for r in preds]); yp = np.array([r["dG_pred"] for r in preds])
mae = np.mean(np.abs(yt - yp)); r2 = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
fig, ax = plt.subplots(figsize=(5.6,5.2))
ax.scatter(yt, yp, s=10, alpha=0.45, color=TERRA, edgecolors="none")
lo, hi = min(yt.min(), yp.min())-0.05, max(yt.max(), yp.max())+0.05
ax.plot([lo,hi],[lo,hi], color=SLATE, lw=1.4, ls="--")
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_xlabel(r"DFT $\Delta G_{H*}$ (eV)"); ax.set_ylabel(r"Predicted $\Delta G_{H*}$ (eV)")
ax.set_title("H* main model: out-of-fold parity")
ax.text(0.04,0.93,f"n = {len(preds)} surfaces\nMAE = {mae:.3f} eV\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffdf8", ec="#e7dccb"))
fig.savefig(os.path.join(OUT,"fig1_parity.png")); plt.close(fig)

# ---------- Fig 2: SHAP ----------
shap = load_json(os.path.join(ROOT,"web","assets","shap_global.json"))["top15"][:15]
names = [s["feature"] for s in shap][::-1]; vals = [s["mean_abs_shap"] for s in shap][::-1]
fig, ax = plt.subplots(figsize=(6.4,5.6))
bars = ax.barh(names, vals, color=TERRA, alpha=0.85)
bars[-1].set_color(TERRA_D); bars[-2].set_color(GOLD)
ax.set_xlabel("mean(|SHAP value|) (eV)")
ax.set_title("Global feature attribution (v3, 84 descriptors)")
fig.savefig(os.path.join(OUT,"fig2_shap.png")); plt.close(fig)

# ---------- Fig 3: volcano ----------
x = np.array([r["dG_pred"] for r in preds]); y = -np.abs(x)
pure = np.array([n_elements(r["comp"])==1 for r in preds])
sym = np.abs(x).max()
fig, ax = plt.subplots(figsize=(6.6,5.0))
ax.axvspan(-0.1,0.1, color=GREEN, alpha=0.16)
ax.axvspan(-0.2,-0.1, color=GOLD, alpha=0.08); ax.axvspan(0.1,0.2, color=GOLD, alpha=0.08)
ax.scatter(x[~pure], y[~pure], s=9, alpha=0.4, color=GOLD, edgecolors="none", label="Alloy")
ax.scatter(x[pure], y[pure], s=22, alpha=0.9, color=TERRA_D, edgecolors="white", linewidths=0.4, label="Pure metal")
ax.axvline(0, color=TERRA, lw=1.4, alpha=0.6)
# dashed outline
tt = np.linspace(-sym,sym,100); ax.plot(tt,-np.abs(tt), color=TERRA, lw=1.4, ls=":", alpha=0.6)
ax.set_xlim(-sym,sym); ax.set_ylim(-sym,0.05)
ax.set_xlabel(r"Predicted $\Delta G_{H*}$ (eV)")
ax.set_ylabel(r"$-|\Delta G_{H*}|$ (Sabatier activity proxy, eV)")
ax.set_title("Predicted HER volcano over 1,836 surfaces")
ax.legend(frameon=False, loc="lower center", fontsize=9)
ax.text(0,0.02,r"optimal window $|\Delta G|\leq0.1$", ha="center", color="#5a7150", fontsize=9)
fig.savefig(os.path.join(OUT,"fig3_volcano.png")); plt.close(fig)

# ---------- Fig 4: external validation (two panels) ----------
ext2 = load_csv(os.path.join(ROOT,"data_processed","extval2_homogeneous.csv"))
ext1 = load_csv(os.path.join(ROOT,"data_processed","extval_results.csv"))
fig, axes = plt.subplots(1,2, figsize=(10.5,4.8))
for ax, rows, tcol, pcol, title in [
    (axes[0], ext2, "E_true","E_pred","CatHub homogeneous (355 surfaces)"),
    (axes[1], ext1, "E_true","E_pred","EqV2-HER cross-database (491 surfaces)")]:
    t = np.array([float(r[tcol]) for r in rows]); p = np.array([float(r[pcol]) for r in rows])
    m = np.mean(np.abs(t-p)); bias = np.mean(p-t)
    ax.scatter(t,p,s=12,alpha=0.5,color=SLATE,edgecolors="none")
    lo,hi = min(t.min(),p.min())-0.05, max(t.max(),p.max())+0.05
    ax.plot([lo,hi],[lo,hi],color=TERRA,ls="--",lw=1.3)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel("DFT $E_{ads}$ (eV)"); ax.set_ylabel("Predicted $E_{ads}$ (eV)")
    ax.set_title(title)
    ax.text(0.04,0.95,f"MAE = {m:.3f} eV\nbias = {bias:+.3f} eV", transform=ax.transAxes,
            va="top", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35",fc="#fffdf8",ec="#e7dccb"))
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig4_external.png")); plt.close(fig)

# ---------- Fig 5: descriptor saturation ----------
fams = [
    ("Mendeleev", load_csv(os.path.join(ROOT,"outputs","dimension_expansion_screen.csv")), GOLD),
    ("Materials Project elastic", load_csv(os.path.join(ROOT,"outputs","mp_elastic_dimensions_screen.csv")), GREEN),
    ("OMat24 reference", load_csv(os.path.join(ROOT,"outputs","omat_reference_dimension_screen.csv")), SLATE),
]
labels=[]; deltas=[]; cols=[]
for fname, rows, c in fams:
    for r in rows:
        if r["config"]=="baseline": continue
        labels.append(r["config"].replace("+","")); deltas.append(float(r["delta_vs_baseline"])); cols.append(c)
order=np.argsort(deltas)
labels=[labels[i] for i in order]; deltas=[deltas[i] for i in order]; cols=[cols[i] for i in order]
fig, ax = plt.subplots(figsize=(7.6,5.4))
ax.axhspan(-0.005,0.005, color=GREEN, alpha=0.12)
ax.barh(range(len(labels)), deltas, color=cols, alpha=0.9)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
ax.axvline(0,color=INK,lw=1)
ax.axvline(0.005,color=TERRA,ls="--",lw=1.2); ax.axvline(-0.005,color=TERRA,ls="--",lw=1.2)
ax.set_xlabel(r"$\Delta$MAE vs baseline (eV); negative = improvement")
ax.set_title("Static/bulk descriptor ablation: all changes within ±0.0016 eV")
ax.legend(handles=[Patch(facecolor=GOLD,label="Mendeleev"),
                   Patch(facecolor=GREEN,label="MP elastic"),
                   Patch(facecolor=SLATE,label="OMat24"),
                   Patch(facecolor=GREEN,alpha=0.2,label="±0.005 adoption band")],
          frameon=False, fontsize=8.5, loc="lower right")
fig.savefig(os.path.join(OUT,"fig5_saturation.png")); plt.close(fig)

# ---------- Fig 6: CHGNet artifact ----------
bc = load_csv(os.path.join(ROOT,"innovation","batch_c_preadjudication","batch_c_direct_comparison.csv"))
chg = np.array([float(r["frozen_chg_surf_disp"]) for r in bc])
dft = np.array([float(r["dft_disp_top_mean"]) for r in bc])
ratio = np.median(chg/dft)
fig, axes = plt.subplots(1,2, figsize=(10.2,4.6))
ax=axes[0]
ax.scatter(dft,chg,s=30,alpha=0.8,color=TERRA,edgecolors="white",linewidths=0.5)
lim=[0.02, max(chg.max(),dft.max())*1.2]
ax.plot(lim,lim,color=SLATE,ls="--",label="y = x")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("DFT top-layer displacement (Å)")
ax.set_ylabel("CHGNet top-layer displacement (Å)")
ax.set_title("Prototype → relaxed: CHGNet vs DFT")
ax.legend(frameon=False, fontsize=9)
ax=axes[1]
med=[np.median(dft),np.median(chg)]
b=ax.bar(["DFT","CHGNet"], med, color=[GREEN,TERRA_D], width=0.55)
ax.set_yscale("log"); ax.set_ylabel("median top-layer displacement (Å)")
ax.set_title(f"Median displacement ratio ≈ {ratio:.0f}x")
for rect,v in zip(b,med): ax.text(rect.get_x()+rect.get_width()/2, v*1.15, f"{v:.3f}", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig6_chgnet_artifact.png")); plt.close(fig)

print("Generated:", sorted(os.listdir(OUT)))
