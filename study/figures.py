#!/usr/bin/env python3
"""Pictures of the cells simulated so far: vorticity snapshots, energy spectra and GIFs.

    python3 study/figures.py study/study.yaml               # -> runs/full/media/
    python3 study/figures.py study/study.yaml --frames 200 --force
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colormaps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402
from features import spectral_resample  # noqa: E402

SHOW = 256

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in"})


def available(c, traj):
    data = R.paths(c)["data"]
    out = {}
    for name, re, n in R.cells(c):
        base = os.path.join(data, f"{name}_traj{traj}")
        if os.path.exists(base + ".json"):
            out[name] = dict(re=re, n=n, npz=base + ".npz", meta=json.load(open(base + ".json")))
    return out


def shown(w):
    return spectral_resample(w.astype(np.float64), SHOW) if w.shape[0] > SHOW else w


def scale(frames):
    return float(np.percentile(np.abs(frames), 99.5)) or 1.0


def label(cell):
    return f"Re$_f$ = {cell['meta']['measured']['Re_f']:.0f}, {cell['n']}$^2$"


def snapshots(c, cells, out):
    grids, res = c["grids"], c["reynolds"]
    fig, axes = plt.subplots(len(grids), len(res), figsize=(1.6 * len(res), 1.65 * len(grids)), squeeze=False)
    names = {(n, re): name for name, re, n in R.cells(c)}
    for r, n in enumerate(grids):
        for k, re in enumerate(res):
            ax = axes[r, k]
            ax.set_xticks([]), ax.set_yticks([])
            name = names.get((n, float(re)))
            if name is None:
                ax.axis("off")
                continue
            if name not in cells:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", color="0.5", transform=ax.transAxes)
                for s in ax.spines.values():
                    s.set_color("0.8")
            else:
                w = shown(np.load(cells[name]["npz"])["omega"][-1])
                v = scale(w)
                ax.imshow(w, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower", interpolation="bilinear")
                ax.set_title(f"Re$_f$ = {cells[name]['meta']['measured']['Re_f']:.0f}", fontsize=7, pad=2)
            if k == 0 or names.get((n, float(res[k - 1]))) is None:
                ax.set_ylabel(f"{n}$^2$")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "snapshots.png"), dpi=200)
    plt.close(fig)


def spectrum(frames):
    n = frames.shape[-1]
    wh = np.fft.rfft2(frames) / n**2
    kx = np.fft.rfftfreq(n, 1.0 / n)
    ky = np.fft.fftfreq(n, 1.0 / n)
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    weight = np.full(K.shape, 2.0)
    weight[:, 0] = 1.0
    weight[:, -1] = 1.0
    e = 0.5 * weight * (np.abs(wh) ** 2).mean(0) / np.where(K > 0, K, np.inf) ** 2
    E = np.bincount(np.rint(K).astype(int).ravel(), e.ravel())
    return np.arange(len(E)), E


def spectra(c, cells, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    cmap = colormaps["viridis"]
    lre = np.log([min(c["reynolds"]), max(c["reynolds"])])
    styles = {g: s for g, s in zip(c["grids"], ["-", "--", "-.", ":"])}
    for name, cell in sorted(cells.items(), key=lambda kv: (kv[1]["re"], kv[1]["n"])):
        frames = np.load(cell["npz"])["omega"][-50:].astype(np.float64)
        k, E = spectrum(frames)
        kmax = cell["n"] // 3
        col = cmap(0.05 + 0.85 * (np.log(cell["re"]) - lre[0]) / max(lre[1] - lre[0], 1e-9))
        ax.loglog(k[1:kmax + 1], E[1:kmax + 1], styles.get(cell["n"], "-"), color=col, lw=0.9, label=label(cell))
    ax.axvline(4, color="0.6", lw=0.5)
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("energy spectrum $E(k)$")
    ax.legend(fontsize=5.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "spectra.png"), dpi=200)
    plt.close(fig)


def gif(panels, path, fps):
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.4), squeeze=False)
    ims = []
    for ax, (title, frames) in zip(axes[0], panels):
        v = scale(frames)
        ims.append(ax.imshow(frames[0], cmap="RdBu_r", vmin=-v, vmax=v, origin="lower", interpolation="bilinear"))
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    n = min(len(f) for _, f in panels)

    def draw(i):
        for im, (_, frames) in zip(ims, panels):
            im.set_data(frames[i])
        return ims

    animation.FuncAnimation(fig, draw, frames=n, blit=True).save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def frames_of(cell, count):
    return np.stack([shown(w) for w in np.load(cell["npz"])["omega"][:count]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frames", type=int, default=150, help="snapshots per GIF (0.4 time units apart)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--force", action="store_true", help="redo GIFs that already exist")
    a = ap.parse_args()
    c = R.load(a.config)
    out = os.path.join(c["out"], "media")
    os.makedirs(os.path.join(out, "gifs"), exist_ok=True)
    cells = available(c, a.traj)
    print(f"{len(cells)} of {len(R.cells(c))} cells have trajectory {a.traj}")
    if not cells:
        return
    snapshots(c, cells, out)
    spectra(c, cells, out)
    print(f"{out}/snapshots.png, spectra.png")
    for name, cell in cells.items():
        path = os.path.join(out, "gifs", f"{name}.gif")
        if a.force or not os.path.exists(path):
            gif([(label(cell), frames_of(cell, a.frames))], path, a.fps)
            print(f"{path}", flush=True)
    finest = {}
    for name, cell in cells.items():
        if cell["re"] not in finest or cell["n"] > cells[finest[cell["re"]]]["n"]:
            finest[cell["re"]] = name
    if len(finest) > 1:
        panels = [(label(cells[finest[re]]), frames_of(cells[finest[re]], a.frames)) for re in sorted(finest)]
        gif(panels, os.path.join(out, "reynolds.gif"), a.fps)
        print(f"{out}/reynolds.gif ({len(panels)} Reynolds numbers, finest grid of each)")


if __name__ == "__main__":
    main()
