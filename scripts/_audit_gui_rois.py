"""TEMPORARY audit (branch only, deleted before the PR is final): GUI decode vs script decode.

Runs inside the nightly lane, where the real checkpoint and the 13 ROI arrays already exist.
Prints one line per ROI and path, then one line per synthetic phenotype and distortion preset,
and writes the raw numbers to the path in argv[1].
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import app as gui  # noqa: E402
from evaluate_experimental import (  # noqa: E402
    coupling_errors,
    load_ground_truth,
    match_and_score,
)
from moldetr.inference import build_model, load_checkpoint, run  # noqa: E402
from moldetr.postprocess import decode_predictions, load_extrema  # noqa: E402
from moldetr.reproducibility import set_seed  # noqa: E402

THRESH = 0.3
OUT = Path(sys.argv[1])
ROI_DIR = Path(os.environ["MOLDETR_ROI_NPZ_DIR"])
CKPT = os.environ.get("MOLDETR_CHECKPOINT", str(gui.CHECKPOINT))
extrema = load_extrema(gui.EXTREMA)
model = load_checkpoint(build_model(), CKPT)


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def gui_preds(path: str, pph: float, noise_seed: int = 0) -> list[dict]:
    """The app.predict decode, but returning raw predictions so match_and_score can use them.

    ``noise_seed`` is run()'s own argument (default 0, which is what the GUI always uses): the
    in-model noise floor is seeded there, NOT by the global set_seed.
    """
    raw, cal = gui._load(path, trusted=True)
    amps = gui.validate_spectrum(raw, points_per_hz=pph)
    return decode_predictions(
        run(model, amps, noise_seed=noise_seed),
        extrema,
        pph,
        ppm_left=cal.get("ppm_left"),
        ppm_right=cal.get("ppm_right"),
        threshold=THRESH,
    )


def script_preds(npz, pph: float) -> list[dict]:
    spec = np.real(npz["spectrum_padded"])
    return decode_predictions(run(model, spec), extrema, pph, threshold=THRESH)


def seeded(fn, *args):
    set_seed(42)
    return fn(*args)


def summarize(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    return {
        "n_spin_systems": n,
        "median_dshift_hz": med([x for r in rows for x in r["dshift"]]),
        "median_dj_hz": med([x for r in rows for x in r["dj"]]),
        "proton_acc_pct": 100 * sum(r["correct"] for r in rows) / n,
        "matches_over_10hz": sum(1 for r in rows for x in r["dshift"] if x > 10),
        "n_pred_total": sum(r["n_pred"] for r in rows),
    }


def experimental() -> dict:
    out: dict = {"per_roi": {}, "summary": {}}
    acc: dict[str, list[dict]] = {"gui_meta_pph": [], "gui_default_pph": [], "script": []}
    for p in sorted(ROI_DIR.glob("roi_S*.npz")):
        npz = np.load(p, allow_pickle=True)  # the project's own Zenodo ROI arrays, as the nightly
        meta_pph = float(npz["metadata"].item().get("points_per_hz", 5.12))
        gts = load_ground_truth(npz)
        runs = (
            ("gui_meta_pph", seeded(gui_preds, str(p), meta_pph), meta_pph),
            ("gui_default_pph", seeded(gui_preds, str(p), 5.12), 5.12),
            ("script", seeded(script_preds, npz, meta_pph), meta_pph),
        )
        entry: dict = {"meta_pph": meta_pph, "n_gt": len(gts)}
        for label, preds, pph in runs:
            sc = match_and_score(preds, gts, pph)
            sc["n_pred"] = len(preds)
            sc["pred_pts"] = [round(q["chemical_shift_in_points"], 3) for q in preds]
            entry[label] = sc
            acc[label].append(sc)
            print(
                f"ROI {p.stem:9s} {label:16s} pph={pph:.3f} gt={sc['n']:2d} pred={len(preds):2d} "
                f"H_ok={sc['correct']:2d} med_dd={med(sc['dshift']):.3f}Hz maxdd="
                f"{max(sc['dshift'], default=float('nan')):.2f}Hz",
                flush=True,
            )
        # Does the served app.predict table agree with the replicated decode? (row count + table)
        table, _fig, msg = gui.predict(str(p), THRESH, gui.AUTO, None, None, meta_pph)
        entry["app_predict_rows"] = 0 if table is None else len(table)
        entry["app_predict_msg"] = msg
        out["per_roi"][p.stem] = entry
    for label, rows in acc.items():
        out["summary"][label] = summarize(rows)
        print("SUMMARY", label, json.dumps(out["summary"][label]), flush=True)
    return out


PRESETS = {
    "clean": dict(add_noise=False, snr=3.0, phase0=0.0, broaden=0.0, baseline=0.0),
    "snr3": dict(add_noise=True, snr=3.0, phase0=0.0, broaden=0.0, baseline=0.0),
    "snr3_phase4": dict(add_noise=True, snr=3.0, phase0=4.0, broaden=0.0, baseline=0.0),
}


def parse_cell(v) -> float | None:
    s = str(v).replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def synthetic() -> dict:
    out: dict = {}
    for name in gui.PHENOTYPE_CHOICES:
        m_rows, w_rows = gui._phenotype_grid(name)
        for preset, kw in PRESETS.items():
            set_seed(42)
            table, _fig, msg = gui.simulate_and_detect(
                m_rows, w_rows, threshold=THRESH, satellites=False, sat_j=0.0, **kw
            )
            if table is None or len(table) == 0:
                out[f"{name}|{preset}"] = {"error": str(msg)}
                print(f"SYN {name:22s} {preset:12s} ERROR {msg}", flush=True)
                continue
            dd = [parse_cell(x) for x in table["Δδ (Hz)"]]
            dh = [parse_cell(x) for x in table["ΔH"]]
            dj = [parse_cell(x) for x in table["ΔJ (Hz)"]]
            status = [str(s) for s in table["status"]]
            rec = {
                "n_rows": len(table),
                "med_dd_hz": med([x for x in dd if x is not None]),
                "med_dj_hz": med([x for x in dj if x is not None]),
                "h_exact": sum(1 for x in dh if x == 0),
                "h_total": sum(1 for x in dh if x is not None),
                "missed": sum(1 for s in status if "missed" in s),
                "extra": sum(1 for s in status if "extra" in s),
            }
            out[f"{name}|{preset}"] = rec
            print(f"SYN {name:22s} {preset:12s} {json.dumps(rec)}", flush=True)
    return out


def replicates(n_seeds: int = 10) -> dict:
    """GUI decode with run()'s noise_seed varied. The paper pooled ~5 stochastic repeats per ROI
    (215 matched pairs over 41 distinct labels in the committed JSON), the GUI uses one (seed 0)."""
    per_seed: dict[int, list[dict]] = {}
    for seed in range(n_seeds):
        rows = []
        for p in sorted(ROI_DIR.glob("roi_S*.npz")):
            npz = np.load(p, allow_pickle=True)  # project's own Zenodo ROI arrays, as above
            gts = load_ground_truth(npz)
            sc = match_and_score(gui_preds(str(p), 5.12, seed), gts, 5.12)
            sc["n_pred"] = len(gui_preds(str(p), 5.12, seed))
            rows.append(sc)
        per_seed[seed] = rows
        s = summarize(rows)
        print(f"SEED {seed} {json.dumps(s)}", flush=True)
    pooled = {}
    for label, seeds in (("pooled_0-4", range(5)), ("pooled_0-9", range(n_seeds))):
        pooled[label] = summarize([r for s in seeds for r in per_seed[s]])
        print("POOLED", label, json.dumps(pooled[label]), flush=True)
    return {"per_seed": {s: summarize(r) for s, r in per_seed.items()}, "pooled": pooled}


PAPER_THRESH = float(1 / (1 + np.exp(0.5)))  # paper: any(logit > -0.5) == max sigmoid > 0.378


def paper_input(npz, noise_seed: int) -> np.ndarray:
    """multiplet_detection_detr/evaluate_experimental.py:1403-1466 on the Zenodo raw ROI:
    sigma-2 Gaussian broadening and 0.5 % noise on the ROI chunk only, divide-by-max, zero pad."""
    from moldetr.dataloader.data_augmentation import add_line_broadening

    md = npz["metadata"].item()
    pb, pa = int(md.get("padding_before", 0)), int(md.get("padding_after", 0))
    full = np.asarray(npz["spectrum_padded"])
    chunk = full[pb : full.size - pa]
    chunk = np.real(add_line_broadening(chunk, custom_sigma=2.0, use_custom_values=True))
    rng = np.random.RandomState(noise_seed)
    chunk = chunk + rng.normal(0, 0.005 * chunk.max(), chunk.shape)
    chunk = chunk / (chunk.max() + 1e-6)
    return np.pad(chunk, (pb, pa), "constant", constant_values=0).astype(np.float32)


def forward_raw(arr: np.ndarray):
    import torch

    with torch.no_grad():
        out = model(torch.from_numpy(arr).float()[None, None, :])
    return out[0].reshape(-1, out.shape[-1])


def live_input(path: str, broaden: bool):
    raw, _cal = gui._load(path, trusted=True)
    if broaden:
        from moldetr.dataloader.data_augmentation import add_line_broadening

        raw = add_line_broadening(np.asarray(raw), custom_sigma=2.0, use_custom_values=True)
    return gui.validate_spectrum(raw, points_per_hz=5.12)


GROUP0 = 10  # out[0] is (n_groups=8, num_queries=10, C) reshaped row-major: rows 0-9 = group 0


def variant_preds(variant: str, path: str, npz, seed: int) -> list[dict]:
    """The paper read group 0 only (evaluate_experimental.py:247, ``[0, 0, ...]``)."""
    paper_in = variant in ("V5_paper_input_public_decode", "V7_paper_input_group0_paperdecode")
    if paper_in:
        out = forward_raw(paper_input(npz, seed))
    else:
        out = run(model, live_input(path, broaden=False), noise_seed=seed)
    if variant.startswith(("V6", "V7", "V8")):
        out = out[:GROUP0]
    paper_decode = variant in ("V6_live_group0_paperdecode", "V7_paper_input_group0_paperdecode")
    thresh = PAPER_THRESH if paper_decode else THRESH
    merge = 0.0 if paper_decode else 20.0
    return decode_predictions(out, extrema, 5.12, threshold=thresh, merge_tol_points=merge)


VARIANTS = (
    "V0_live",
    "V6_live_group0_paperdecode",
    "V7_paper_input_group0_paperdecode",
)


def paper_match(preds: list[dict], gts: list[dict], pph: float = 5.12) -> dict:
    """Transcribed from multiplet_detection_detr/matching_4_experimental_evaluation.py:100-190:
    Hungarian on cost = (proton mismatch ? 1 : 0) + |dshift Hz| / 80, pairs > 20 Hz excluded."""
    from scipy.optimize import linear_sum_assignment

    out = {"dshift": [], "dj": [], "correct": 0, "n": len(gts), "n_matched": 0}
    if not preds or not gts:
        return out
    cost = np.full((len(preds), len(gts)), np.inf)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            d = abs(p["chemical_shift_in_points"] - g["chemical_shift_in_points"]) / pph
            if d <= 20.0:
                cost[i, j] = float(p["proton_count"] != g["proton_count"]) + d / 80.0
    if np.isinf(cost).all():
        return out
    finite = np.where(np.isinf(cost), 1e9, cost)  # scipy rejects infeasible inf matrices
    for i, j in zip(*linear_sum_assignment(finite)):
        if np.isinf(cost[i, j]):
            continue
        p, g = preds[i], gts[j]
        out["n_matched"] += 1
        out["dshift"].append(abs(p["chemical_shift_in_points"] - g["chemical_shift_in_points"]) / pph)
        out["correct"] += int(p["proton_count"] == g["proton_count"])
        out["dj"].extend(coupling_errors(p["coupling_constants_hz"], g.get("coupling_constants", [])))
    return out


def ablation_paper_matcher(seeds=range(5)) -> dict:
    res: dict = {}
    files = sorted(ROI_DIR.glob("roi_S*.npz"))
    for variant in VARIANTS:
        rows, per_roi = [], {}
        for p in files:
            npz = np.load(p, allow_pickle=True)  # project's own Zenodo ROI arrays, as above
            gts = load_ground_truth(npz)
            ok = n = nm = 0
            for seed in seeds:
                sc = paper_match(variant_preds(variant, str(p), npz, seed), gts)
                sc["n_pred"] = 0
                rows.append(sc)
                ok, n, nm = ok + sc["correct"], n + sc["n"], nm + sc["n_matched"]
            per_roi[p.stem] = {"H_ok": ok, "n": n, "n_matched": nm}
        pooled = summarize(rows)
        nmatch = sum(r["n_matched"] for r in rows)
        pooled["n_matched"] = nmatch
        pooled["proton_acc_matched_pct"] = 100 * sum(r["correct"] for r in rows) / nmatch
        res[variant] = {"pooled": pooled, "per_roi": per_roi}
        print("PAPERMATCH", variant, json.dumps(pooled), flush=True)
        print("PMPERROI", variant, json.dumps(per_roi), flush=True)
    return res


def ablation(seeds=range(5)) -> dict:
    res: dict = {}
    files = sorted(ROI_DIR.glob("roi_S*.npz"))
    for variant in VARIANTS:
        rows, per_roi = [], {}
        for p in files:
            npz = np.load(p, allow_pickle=True)  # project's own Zenodo ROI arrays, as above
            gts = load_ground_truth(npz)
            ok = n = npred = 0
            for seed in seeds:
                preds = variant_preds(variant, str(p), npz, seed)
                sc = match_and_score(preds, gts, 5.12)
                sc["n_pred"] = len(preds)
                rows.append(sc)
                ok, n, npred = ok + sc["correct"], n + sc["n"], npred + len(preds)
            per_roi[p.stem] = {"H_ok": ok, "n": n, "n_pred": npred}
        res[variant] = {"pooled": summarize(rows), "per_roi": per_roi}
        print("ABLATION", variant, json.dumps(res[variant]["pooled"]), flush=True)
        print("PERROI", variant, json.dumps(per_roi), flush=True)
    return res


if __name__ == "__main__":
    result = {"paper_matcher": ablation_paper_matcher()}
    OUT.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT)
