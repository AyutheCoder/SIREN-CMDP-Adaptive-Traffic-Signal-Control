"""Train the acoustic siren detector and fit its empirical error model.

Two artefacts come out of this, and the second matters as much as the first:

1. ``models/siren_detector.h5``   - the CNN itself.
2. ``models/sensor_error_model.json`` - a statistical description of how the CNN
   behaves on held-out data, conditioned on SNR. The RL loop samples detections
   from this instead of rendering audio, which is ~30x faster and lets the agent
   train against the detector's *real* failure modes rather than an invented
   noise model. ``scripts/validate_surrogate.py`` checks the two agree.

Usage
-----
    python scripts/train_detector.py --epochs 25 --train-clips 12000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.config import AcousticConfig, DetectorConfig, MODELS_DIR, RESULTS_DIR, ensure_dirs
from atsc.sensing.detector import SirenDetector, evaluate_detector
from atsc.sensing.sensor import SensorErrorModel
from atsc.sensing.siren_synth import SirenSynthesizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--train-clips", type=int, default=12000)
    p.add_argument("--val-clips", type=int, default=2000)
    p.add_argument("--test-clips", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-siren", type=float, default=0.5)
    p.add_argument("--out", type=str, default=str(MODELS_DIR / "siren_detector.h5"))
    args = p.parse_args()

    ensure_dirs()
    np.random.seed(args.seed)

    acoustic = AcousticConfig()
    cfg = DetectorConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_clips=args.train_clips,
        val_clips=args.val_clips,
        test_clips=args.test_clips,
        seed=args.seed,
    )

    detector = SirenDetector(acoustic, cfg)
    print(f"detector parameters: {detector.model.count_params():,}")

    # Distinct seeds so train / val / test scenes cannot coincide.
    t0 = time.time()
    print("synthesising training data ...", flush=True)
    train = SirenSynthesizer(acoustic, seed=args.seed + 100).build_dataset(
        cfg.train_clips, detector.mel, p_siren=args.p_siren, progress_every=2000
    )
    val = SirenSynthesizer(acoustic, seed=args.seed + 200).build_dataset(
        cfg.val_clips, detector.mel, p_siren=args.p_siren
    )
    test = SirenSynthesizer(acoustic, seed=args.seed + 300).build_dataset(
        cfg.test_clips, detector.mel, p_siren=args.p_siren
    )
    print(f"  synthesis took {time.time() - t0:.1f}s "
          f"({cfg.train_clips + cfg.val_clips + cfg.test_clips} clips)", flush=True)

    t0 = time.time()
    detector.fit(train, val, verbose=2)
    print(f"training took {time.time() - t0:.1f}s", flush=True)

    metrics = evaluate_detector(detector, test, acoustic)
    print("\n=== held-out detector performance ===")
    for k in ("detection_precision", "detection_recall", "detection_f1",
              "false_positive_rate", "class_accuracy_given_siren",
              "vr_mae_ms", "range_mape", "eta_mae_s", "eta_median_abs_err_s"):
        print(f"  {k:32s} {metrics[k]:.4f}")
    print("\n  recall by SNR:")
    for row in metrics["by_snr"]:
        print(f"    {row['snr_bin']:>14s}  n={row['n']:5d}  recall={row['recall']:.3f}  "
              f"class_acc={row['class_accuracy']:.3f}  vr_mae={row['vr_mae_ms']:.2f} m/s")
    print("\n  range error by true range:")
    for row in metrics["by_range"]:
        print(f"    {row['range_bin_m']:>14s}  n={row['n']:5d}  "
              f"mae={row['range_mae_m']:7.2f} m  mape={row['range_mape']:.3f}")
    if "range_mape_on_axis" in metrics:
        print(f"\n  range MAPE on-axis (<25 deg): {metrics['range_mape_on_axis']:.3f}")
        print(f"  range MAPE off-axis:          {metrics.get('range_mape_off_axis', float('nan')):.3f}")

    out = Path(args.out)
    detector.save(out)
    print(f"\nsaved detector -> {out}")

    pred = detector.predict_batch(test["x"])
    error_model = SensorErrorModel.fit(
        y_class=test["y_class"],
        snr_db=np.where(np.isfinite(test["snr_db"]), test["snr_db"], -99.0),
        pred_class_probs=pred["cls"],
        pred_vr=pred["vr"],
        pred_logr=pred["logr"],
        true_vr=test["y_vr"],
        true_logr=test["y_logr"],
        cfg=acoustic,
        detector_prior=args.p_siren,
    )
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model.save(em_path)
    print(f"saved sensor error model -> {em_path}")

    metrics_path = RESULTS_DIR / "detector_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
