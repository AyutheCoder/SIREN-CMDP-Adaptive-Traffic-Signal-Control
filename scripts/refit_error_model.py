"""Refit the surrogate's error model from the *saved* detector, without retraining.

``train_detector.py`` fits the error model at the end of a training run, which
couples two things that need not be coupled: the CNN's weights, and the
statistical summary of its behaviour that the fast surrogate samples from. When
only the summary needs to change -- a new error-model field, a different SNR
binning, or, as here, a confuser class added after a training run was already
under way -- retraining the network to obtain it wastes half an hour and, worse,
changes the weights the summary is supposed to describe.

This script loads the detector as it stands, synthesises a fresh held-out set,
runs inference, and refits. The detector is not touched.

    python scripts/refit_error_model.py
    python scripts/refit_error_model.py --clips 8000 --p-siren 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.config import AcousticConfig, MODELS_DIR, ensure_dirs
from atsc.sensing.detector import SirenDetector
from atsc.sensing.sensor import SensorErrorModel
from atsc.sensing.siren_synth import SirenSynthesizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", type=int, default=6000)
    p.add_argument("--p-siren", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=777,
                   help="disjoint from every seed train_detector.py uses, so "
                        "the refit is still held out")
    p.add_argument("--detector", default=str(MODELS_DIR / "siren_detector.h5"))
    p.add_argument("--out", default=str(MODELS_DIR / "sensor_error_model.json"))
    args = p.parse_args()

    ensure_dirs()
    acoustic = AcousticConfig()
    detector = SirenDetector(acoustic)
    detector.load(Path(args.detector))
    print(f"loaded detector <- {args.detector}")

    synth = SirenSynthesizer(acoustic, seed=args.seed)
    test = synth.build_dataset(args.clips, detector.mel, p_siren=args.p_siren,
                               progress_every=2000)
    counts = np.bincount(test["y_class"], minlength=len(acoustic.class_map))
    names = sorted(acoustic.class_map, key=lambda k: acoustic.class_map[k])
    print("held-out class counts: "
          + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    if acoustic.class_map.get("horn") is not None and counts[acoustic.class_map["horn"]] == 0:
        print("WARNING: no horn clips in the held-out set; the horn curve will "
              "be empty and horns will fall back to the siren curve.")

    pred = detector.predict_batch(test["x"])
    model = SensorErrorModel.fit(
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
    model.save(Path(args.out))
    print(f"saved sensor error model -> {args.out}")

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    edges = model.snr_edges
    labels = ([f"< {edges[0]:g}"]
              + [f"{edges[i]:g}..{edges[i + 1]:g}" for i in range(len(edges) - 1)]
              + [f"> {edges[-1]:g}"])
    print("\nprobability the detector calls it a siren:")
    print(f"{'SNR bin (dB)':>14} {'true siren':>12} {'road horn':>12}")
    print("-" * 42)
    for i, sm in enumerate(model.siren_logit_mean):
        horn = (f"{sig(model.horn_logit_mean[i]):12.3f}"
                if i < len(model.horn_logit_mean) else f"{'--':>12}")
        print(f"{labels[i]:>14} {sig(sm):12.3f} {horn}")
    print("-" * 42)
    print(f"{'broadband noise':>14} {sig(model.noise_logit_mean):12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
