"""Siren detector: a small CNN with three heads on a half-second log-mel frame.

    log-mel (40 x 47 dB)  ->  conv stack  ->  { class      (4-way softmax)
                                              { v_radial   (regression, m/s)
                                              { log range  (regression, ln m)

The two regression heads are what make the detector *actionable*. A binary
"siren / no siren" flag tells the controller nothing about whether to start a
changeover now or in twenty seconds; range and range-rate together give an
arrival-time estimate, and it is that estimate the belief filter tracks and the
policy conditions on.

Both regression losses are masked on noise-only frames, where the targets are
undefined. Range is regressed in log space because the useful resolution is
multiplicative: confusing 20 m with 40 m matters, confusing 220 m with 240 m
does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import AcousticConfig, DetectorConfig
from .dsp import MelSpectrogram

# Band levels land roughly in [-20, 100] dB SPL, with urban ambient around
# 50-60 dB. Centring keeps the first layer well conditioned; note this is an
# affine shift, not a per-frame normalisation, so absolute level survives - it
# is the range cue.
_DB_OFFSET = 55.0
_DB_SCALE = 20.0


class SirenDetector:
    def __init__(
        self,
        acoustic: Optional[AcousticConfig] = None,
        cfg: Optional[DetectorConfig] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
    ):
        self.acoustic = acoustic or AcousticConfig()
        self.cfg = cfg or DetectorConfig()
        self.mel = MelSpectrogram(
            self.acoustic.sample_rate,
            self.acoustic.n_fft,
            self.acoustic.hop_length,
            self.acoustic.n_mels,
            self.acoustic.fmin,
            self.acoustic.fmax,
        )
        if input_shape is None:
            n_samples = int(self.acoustic.sample_rate * self.acoustic.frame_s)
            input_shape = (self.acoustic.n_mels, self.mel.n_frames(n_samples), 1)
        self.input_shape = input_shape
        self.n_classes = len(self.acoustic.class_map)
        self.model = self._build()

    # ------------------------------------------------------------------ model
    def _build(self) -> keras.Model:
        cfg = self.cfg
        inp = keras.Input(shape=self.input_shape, name="logmel")
        x = layers.Rescaling(1.0 / _DB_SCALE, offset=-_DB_OFFSET / _DB_SCALE)(inp)

        for i, ch in enumerate(cfg.conv_channels):
            x = layers.Conv2D(ch, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
            x = layers.BatchNormalization(name=f"bn{i}")(x)
            x = layers.Activation("relu", name=f"relu{i}")(x)
            # Pool frequency harder than time: the temporal envelope carries the
            # range-rate cue and must survive to the pooling stage.
            x = layers.MaxPooling2D((2, 2) if i < 2 else (2, 1), name=f"pool{i}")(x)

        # Collapse frequency, keep time.
        t_seq = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="freq_pool")(x)

        # Average preserves overall level (a range cue); max preserves peak tonal
        # energy (the siren cue); the first temporal moment measures how fast the
        # level is *changing* across the frame, which for a source closing at
        # 20 m/s from 100 m is a ~1.9 dB ramp over one second. That ramp is the
        # only range-rate information physically present in a single frame -
        # Doppler is sub-bin at mel resolution - so it has to be pooled for
        # explicitly. Global pooling over time alone discards it entirely.
        gap = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1), name="gap")(t_seq)
        gmp = layers.Lambda(lambda z: tf.reduce_max(z, axis=1), name="gmp")(t_seq)

        def _temporal_moment(z):
            n = tf.shape(z)[1]
            w = tf.linspace(-1.0, 1.0, n)
            w = w / (tf.sqrt(tf.reduce_sum(w * w)) + 1e-6)
            return tf.reduce_sum(z * w[None, :, None], axis=1)

        slope = layers.Lambda(_temporal_moment, name="temporal_slope")(t_seq)
        h = layers.Concatenate(name="pooled")([gap, gmp, slope])
        h = layers.Dense(cfg.dense_units, activation="relu", name="fc")(h)
        h = layers.Dropout(cfg.dropout, name="drop")(h)

        cls = layers.Dense(self.n_classes, activation="softmax", name="cls")(h)
        vr = layers.Dense(1, activation="tanh", name="vr")(h)
        logr = layers.Dense(1, activation=None, name="logr")(h)

        model = keras.Model(inp, {"cls": cls, "vr": vr, "logr": logr}, name="siren_detector")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss={
                "cls": keras.losses.SparseCategoricalCrossentropy(),
                "vr": keras.losses.Huber(delta=0.2),
                "logr": keras.losses.Huber(delta=0.5),
            },
            loss_weights={"cls": 1.0, "vr": cfg.velocity_loss_weight, "logr": 0.5},
            metrics={"cls": ["accuracy"]},
            weighted_metrics=[],
        )
        return model

    # ------------------------------------------------------------------ train
    def fit(
        self,
        train: Dict[str, np.ndarray],
        val: Optional[Dict[str, np.ndarray]] = None,
        verbose: int = 2,
    ) -> keras.callbacks.History:
        cfg = self.cfg
        none_idx = self.acoustic.class_map["none"]

        def _pack(d: Dict[str, np.ndarray]):
            mask = (d["y_class"] != none_idx).astype(np.float32)
            y = {
                "cls": d["y_class"],
                "vr": d["y_vr"].reshape(-1, 1),
                "logr": d["y_logr"].reshape(-1, 1),
            }
            w = {"cls": np.ones_like(mask), "vr": mask, "logr": mask}
            return d["x"], y, w

        x, y, w = _pack(train)
        validation_data = None
        if val is not None:
            vx, vy, vw = _pack(val)
            validation_data = (vx, vy, vw)

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val is not None else "loss",
                factor=0.5, patience=3, min_lr=1e-5, verbose=0,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if val is not None else "loss",
                patience=7, restore_best_weights=True, verbose=0,
            ),
        ]
        return self.model.fit(
            x, y,
            sample_weight=w,
            validation_data=validation_data,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    # --------------------------------------------------------------- inference
    def predict_batch(self, x: np.ndarray, batch_size: int = 512) -> Dict[str, np.ndarray]:
        out = self.model.predict(x, batch_size=batch_size, verbose=0)
        return {
            "cls": np.asarray(out["cls"]),
            "vr": np.asarray(out["vr"]).reshape(-1),
            "logr": np.asarray(out["logr"]).reshape(-1),
        }

    def predict_frame(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Single frame, shape ``(1, n_mels, n_frames, 1)``.

        Uses the model's ``__call__`` rather than ``predict`` to avoid the
        per-call tf.data and progress-bar overhead, which dominates at batch 1.
        """
        out = self.model(x, training=False)
        cls = np.asarray(out["cls"])[0]
        vr = float(np.asarray(out["vr"]).reshape(-1)[0])
        logr = float(np.asarray(out["logr"]).reshape(-1)[0])
        return cls, vr, logr

    # ------------------------------------------------------------------- i/o
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(path))
        meta = {
            "input_shape": list(self.input_shape),
            "class_map": self.acoustic.class_map,
            "max_radial_speed_ms": self.acoustic.max_radial_speed_ms,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: Path) -> "SirenDetector":
        self.model.load_weights(str(path))
        return self

    @classmethod
    def from_saved(
        cls,
        path: Path,
        acoustic: Optional[AcousticConfig] = None,
        cfg: Optional[DetectorConfig] = None,
    ) -> "SirenDetector":
        det = cls(acoustic=acoustic, cfg=cfg)
        det.load(path)
        return det


def evaluate_detector(
    detector: SirenDetector, test: Dict[str, np.ndarray], acoustic: AcousticConfig
) -> Dict[str, object]:
    """Held-out metrics, reported overall and stratified by SNR."""
    pred = detector.predict_batch(test["x"])
    y = test["y_class"]
    none_idx = acoustic.class_map["none"]
    yhat = pred["cls"].argmax(axis=1)

    # "Siren" means one of the emergency classes, not "anything but silence".
    # With the horn class present those differ, and using the latter would score
    # a correctly rejected horn as a *missed siren* -- inverting the sign of the
    # thing this evaluation exists to measure.
    from .belief import SIREN_CLASSES as _SC
    siren_idx = {acoustic.class_map[c] for c in _SC if c in acoustic.class_map}
    is_siren = np.isin(y, list(siren_idx))
    pred_siren = np.isin(yhat, list(siren_idx))

    tp = int(np.sum(is_siren & pred_siren))
    fp = int(np.sum(~is_siren & pred_siren))
    fn = int(np.sum(is_siren & ~pred_siren))
    tn = int(np.sum(~is_siren & ~pred_siren))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    correct_class = float(np.mean(yhat[is_siren] == y[is_siren])) if is_siren.any() else 0.0

    vr_err = (pred["vr"][is_siren] - test["y_vr"][is_siren]) * acoustic.max_radial_speed_ms
    r_pred = np.exp(pred["logr"][is_siren])
    r_true = np.exp(test["y_logr"][is_siren])
    range_ape = np.abs(r_pred - r_true) / np.maximum(r_true, 1.0)

    # Arrival-time error, restricted to genuinely approaching sources - the
    # only case where an ETA is defined and the only case the controller uses.
    vr_true = test["y_vr"][is_siren] * acoustic.max_radial_speed_ms
    vr_hat = pred["vr"][is_siren] * acoustic.max_radial_speed_ms
    approaching = vr_true < -2.0
    eta_true = r_true[approaching] / np.abs(vr_true[approaching])
    eta_hat = r_pred[approaching] / np.maximum(np.abs(vr_hat[approaching]), 0.5)
    eta_err = eta_hat - eta_true

    result: Dict[str, object] = {
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "false_positive_rate": fp / max(fp + tn, 1),
        "class_accuracy_given_siren": correct_class,
        "overall_accuracy": float(np.mean(yhat == y)),
        "vr_mae_ms": float(np.mean(np.abs(vr_err))),
        "vr_rmse_ms": float(np.sqrt(np.mean(vr_err ** 2))),
        "range_mape": float(np.mean(range_ape)),
        "range_mae_m": float(np.mean(np.abs(r_pred - r_true))),
        "eta_mae_s": float(np.mean(np.abs(eta_err))) if eta_err.size else float("nan"),
        "eta_median_abs_err_s": float(np.median(np.abs(eta_err))) if eta_err.size else float("nan"),
        "n_test": int(len(y)),
    }

    # Stratified recall: where does the detector actually break down?
    snr = test["snr_db"]
    edges = [-20.0, -10.0, -5.0, 0.0, 5.0, 15.0]
    by_snr = []
    bins = np.digitize(snr, edges)
    for b in range(len(edges) + 1):
        sel = is_siren & (bins == b)
        if sel.sum() == 0:
            continue
        lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
        hi = "inf" if b == len(edges) else f"{edges[b]:g}"
        sub_vr = (pred["vr"][sel] - test["y_vr"][sel]) * acoustic.max_radial_speed_ms
        by_snr.append({
            "snr_bin": f"[{lo}, {hi})",
            "n": int(sel.sum()),
            "recall": float(np.mean(np.isin(yhat[sel], list(siren_idx)))),
            "class_accuracy": float(np.mean(yhat[sel] == y[sel])),
            "vr_mae_ms": float(np.mean(np.abs(sub_vr))),
        })
    result["by_snr"] = by_snr

    # Range error by true range. Absorption-driven spectral tilt is a strong
    # range cue at short range and a weak one far out, where the upper harmonics
    # have already fallen below the noise floor; the breakdown makes that
    # limitation explicit rather than hiding it in a single averaged number.
    range_edges = [30.0, 60.0, 100.0, 160.0, 220.0]
    by_range = []
    rbins = np.digitize(r_true, range_edges)
    for b in range(len(range_edges) + 1):
        sel = rbins == b
        if sel.sum() == 0:
            continue
        lo = "0" if b == 0 else f"{range_edges[b - 1]:g}"
        hi = "inf" if b == len(range_edges) else f"{range_edges[b]:g}"
        by_range.append({
            "range_bin_m": f"[{lo}, {hi})",
            "n": int(sel.sum()),
            "range_mae_m": float(np.mean(np.abs(r_pred[sel] - r_true[sel]))),
            "range_mape": float(np.mean(np.abs(r_pred[sel] - r_true[sel]) / np.maximum(r_true[sel], 1.0))),
        })
    result["by_range"] = by_range

    if "off_axis_rad" in test:
        theta = test["off_axis_rad"][is_siren]
        on_axis = theta < np.deg2rad(25.0)
        if on_axis.any():
            result["range_mape_on_axis"] = float(
                np.mean(np.abs(r_pred[on_axis] - r_true[on_axis]) / np.maximum(r_true[on_axis], 1.0))
            )
        if (~on_axis).any():
            result["range_mape_off_axis"] = float(
                np.mean(np.abs(r_pred[~on_axis] - r_true[~on_axis]) / np.maximum(r_true[~on_axis], 1.0))
            )

    labels = sorted(acoustic.class_map, key=lambda k: acoustic.class_map[k])
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y, yhat):
        cm[t, p] += 1
    result["confusion_matrix"] = cm.tolist()
    result["confusion_labels"] = labels
    return result
