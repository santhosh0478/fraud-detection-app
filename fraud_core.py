"""Shared scoring code. The Gradio demo now and the Streamlit app later both use this file."""
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb


def _build_autoencoder(n):
    import torch.nn as nn            # torch is only imported when the autoencoder is actually loaded

    class AutoEncoder(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
            self.dec = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, n))
        def forward(self, x):
            return self.dec(self.enc(x))

    return AutoEncoder(n)


class _NumpyAE:
    """The same 29-16-8-4-8-16-29 autoencoder as in training, as plain numpy (no PyTorch needed to serve it)."""

    def __init__(self, path):
        w = np.load(path)
        self.layers = [(w[f"{p}.weight"].astype(np.float32), w[f"{p}.bias"].astype(np.float32))
                       for p in ("enc.0", "enc.2", "enc.4", "dec.0", "dec.2", "dec.4")]

    def __call__(self, X):
        h = X
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i not in (2, 5):            # no ReLU after the bottleneck layer or the output layer
                h = np.maximum(h, 0)
        return h


class FraudModels:
    """Loads the artifacts ONCE; every prediction goes through the same preprocessing as training."""

    def __init__(self, art_dir=".", load_ae=None):
        if load_ae is None:                                   # switch used only for testing without PyTorch
            load_ae = os.environ.get("FRAUD_LOAD_AE", "1") == "1"
        cfg = json.load(open(os.path.join(art_dir, "config.json")))
        self.cols = cfg["cols"]                                   # feature ORDER used in training
        self.mean = np.array(cfg["mean"], dtype=np.float64)
        self.scale = np.array(cfg["scale"], dtype=np.float64)
        self.ae_thr = float(cfg["ae_thr"])
        self.amount_idx = self.cols.index("Amount")

        xcfg = json.load(open(os.path.join(art_dir, "xgb_config.json")))
        self.xgb_thr = float(xcfg["xgb_thr"])
        self.xgb_range = (0, int(xcfg["best_iteration"]) + 1)     # use only the trees up to the best iteration
        self.booster = xgb.Booster()
        self.booster.load_model(os.path.join(art_dir, "xgb_baseline.json"))

        self.ae, self.torch, self.ae_kind = None, None, None
        if load_ae:
            npz = os.path.join(art_dir, "ae_weights.npz")
            if os.path.exists(npz):                      # light path: numpy only, no PyTorch needed
                self.ae, self.ae_kind = _NumpyAE(npz), "numpy"
            else:                                        # original path: PyTorch weights
                import torch
                self.torch = torch
                self.ae = _build_autoencoder(len(self.cols))
                self.ae.load_state_dict(torch.load(os.path.join(art_dir, "ae.pt"), map_location="cpu"))
                self.ae.eval()
                self.ae_kind = "torch"

    # ---- preprocessing: identical arithmetic to training --------------------------------------
    def _prepare(self, df):
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        x = df[self.cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("Features must be finite numbers (found text, blanks or NaN)")
        if (x[:, self.amount_idx] < 0).any():
            raise ValueError("Amount must be >= 0")
        x = x.copy()
        x[:, self.amount_idx] = np.log1p(x[:, self.amount_idx])
        return ((x - self.mean) / self.scale).astype(np.float32)

    def _score(self, X, with_contribs):
        dm = xgb.DMatrix(X, feature_names=self.cols)
        prob = self.booster.predict(dm, iteration_range=self.xgb_range)
        contribs = None
        if with_contribs:       # exact TreeSHAP values from XGBoost itself (log-odds); last column is the bias
            contribs = self.booster.predict(dm, pred_contribs=True, iteration_range=self.xgb_range)[:, :-1]
        err = None
        if self.ae is not None:
            if self.ae_kind == "numpy":
                err = (X - self.ae(X)) ** 2
            else:
                t = self.torch.tensor(X)
                with self.torch.no_grad():
                    err = ((t - self.ae(t)) ** 2).numpy()
        return prob, contribs, err

    @staticmethod
    def _risk(xgb_flag, ae_flag):
        # HIGH   = XGBoost flags it (the main decision)
        # REVIEW = XGBoost does not flag it, but the autoencoder finds it unusual (second opinion; not separately validated)
        # LOW    = neither flags it
        return "HIGH" if xgb_flag else ("REVIEW" if ae_flag else "LOW")

    # ---- many transactions -----------------------------------------------------------------------
    def predict_df(self, df):
        prob, _, err = self._score(self._prepare(df), with_contribs=False)
        out = pd.DataFrame({"xgb_probability": prob, "xgb_flag": prob >= self.xgb_thr})
        if err is not None:
            s = err.mean(axis=1)
            out["ae_score"] = s
            out["ae_flag"] = s >= self.ae_thr
        else:
            out["ae_flag"] = False
        out["risk_level"] = [self._risk(a, b) for a, b in zip(out["xgb_flag"], out["ae_flag"])]
        return out

    # ---- one transaction, with explanations -------------------------------------------------
    def predict_one(self, features):
        prob, contribs, err = self._score(self._prepare(pd.DataFrame([features])), with_contribs=True)
        p = float(prob[0])
        xgb_flag = bool(p >= self.xgb_thr)
        top = np.argsort(-np.abs(contribs[0]))[:3]
        out = {"xgb_probability": round(p, 6), "xgb_flag": xgb_flag, "xgb_threshold": round(self.xgb_thr, 6),
               "xgb_top_drivers": [{"feature": self.cols[i], "contribution": round(float(contribs[0][i]), 3)} for i in top]}
        ae_flag = False
        if err is not None:
            s = float(err[0].mean())
            ae_flag = bool(s >= self.ae_thr)
            topae = np.argsort(err[0])[::-1][:3]
            out.update({"ae_score": round(s, 4), "ae_flag": ae_flag, "ae_threshold": round(self.ae_thr, 4),
                        "ae_top_errors": [{"feature": self.cols[i], "error": round(float(err[0][i]), 3)} for i in topae]})
        out["risk_level"] = self._risk(xgb_flag, ae_flag)
        return out
