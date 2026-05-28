import os
import subprocess
import sys

import streamlit as st  # type: ignore[import-not-found]
import streamlit.runtime as st_runtime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_with_streamlit() -> None:
    app_path = os.path.abspath(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, *sys.argv[1:]], check=False)


if __name__ == "__main__" and not st_runtime.exists():
    _run_with_streamlit()
    raise SystemExit(0)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KAN Toxicity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #090d12;
    --surface:   #0e151e;
    --card:      #131c27;
    --border:    #1e2d3d;
    --accent:    #00e5a0;
    --accent2:   #00aaff;
    --danger:    #ff4d6d;
    --muted:     #4a6070;
    --text:      #c8d8e8;
    --text-dim:  #6a8398;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}
.stApp { background-color: var(--bg) !important; font-family: var(--sans) !important; color: var(--text) !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px !important; }

.hero-banner {
    background: linear-gradient(135deg, #0a1520 0%, #0d1f2d 50%, #071018 100%);
    border: 1px solid var(--border); border-left: 4px solid var(--accent);
    border-radius: 4px 12px 12px 4px; padding: 2.2rem 2.8rem; margin-bottom: 2.5rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,229,160,0.07) 0%, transparent 70%);
}
.hero-title { font-family: var(--mono); font-size: 1.9rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; margin: 0 0 0.5rem 0; }
.hero-title span { color: var(--accent); }
.hero-subtitle { font-family: var(--sans); font-size: 0.92rem; color: var(--text-dim); margin: 0; letter-spacing: 0.02em; }
.hero-tag {
    display: inline-block; background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.25);
    color: var(--accent); font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
    padding: 3px 10px; border-radius: 2px; margin-right: 8px; margin-bottom: 1rem;
}
.section-label {
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.18em; color: var(--accent);
    text-transform: uppercase; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.metrics-row { display: flex; gap: 1.2rem; margin: 1.8rem 0; }
.metric-card { flex: 1; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.4rem 1.6rem; position: relative; overflow: hidden; }
.metric-card.toxic { border-top: 3px solid var(--danger); }
.metric-card.safe  { border-top: 3px solid var(--accent); }
.metric-card.total { border-top: 3px solid var(--accent2); }
.metric-num { font-family: var(--mono); font-size: 2.6rem; font-weight: 700; line-height: 1; margin-bottom: 0.3rem; }
.metric-card.toxic .metric-num { color: var(--danger); }
.metric-card.safe  .metric-num { color: var(--accent); }
.metric-card.total .metric-num { color: var(--accent2); }
.metric-label { font-size: 0.8rem; color: var(--text-dim); font-weight: 500; letter-spacing: 0.04em; }
.metric-pct { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

.upload-hint { background: var(--card); border: 1px dashed var(--border); border-radius: 10px; padding: 1.6rem 2rem; margin-bottom: 1rem; color: var(--text-dim); font-size: 0.88rem; line-height: 1.7; }
.upload-hint code { background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.2); color: var(--accent); font-family: var(--mono); padding: 1px 6px; border-radius: 3px; font-size: 0.82rem; }

.prob-bar-wrap { width: 100%; background: var(--surface); border-radius: 4px; height: 8px; overflow: hidden; margin-top: 4px; }
.prob-bar-fill-safe  { height: 100%; background: var(--accent); border-radius: 4px; }
.prob-bar-fill-toxic { height: 100%; background: var(--danger); border-radius: 4px; }

.results-table-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-top: 0.5rem; }
.pill-toxic { display: inline-block; background: rgba(255,77,109,0.12); border: 1px solid rgba(255,77,109,0.35); color: var(--danger); font-family: var(--mono); font-size: 0.75rem; padding: 2px 10px; border-radius: 20px; font-weight: 700; }
.pill-safe  { display: inline-block; background: rgba(0,229,160,0.1);  border: 1px solid rgba(0,229,160,0.3);  color: var(--accent); font-family: var(--mono); font-size: 0.75rem; padding: 2px 10px; border-radius: 20px; font-weight: 700; }

div[data-testid="stFileUploader"] { background: var(--card) !important; border: 1px dashed var(--border) !important; border-radius: 10px !important; }
div[data-testid="stDataFrame"] > div { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--card) !important; }
div[data-testid="stDataFrame"] table { background: var(--card) !important; color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }
div[data-testid="stDataFrame"] th { background: var(--surface) !important; color: var(--accent) !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; border-bottom: 1px solid var(--border) !important; }
div[data-testid="stDataFrame"] td { border-color: var(--border) !important; color: var(--text) !important; }

.stDownloadButton > button { background: transparent !important; border: 1px solid var(--accent) !important; color: var(--accent) !important; font-family: var(--mono) !important; font-size: 0.78rem !important; letter-spacing: 0.08em !important; border-radius: 6px !important; padding: 0.5rem 1.4rem !important; }
.stDownloadButton > button:hover { background: var(--accent) !important; color: var(--bg) !important; }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Definitions — MUST EXACTLY MATCH TRAINING NOTEBOOK ────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DENSE_END   = 801
SPARSE_DIM  = 1644 - DENSE_END   # 843


class DenseEncoder(nn.Module):
    def __init__(self, in_dim=801, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),   # 384 → 192
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    def forward(self, x): return self.net(x)


class SparseEncoder(nn.Module):
    def __init__(self, in_dim=SPARSE_DIM, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),   # 384 → 192
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    def forward(self, x): return self.net(x)


class ModalityKANFusion(nn.Module):
    """Truncated Fourier-series KAN fusion layer — matches training notebook."""
    def __init__(self, in_dim, out_dim=384, fourier_terms=8):
        super().__init__()
        self.linear       = nn.Linear(in_dim, out_dim)
        self.fourier_terms = fourier_terms
        self.a0 = nn.Parameter(torch.zeros(out_dim))
        self.an = nn.Parameter(torch.randn(fourier_terms, out_dim) * 0.01)
        self.bn = nn.Parameter(torch.randn(fourier_terms, out_dim) * 0.01)
        self.post = nn.Sequential(nn.BatchNorm1d(out_dim), nn.Dropout(0.3))

    def forward(self, x):
        base  = self.linear(x)
        fused = self.a0.clone()
        for n in range(1, self.fourier_terms + 1):
            fused = fused + self.an[n-1] * torch.cos(n * base) + self.bn[n-1] * torch.sin(n * base)
        fused = fused + base   # residual
        return self.post(fused)


class DeepADR_KAN(nn.Module):
    def __init__(self, dense_dim=801, sparse_dim=SPARSE_DIM, fusion_dim=384, fourier_terms=8):
        super().__init__()
        self.dense_encoder  = DenseEncoder(dense_dim,  hidden=384)
        self.sparse_encoder = SparseEncoder(sparse_dim, hidden=384)
        # 192 + 192 = 384 fused input
        self.fusion = ModalityKANFusion(192 + 192, out_dim=fusion_dim, fourier_terms=fourier_terms)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        d   = self.dense_encoder(x[:, :DENSE_END])
        s   = self.sparse_encoder(x[:, DENSE_END:])
        z   = torch.cat([d, s], dim=1)
        f   = self.fusion(z)
        return self.head(f)


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model():
    m = DeepADR_KAN().to(device)
    model_path = os.path.join(BASE_DIR, "sparse_modality_kan.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model weights: {model_path}")
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    return m


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5a0;text-transform:uppercase;margin-bottom:0.8rem'>SYSTEM INFO</div>
    </div>
    <div style='background:#0e151e;border:1px solid #1e2d3d;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1.2rem'>
        <div style='font-family:monospace;font-size:0.75rem;color:#4a6070;margin-bottom:0.3rem'>INFERENCE DEVICE</div>
        <div style='font-family:monospace;font-size:1rem;color:#00aaff;font-weight:700'>{device_label}</div>
    </div>
    <div style='font-family:monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5a0;text-transform:uppercase;margin-bottom:0.8rem'>INPUT FORMAT</div>
    <div style='font-size:0.84rem;color:#6a8398;line-height:1.9'>
        • CSV, any number of rows<br>
        • Exactly <code style='background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.2);color:#00e5a0;padding:1px 5px;border-radius:3px'>1644</code> feature columns<br>
        • No label / index column<br>
        • Dense cols <code style='background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.2);color:#00e5a0;padding:1px 5px;border-radius:3px'>0–800</code>
          → MinMaxScaled<br>
        • Sparse cols <code style='background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.2);color:#00e5a0;padding:1px 5px;border-radius:3px'>801–1643</code>
          → log1p
    </div>
    <hr style='border-color:#1e2d3d;margin:1.2rem 0'>
    <div style='font-family:monospace;font-size:0.65rem;letter-spacing:0.2em;color:#00e5a0;text-transform:uppercase;margin-bottom:0.8rem'>ARCHITECTURE</div>
    <div style='font-size:0.82rem;color:#6a8398;line-height:2'>
        🔷 Dual-stream encoder (384→192)<br>
        🔷 Modality-KAN fusion (384-dim)<br>
        🔷 8 Fourier terms per neuron<br>
        🔷 Head: 384→192→64→1<br>
        🔷 Sigmoid output (binary)
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div>
        <span class='hero-tag'>SPARSE MODALITY</span>
        <span class='hero-tag'>FOURIER-KAN</span>
        <span class='hero-tag'>TOX21</span>
    </div>
    <div class='hero-title'>🧬 <span>Molecular</span> Toxicity Predictor</div>
    <p class='hero-subtitle'>
        Deep ADR · Sparse-modality Kolmogorov–Arnold Network ·
        Upload molecular fingerprints to classify compounds as toxic or non-toxic
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Load model ──────────────────────────────────────────────────────────────
model_ok = True
try:
    scaler = load_scaler()
    model  = load_model()
except Exception as e:
    st.error(f"**Model load failed** — `{e}`")
    st.markdown("""
    <div style='font-size:0.85rem;color:#6a8398;margin-top:0.5rem'>
    Ensure <code>scaler.pkl</code> and <code>sparse_modality_kan.pth</code>
    are in the same directory as <code>app.py</code>.
    </div>""", unsafe_allow_html=True)
    model_ok = False


# ─── Upload ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>01 — DATA INPUT</div>", unsafe_allow_html=True)
st.markdown("""
<div class='upload-hint'>
    Upload a <code>.csv</code> file with exactly <code>1644</code> feature columns (no label column).
    Each row = one compound. Dense features <code>0–800</code> are MinMax-scaled;
    sparse features <code>801–1643</code> are log1p-transformed before inference.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop molecular feature CSV here", type=["csv"], label_visibility="collapsed")


# ─── Prediction ──────────────────────────────────────────────────────────────
if uploaded_file is not None and model_ok:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1645:
            df = df.iloc[:, :-1]
            st.info("ℹ️ Label column detected and removed automatically.")
        st.markdown("<div class='section-label' style='margin-top:2rem'>02 — RAW DATA PREVIEW</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:monospace;font-size:0.78rem;color:#4a6070;margin-bottom:0.5rem'>{df.shape[0]} rows × {df.shape[1]} columns detected</div>", unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True, height=240)

        if df.shape[1] != 1644:
            st.error(f"❌ **Column mismatch** — expected `1644` features, got `{df.shape[1]}`.\nRemove any label, index, or extra columns.")
            st.stop()

        with st.spinner("Running inference…"):
            X        = np.maximum(df.values.astype(np.float32), 0)
            X_dense  = scaler.transform(X[:, :DENSE_END])         # MinMaxScaler (from scaler.pkl)
            X_sparse = np.log1p(X[:, DENSE_END:])                  # log1p, matches training
            X_input  = torch.tensor(np.hstack([X_dense, X_sparse]), dtype=torch.float32).to(device)

            if X_input.ndim == 1:
                X_input = X_input.unsqueeze(0)

            with torch.no_grad():
                logits = model(X_input)
                probs  = np.atleast_1d(torch.sigmoid(logits).cpu().numpy().squeeze())

        preds  = (probs >= 0.5).astype(int)
        n_tot  = len(preds)
        n_tox  = int((preds == 1).sum())
        n_safe = int((preds == 0).sum())

        # ── Summary cards ──
        st.markdown("<div class='section-label' style='margin-top:2rem'>03 — SUMMARY</div>", unsafe_allow_html=True)
        pct_tox  = f"{100 * n_tox  / n_tot:.1f}%" if n_tot else "—"
        pct_safe = f"{100 * n_safe / n_tot:.1f}%" if n_tot else "—"
        st.markdown(f"""
        <div class='metrics-row'>
            <div class='metric-card total'>
                <div class='metric-num'>{n_tot}</div>
                <div class='metric-label'>Total Compounds</div>
                <div class='metric-pct'>100%</div>
            </div>
            <div class='metric-card toxic'>
                <div class='metric-num'>{n_tox}</div>
                <div class='metric-label'>☠️ Toxic</div>
                <div class='metric-pct'>{pct_tox} of dataset</div>
            </div>
            <div class='metric-card safe'>
                <div class='metric-num'>{n_safe}</div>
                <div class='metric-label'>✅ Non-Toxic</div>
                <div class='metric-pct'>{pct_safe} of dataset</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-compound table ──
        st.markdown("<div class='section-label' style='margin-top:2rem'>04 — COMPOUND RESULTS</div>", unsafe_allow_html=True)
        rows_html = ""
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            is_toxic = pred == 1
            bar_cls  = "prob-bar-fill-toxic" if is_toxic else "prob-bar-fill-safe"
            pill     = "<span class='pill-toxic'>TOXIC</span>" if is_toxic else "<span class='pill-safe'>SAFE</span>"
            prob_col = "#ff4d6d" if is_toxic else "#00e5a0"
            rows_html += f"""
            <tr style='border-bottom:1px solid #1e2d3d'>
                <td style='padding:10px 16px;font-family:monospace;font-size:0.82rem;color:#4a6070'>#{i+1:04d}</td>
                <td style='padding:10px 16px'>{pill}</td>
                <td style='padding:10px 16px;min-width:180px'>
                    <div style='font-family:monospace;font-size:0.78rem;color:{prob_col};margin-bottom:4px'>{prob:.4f}</div>
                    <div class='prob-bar-wrap'><div class='{bar_cls}' style='width:{prob*100:.1f}%'></div></div>
                </td>
                <td style='padding:10px 16px;font-family:monospace;font-size:0.78rem;color:#4a6070'>{"HIGH RISK" if is_toxic else "LOW RISK"}</td>
            </tr>"""

        st.markdown(f"""
        <div class='results-table-wrap'>
            <table style='width:100%;border-collapse:collapse'>
                <thead><tr style='background:#0e151e;border-bottom:2px solid #1e2d3d'>
                    <th style='padding:10px 16px;text-align:left;font-family:monospace;font-size:0.68rem;letter-spacing:0.14em;color:#4a6070;text-transform:uppercase'>COMPOUND</th>
                    <th style='padding:10px 16px;text-align:left;font-family:monospace;font-size:0.68rem;letter-spacing:0.14em;color:#4a6070;text-transform:uppercase'>VERDICT</th>
                    <th style='padding:10px 16px;text-align:left;font-family:monospace;font-size:0.68rem;letter-spacing:0.14em;color:#4a6070;text-transform:uppercase'>PROBABILITY</th>
                    <th style='padding:10px 16px;text-align:left;font-family:monospace;font-size:0.68rem;letter-spacing:0.14em;color:#4a6070;text-transform:uppercase'>RISK LEVEL</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Export ──
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        results_df = pd.DataFrame({
            "Compound":             [f"#{i+1:04d}" for i in range(n_tot)],
            "Prediction":           ["Toxic" if p == 1 else "Non-Toxic" for p in preds],
            "Toxicity_Probability": np.round(probs, 6),
            "Risk_Level":           ["HIGH" if p == 1 else "LOW" for p in preds],
        })
        col_dl, col_msg = st.columns([2, 5])
        with col_dl:
            st.download_button(
                label="⬇  EXPORT RESULTS CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="toxicity_predictions.csv",
                mime="text/csv",
            )
        with col_msg:
            st.markdown(
                f"<div style='padding-top:0.55rem;font-size:0.82rem;color:#4a6070;font-family:monospace'>✓ inference complete · {n_tot} compounds scored</div>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"**Prediction error** — {e}")

elif not model_ok:
    pass

else:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#1e2d3d'>
        <div style='font-size:3.5rem;margin-bottom:1rem'>🧬</div>
        <div style='font-family:monospace;font-size:0.9rem;color:#4a6070;letter-spacing:0.08em'>AWAITING MOLECULAR DATA</div>
        <div style='font-size:0.8rem;color:#2a3d4d;margin-top:0.5rem'>Upload a CSV file above to begin analysis</div>
    </div>
    """, unsafe_allow_html=True)