"""
fraud_final.py
FraudShield – Complete Dashboard + Predict Tab (FIXED)
"""
import io
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -------------------------------------------------
# 1. Page & CSS
# -------------------------------------------------
st.set_page_config(page_title="FraudShield", layout="wide", page_icon="shield")

# -------------------------------------------------
# 2. Title + Icon (Perfectly Centered)
# -------------------------------------------------
st.markdown("""
<style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        margin: 20px 0;
    }
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        color: #818cf8;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <img src="https://img.icons8.com/fluency/96/000000/shield.png" width="80">
    <h1 class="title-text">FraudShield</h1>
</div>
""", unsafe_allow_html=True)

st.caption("Edge-level Fraud Detection – GraphSAGE + XGBoost + Explainability", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# 2. Title + Upload
# -------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

# -------------------------------------------------
# 3. Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Controls")
    NONFRAUD_RATIO = st.slider("Non-fraud per fraud", 1, 20, 10)
    EPOCHS = st.slider("GNN Epochs", 10, 100, 35)
    HIDDEN_DIM = st.selectbox("Hidden Dim", [64, 128, 256], index=1)

# -------------------------------------------------
# 4. File Upload
# -------------------------------------------------
st.info("Run with: `streamlit run fraud_final.py --server.maxUploadSize=1024`")
uploaded_file = st.file_uploader(
    "Upload `onlinefraud.csv` (up to 1 GB)",
    type=["csv"],
    help="Required: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud"
)

if uploaded_file is None:
    st.info("Upload your CSV to start.")
    st.stop()

# -------------------------------------------------
# 5. Load CSV in chunks
# -------------------------------------------------
@st.cache_data
def load_large_csv(_file):
    chunk_size = 100_000
    chunks = []
    total = 0
    prog = st.progress(0)
    stat = st.empty()
    _file.seek(0)
    for i, chunk in enumerate(pd.read_csv(_file, chunksize=chunk_size)):
        chunks.append(chunk)
        total += len(chunk)
        prog.progress(min((i+1)*chunk_size/(total+1), 1.0))
        stat.text(f"Loading... {total:,} rows")
    prog.empty()
    stat.empty()
    df = pd.concat(chunks, ignore_index=True)
    st.success(f"Loaded {len(df):,} rows ({_file.size/1e6:.1f} MB)")
    return df

df_raw = load_large_csv(uploaded_file)

# -------------------------------------------------
# 6. Preprocess & balance
# -------------------------------------------------
@st.cache_data
def preprocess_and_balance(_df, _ratio):
    required = {'step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig',
                'nameDest','oldbalanceDest','newbalanceDest','isFraud'}
    if not required.issubset(_df.columns):
        st.error(f"Missing columns: {required - set(_df.columns)}")
        st.stop()

    le_type = LabelEncoder()
    _df = _df.copy()
    _df['type_enc'] = le_type.fit_transform(_df['type'].astype(str))
    _df['errOrig'] = _df['newbalanceOrig'] + _df['amount'] - _df['oldbalanceOrg']
    _df['errDest'] = _df['oldbalanceDest'] + _df['amount'] - _df['newbalanceDest']

    frauds = _df[_df['isFraud'] == 1]
    nonfrauds = _df[_df['isFraud'] == 0]
    n_non = min(len(nonfrauds), len(frauds) * _ratio)
    non_sample = nonfrauds.sample(n=n_non, random_state=42)
    df_bal = pd.concat([frauds, non_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_bal, le_type

df_bal, le_type = preprocess_and_balance(df_raw, NONFRAUD_RATIO)

# -------------------------------------------------
# 7. Build graph
# -------------------------------------------------
@st.cache_data
def build_graph(_df):
    accounts = pd.concat([_df['nameOrig'], _df['nameDest']]).unique()
    acc_to_idx = {acc: i for i, acc in enumerate(accounts)}
    num_nodes = len(accounts)

    src = _df['nameOrig'].map(acc_to_idx).values
    dst = _df['nameDest'].map(acc_to_idx).values
    edge_index = np.vstack([src, dst])

    sent_sum = np.bincount(src, weights=_df['amount'].values, minlength=num_nodes)
    recv_sum = np.bincount(dst, weights=_df['amount'].values, minlength=num_nodes)
    sent_cnt = np.bincount(src, minlength=num_nodes)
    recv_cnt = np.bincount(dst, minlength=num_nodes)
    node_feats = np.column_stack([
        sent_sum, recv_sum, sent_cnt, recv_cnt,
        np.divide(sent_sum, sent_cnt, where=sent_cnt>0, out=np.zeros_like(sent_sum)),
        np.divide(recv_sum, recv_cnt, where=recv_cnt>0, out=np.zeros_like(recv_sum))
    ])
    node_feats = StandardScaler().fit_transform(node_feats)

    edge_numeric = StandardScaler().fit_transform(_df[['amount', 'errOrig', 'errDest', 'step']].values)
    edge_type_idx = _df['type_enc'].values
    y = _df['isFraud'].values

    return node_feats, edge_index, edge_numeric, edge_type_idx, y, num_nodes

node_feats, edge_index, edge_numeric, edge_type_idx, y, num_nodes = build_graph(df_bal)

train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
train_mask = np.zeros(len(y), dtype=bool); train_mask[train_idx] = True
test_mask = np.zeros(len(y), dtype=bool); test_mask[test_idx] = True

data = Data(
    x=torch.tensor(node_feats, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_numeric, dtype=torch.float),
    y=torch.tensor(y, dtype=torch.long),
    type_idx=torch.tensor(edge_type_idx, dtype=torch.long),
    train_mask=torch.tensor(train_mask),
    test_mask=torch.tensor(test_mask)
).to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 8. GNN Model
# -------------------------------------------------
class ImprovedEdgeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, n_types, dropout=0.3):
        super().__init__()
        self.type_emb = torch.nn.Embedding(n_types, 8)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        mlp_in = hidden_channels*2 + edge_attr_dim + 8
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels//2), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels//2, 2)
        )
    def forward(self, x, edge_index, edge_attr, type_idx):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr, self.type_emb(type_idx)], dim=1)
        return self.edge_mlp(edge_input)

# -------------------------------------------------
# 9. Train GNN
# -------------------------------------------------
@st.cache_resource
def train_gnn(_data, _hidden_dim, _epochs):
    model = ImprovedEdgeGNN(
        in_channels=_data.x.size(1),
        hidden_channels=_hidden_dim,
        edge_attr_dim=edge_numeric.shape[1],
        n_types=len(le_type.classes_)
    ).to(_data.x.device)

    train_labels = _data.y[_data.train_mask].cpu().numpy()
    classes = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(_data.x.device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    prog = st.progress(0)
    stat = st.empty()
    for epoch in range(_epochs):
        optimizer.zero_grad()
        out = model(_data.x, _data.edge_index, _data.edge_attr, _data.type_idx)
        loss = criterion(out[_data.train_mask], _data.y[_data.train_mask])
        loss.backward()
        optimizer.step()
        prog.progress((epoch+1)/_epochs)
        stat.text(f"Epoch {epoch+1}/{_epochs} – Loss: {loss.item():.4f}")
    prog.empty()
    stat.empty()
    return model

with st.spinner("Training GraphSAGE..."):
    gnn_model = train_gnn(data, HIDDEN_DIM, EPOCHS)

# -------------------------------------------------
# 10. XGBoost
# -------------------------------------------------
@st.cache_resource
def train_xgb():
    X_train = np.hstack([
        edge_numeric[train_idx],
        data.x[data.edge_index[0][train_idx]].cpu().numpy(),
        data.x[data.edge_index[1][train_idx]].cpu().numpy()
    ])
    X_test = np.hstack([
        edge_numeric[test_idx],
        data.x[data.edge_index[0][test_idx]].cpu().numpy(),
        data.x[data.edge_index[1][test_idx]].cpu().numpy()
    ])
    pos_weight = len(y[train_idx][y[train_idx]==0]) / max(1, len(y[train_idx][y[train_idx]==1]))
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=pos_weight, eval_metric='logloss')
    model.fit(X_train, y[train_idx])
    return model, X_test

with st.spinner("Training XGBoost..."):
    xgb_model, X_test = train_xgb()

# -------------------------------------------------
# 11. Predictions
# -------------------------------------------------
with torch.no_grad():
    gnn_out = gnn_model(data.x, data.edge_index, data.edge_attr, data.type_idx)
    gnn_probs = F.softmax(gnn_out, dim=1)[:,1].cpu().numpy()
    gnn_preds = (gnn_probs > 0.5).astype(int)

xgb_probs = xgb_model.predict_proba(X_test)[:,1]
xgb_preds = (xgb_probs > 0.5).astype(int)
labels_test = y[test_idx]

# -------------------------------------------------
# 12. Safe Metrics
# -------------------------------------------------
def metrics(probs, preds, labels):
    roc_auc = roc_auc_score(labels, probs)
    try:
        p, r, _ = precision_recall_curve(labels, probs + 1e-12)
        idx = np.argsort(r)
        r, p = r[idx], p[idx]
        pr_auc = 0.0 if r.size == 0 or r[-1] == 0 else auc(r, p)
    except Exception:
        pr_auc = 0.0
    return {
        "ROC-AUC"   : roc_auc,
        "PR-AUC"    : pr_auc,
        "F1"        : f1_score(labels, preds, zero_division=0),
        "Precision" : precision_score(labels, preds, zero_division=0),
        "Recall"    : recall_score(labels, preds, zero_division=0)
    }

gnn_metrics = metrics(gnn_probs[test_idx], gnn_preds[test_idx], labels_test)
xgb_metrics = metrics(xgb_probs, xgb_preds, labels_test)

# -------------------------------------------------
# 13. UI – Dashboard
# -------------------------------------------------
st.markdown("## Model Performance")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### GraphSAGE GNN")
    for k, v in gnn_metrics.items():
        st.markdown(f"<div class='metric-card'><h4>{k}</h4><h2 style='color:#818cf8'>{v:.4f}</h2></div>", unsafe_allow_html=True)
with c2:
    st.markdown("### XGBoost")
    for k, v in xgb_metrics.items():
        st.markdown(f"<div class='metric-card'><h4>{k}</h4><h2 style='color:#34d399'>{v:.4f}</h2></div>", unsafe_allow_html=True)

# -------------------------------------------------
# 14. ROC / PR Curves
# -------------------------------------------------
fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall"))
fpr_g, tpr_g, _ = roc_curve(labels_test, gnn_probs[test_idx])
fpr_x, tpr_x, _ = roc_curve(labels_test, xgb_probs)
prec_g, rec_g, _ = precision_recall_curve(labels_test, gnn_probs[test_idx])
prec_x, rec_x, _ = precision_recall_curve(labels_test, xgb_probs)

fig.add_trace(go.Scatter(x=fpr_g, y=tpr_g, name=f"GNN ({gnn_metrics['ROC-AUC']:.3f})", line=dict(color="#818cf8")), row=1, col=1)
fig.add_trace(go.Scatter(x=fpr_x, y=tpr_x, name=f"XGB ({xgb_metrics['ROC-AUC']:.3f})", line=dict(color="#34d399")), row=1, col=1)
fig.add_trace(go.Scatter(x=rec_g, y=prec_g, name=f"GNN ({gnn_metrics['PR-AUC']:.3f})", line=dict(color="#818cf8")), row=1, col=2)
fig.add_trace(go.Scatter(x=rec_x, y=prec_x, name=f"XGB ({xgb_metrics['PR-AUC']:.3f})", line=dict(color="#34d399")), row=1, col=2)
fig.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# -------------------------------------------------
# 15. EXPLAINABILITY + ABOUT + PREDICT
# -------------------------------------------------
st.markdown("## Analysis & Prediction", unsafe_allow_html=True)

# Compute edge_inputs once
with st.spinner("Preparing edge features..."):
    with torch.no_grad():
        h = gnn_model.conv2(
            gnn_model.bn1(F.relu(gnn_model.conv1(data.x, data.edge_index))),
            data.edge_index
        )
        src_h = h[data.edge_index[0]]
        dst_h = h[data.edge_index[1]]
        type_emb = gnn_model.type_emb(data.type_idx)

    edge_inputs = np.hstack([
        src_h.cpu().numpy(),
        dst_h.cpu().numpy(),
        edge_numeric,
        type_emb.cpu().numpy()
    ])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["SHAP Explainer", "LIME Inspector", "About", "Predict"])

# =============================================
# TAB 1: SHAP
# =============================================
with tab1:
    st.markdown("### Interactive SHAP Force Plot")
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.spinner("Computing SHAP values..."):
            def shap_predict(x):
                with torch.no_grad():
                    x_t = torch.tensor(x, dtype=torch.float, device=data.x.device)
                    return F.softmax(gnn_model.edge_mlp(x_t), dim=1).cpu().numpy()[:, 1]

            bg = edge_inputs[train_idx[:50]]
            test_edge = edge_inputs[test_idx[0]:test_idx[0]+1]

            explainer = shap.KernelExplainer(shap_predict, bg)
            shap_vals = explainer.shap_values(test_edge, nsamples=100)

            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_vals,
                test_edge,
                show=False,
                matplotlib=False
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            st.components.v1.html(shap_html, height=450, scrolling=False)

    with col2:
        st.markdown("### Waterfall")
        fig, ax = plt.subplots(figsize=(6, 5))
        shap.waterfall_plot(
            shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value, data=test_edge[0]),
            show=False, max_display=10
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1e293b')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()

    st.markdown("### Global SHAP")
    with st.spinner("Computing global SHAP..."):
        shap_vals_all = explainer.shap_values(edge_inputs[test_idx[:30]], nsamples=100)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals_all, edge_inputs[test_idx[:30]], plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        st.pyplot(fig)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("Force Plot (HTML)", data=shap_html, file_name="shap_force.html", mime="text/html")
    with col_d2:
        buf.seek(0)
        st.download_button("Waterfall (PNG)", data=buf.read(), file_name="shap_waterfall.png", mime="image/png")

# =============================================
# TAB 2: LIME (White Text)
# =============================================
with tab2:
    st.markdown("### LIME Local Explanation")
    edge_to_explain = st.selectbox("Select Edge", test_idx[:20], format_func=lambda x: f"Edge {x}")
    sample_feat = edge_inputs[edge_to_explain:edge_to_explain+1]

    lime_expl = LimeTabularExplainer(
        edge_inputs[train_idx],
        mode="classification",
        feature_names=[f"feat_{i}" for i in range(edge_inputs.shape[1])],
        class_names=["Non-Fraud", "Fraud"]
    )

    def lime_predict(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float, device=data.x.device)
            return F.softmax(gnn_model.edge_mlp(x_t), dim=1).cpu().numpy()

    with st.spinner("Generating LIME..."):
        exp = lime_expl.explain_instance(sample_feat[0], lime_predict, num_features=10)

    lime_html = exp.as_html()
    dark_lime = lime_html.replace(
        "</head>",
        "<style>body{background:#0f172a;color:#ffffff !important;font-family:'Segoe UI'} "
        ".lime table{color:#ffffff !important;border-color:#6366f1}</style></head>"
    )
    st.components.v1.html(dark_lime, height=700, scrolling=True)

    probs = lime_predict(sample_feat)[0]
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Non-Fraud", f"{probs[0]:.1%}")
    with col_p2:
        st.metric("Fraud Risk", f"{probs[1]:.1%}")

    st.download_button("Download LIME", data=dark_lime, file_name=f"lime_edge_{edge_to_explain}.html", mime="text/html")

# =============================================
# TAB 3: ABOUT – Full Transaction Type Histogram
# =============================================
with tab3:
    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Transactions:** {len(df_bal):,}")
    with col2:
        st.write(f"**Fraud Rate:** {df_bal['isFraud'].mean():.2%}")

    st.markdown("### Transaction Type Distribution")
    type_counts = df_bal['type'].value_counts().sort_values(ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=type_counts.index,
        y=type_counts.values,
        text=type_counts.values,
        textposition='outside',
        marker_color='#6366f1',
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
    )])
    fig.update_layout(
        title="All Transaction Types",
        xaxis_title="Type",
        yaxis_title="Count",
        template="plotly_dark",
        height=500,
        margin=dict(t=80)
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# TAB 4: PREDICT – Real Fields (No Errors)
# =============================================
with tab4:
    st.markdown("### Live Fraud Prediction")
    st.info("Enter real transaction details → Click **Predict**")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            step = st.number_input("Step (Hour)", min_value=1, max_value=743, value=1, key="step")
            amount = st.number_input("Amount", min_value=0.0, value=9839.64, step=100.0, key="amt")
            nameOrig = st.text_input("Origin Account (nameOrig)", value="C1231006815", key="orig")
            oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=170136.0, step=100.0, key="oldorg")
            newbalanceOrig = st.number_input("Origin New Balance", min_value=0.0, value=160296.36, step=100.0, key="neworg")

        with col2:
            type_str = st.selectbox(
                "Transaction Type",
                options=le_type.classes_.tolist(),
                index=0,
                key="type"
            )
            nameDest = st.text_input("Destination Account (nameDest)", value="M1979787155", key="dest")
            oldbalanceDest = st.number_input("Dest Old Balance", min_value=0.0, value=0.0, step=100.0, key="olddest")
            newbalanceDest = st.number_input("Dest New Balance", min_value=0.0, value=0.0, step=100.0, key="newdest")

        submitted = st.form_submit_button("Predict Fraud Risk", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Predicting..."):
            # --- 1. Compute errors ---
            errOrig = newbalanceOrig + amount - oldbalanceOrg
            errDest = oldbalanceDest + amount - newbalanceDest

            # --- 2. Get node indices (or use mean if missing) ---
            acc_to_idx = {acc: i for i, acc in enumerate(pd.concat([df_bal['nameOrig'], df_bal['nameDest']]).unique())}
            src_idx = acc_to_idx.get(nameOrig, -1)
            dst_idx = acc_to_idx.get(nameDest, -1)

            with torch.no_grad():
                # Forward pass to get node embeddings
                h = gnn_model.conv2(
                    gnn_model.bn1(F.relu(gnn_model.conv1(data.x, data.edge_index))),
                    data.edge_index
                )

                # Use real embeddings if accounts exist, else mean
                if src_idx != -1:
                    src_emb = h[src_idx]
                else:
                    src_emb = h.mean(dim=0)

                if dst_idx != -1:
                    dst_emb = h[dst_idx]
                else:
                    dst_emb = h.mean(dim=0)

            # --- 3. Scale edge numeric ---
            edge_raw = np.array([[amount, errOrig, errDest, step]])
            scaler = StandardScaler().fit(edge_numeric)
            edge_scaled = scaler.transform(edge_raw)[0]

            # --- 4. Type embedding ---
            type_idx = np.where(le_type.classes_ == type_str)[0][0]
            type_emb_vec = gnn_model.type_emb.weight[type_idx].detach().cpu().numpy()

            # --- 5. Final input ---
            input_vec = np.concatenate([
                src_emb.cpu().numpy(),
                dst_emb.cpu().numpy(),
                edge_scaled,
                type_emb_vec
            ]).reshape(1, -1)

            # --- 6. Predict ---
            with torch.no_grad():
                logits = gnn_model.edge_mlp(
                    torch.tensor(input_vec, dtype=torch.float, device=data.x.device)
                )
                fraud_prob = F.softmax(logits, dim=1)[0, 1].item()

        # --- Results ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraud Probability", f"{fraud_prob:.1%}")
        with col2:
            status = "FRAUD" if fraud_prob > 0.5 else "SAFE"
            color = "red" if fraud_prob > 0.5 else "green"
            st.markdown(f"<h2 style='color:{color}; text-align:center; margin:0'>{status}</h2>", unsafe_allow_html=True)

        with st.expander("Input Summary"):
            st.json({
                "step": step,
                "type": type_str,
                "amount": amount,
                "nameOrig": nameOrig,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "nameDest": nameDest,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "errOrig": round(errOrig, 2),
                "errDest": round(errDest, 2)
            })


            
# =============================================
# FINAL METRICS
# =============================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### Final Metrics")
    for name, mets in [("GNN", gnn_metrics), ("XGBoost", xgb_metrics)]:
        st.markdown(f"**{name}**")
        for k, v in mets.items():
            color = "#34d399" if name == "XGBoost" else "#818cf8"
            st.markdown(f"<small style='color:{color}'>{k}: <b>{v:.4f}</b></small>", unsafe_allow_html=True)
        st.markdown("")

#st.success("FraudShield – Ready for Production")
"""
fraud_final.py
FraudShield – Complete Dashboard + Predict Tab (FIXED)
"""
import io
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -------------------------------------------------
# 1. Page & CSS
# -------------------------------------------------
st.set_page_config(page_title="FraudShield", layout="wide", page_icon="shield")

# -------------------------------------------------
# 2. Title + Icon (Perfectly Centered)
# -------------------------------------------------
st.markdown("""
<style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        margin: 20px 0;
    }
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        color: #818cf8;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <img src="https://img.icons8.com/fluency/96/000000/shield.png" width="80">
    <h1 class="title-text">FraudShield</h1>
</div>
""", unsafe_allow_html=True)

st.caption("Edge-level Fraud Detection – GraphSAGE + XGBoost + Explainability", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# 2. Title + Upload
# -------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

# -------------------------------------------------
# 3. Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Controls")
    NONFRAUD_RATIO = st.slider("Non-fraud per fraud", 1, 20, 10)
    EPOCHS = st.slider("GNN Epochs", 10, 100, 35)
    HIDDEN_DIM = st.selectbox("Hidden Dim", [64, 128, 256], index=1)

# -------------------------------------------------
# 4. File Upload
# -------------------------------------------------
st.info("Run with: `streamlit run fraud_final.py --server.maxUploadSize=1024`")
uploaded_file = st.file_uploader(
    "Upload `onlinefraud.csv` (up to 1 GB)",
    type=["csv"],
    help="Required: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud"
)

if uploaded_file is None:
    st.info("Upload your CSV to start.")
    st.stop()

# -------------------------------------------------
# 5. Load CSV in chunks
# -------------------------------------------------
@st.cache_data
def load_large_csv(_file):
    chunk_size = 100_000
    chunks = []
    total = 0
    prog = st.progress(0)
    stat = st.empty()
    _file.seek(0)
    for i, chunk in enumerate(pd.read_csv(_file, chunksize=chunk_size)):
        chunks.append(chunk)
        total += len(chunk)
        prog.progress(min((i+1)*chunk_size/(total+1), 1.0))
        stat.text(f"Loading... {total:,} rows")
    prog.empty()
    stat.empty()
    df = pd.concat(chunks, ignore_index=True)
    st.success(f"Loaded {len(df):,} rows ({_file.size/1e6:.1f} MB)")
    return df

df_raw = load_large_csv(uploaded_file)

# -------------------------------------------------
# 6. Preprocess & balance
# -------------------------------------------------
@st.cache_data
def preprocess_and_balance(_df, _ratio):
    required = {'step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig',
                'nameDest','oldbalanceDest','newbalanceDest','isFraud'}
    if not required.issubset(_df.columns):
        st.error(f"Missing columns: {required - set(_df.columns)}")
        st.stop()

    le_type = LabelEncoder()
    _df = _df.copy()
    _df['type_enc'] = le_type.fit_transform(_df['type'].astype(str))
    _df['errOrig'] = _df['newbalanceOrig'] + _df['amount'] - _df['oldbalanceOrg']
    _df['errDest'] = _df['oldbalanceDest'] + _df['amount'] - _df['newbalanceDest']

    frauds = _df[_df['isFraud'] == 1]
    nonfrauds = _df[_df['isFraud'] == 0]
    n_non = min(len(nonfrauds), len(frauds) * _ratio)
    non_sample = nonfrauds.sample(n=n_non, random_state=42)
    df_bal = pd.concat([frauds, non_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_bal, le_type

df_bal, le_type = preprocess_and_balance(df_raw, NONFRAUD_RATIO)

# -------------------------------------------------
# 7. Build graph
# -------------------------------------------------
@st.cache_data
def build_graph(_df):
    accounts = pd.concat([_df['nameOrig'], _df['nameDest']]).unique()
    acc_to_idx = {acc: i for i, acc in enumerate(accounts)}
    num_nodes = len(accounts)

    src = _df['nameOrig'].map(acc_to_idx).values
    dst = _df['nameDest'].map(acc_to_idx).values
    edge_index = np.vstack([src, dst])

    sent_sum = np.bincount(src, weights=_df['amount'].values, minlength=num_nodes)
    recv_sum = np.bincount(dst, weights=_df['amount'].values, minlength=num_nodes)
    sent_cnt = np.bincount(src, minlength=num_nodes)
    recv_cnt = np.bincount(dst, minlength=num_nodes)
    node_feats = np.column_stack([
        sent_sum, recv_sum, sent_cnt, recv_cnt,
        np.divide(sent_sum, sent_cnt, where=sent_cnt>0, out=np.zeros_like(sent_sum)),
        np.divide(recv_sum, recv_cnt, where=recv_cnt>0, out=np.zeros_like(recv_sum))
    ])
    node_feats = StandardScaler().fit_transform(node_feats)

    edge_numeric = StandardScaler().fit_transform(_df[['amount', 'errOrig', 'errDest', 'step']].values)
    edge_type_idx = _df['type_enc'].values
    y = _df['isFraud'].values

    return node_feats, edge_index, edge_numeric, edge_type_idx, y, num_nodes

node_feats, edge_index, edge_numeric, edge_type_idx, y, num_nodes = build_graph(df_bal)

train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
train_mask = np.zeros(len(y), dtype=bool); train_mask[train_idx] = True
test_mask = np.zeros(len(y), dtype=bool); test_mask[test_idx] = True

data = Data(
    x=torch.tensor(node_feats, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_numeric, dtype=torch.float),
    y=torch.tensor(y, dtype=torch.long),
    type_idx=torch.tensor(edge_type_idx, dtype=torch.long),
    train_mask=torch.tensor(train_mask),
    test_mask=torch.tensor(test_mask)
).to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 8. GNN Model
# -------------------------------------------------
class ImprovedEdgeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, n_types, dropout=0.3):
        super().__init__()
        self.type_emb = torch.nn.Embedding(n_types, 8)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        mlp_in = hidden_channels*2 + edge_attr_dim + 8
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels//2), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels//2, 2)
        )
    def forward(self, x, edge_index, edge_attr, type_idx):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr, self.type_emb(type_idx)], dim=1)
        return self.edge_mlp(edge_input)

# -------------------------------------------------
# 9. Train GNN
# -------------------------------------------------
@st.cache_resource
def train_gnn(_data, _hidden_dim, _epochs):
    model = ImprovedEdgeGNN(
        in_channels=_data.x.size(1),
        hidden_channels=_hidden_dim,
        edge_attr_dim=edge_numeric.shape[1],
        n_types=len(le_type.classes_)
    ).to(_data.x.device)

    train_labels = _data.y[_data.train_mask].cpu().numpy()
    classes = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(_data.x.device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    prog = st.progress(0)
    stat = st.empty()
    for epoch in range(_epochs):
        optimizer.zero_grad()
        out = model(_data.x, _data.edge_index, _data.edge_attr, _data.type_idx)
        loss = criterion(out[_data.train_mask], _data.y[_data.train_mask])
        loss.backward()
        optimizer.step()
        prog.progress((epoch+1)/_epochs)
        stat.text(f"Epoch {epoch+1}/{_epochs} – Loss: {loss.item():.4f}")
    prog.empty()
    stat.empty()
    return model

with st.spinner("Training GraphSAGE..."):
    gnn_model = train_gnn(data, HIDDEN_DIM, EPOCHS)

# -------------------------------------------------
# 10. XGBoost
# -------------------------------------------------
@st.cache_resource
def train_xgb():
    X_train = np.hstack([
        edge_numeric[train_idx],
        data.x[data.edge_index[0][train_idx]].cpu().numpy(),
        data.x[data.edge_index[1][train_idx]].cpu().numpy()
    ])
    X_test = np.hstack([
        edge_numeric[test_idx],
        data.x[data.edge_index[0][test_idx]].cpu().numpy(),
        data.x[data.edge_index[1][test_idx]].cpu().numpy()
    ])
    pos_weight = len(y[train_idx][y[train_idx]==0]) / max(1, len(y[train_idx][y[train_idx]==1]))
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=pos_weight, eval_metric='logloss')
    model.fit(X_train, y[train_idx])
    return model, X_test

with st.spinner("Training XGBoost..."):
    xgb_model, X_test = train_xgb()

# -------------------------------------------------
# 11. Predictions
# -------------------------------------------------
with torch.no_grad():
    gnn_out = gnn_model(data.x, data.edge_index, data.edge_attr, data.type_idx)
    gnn_probs = F.softmax(gnn_out, dim=1)[:,1].cpu().numpy()
    gnn_preds = (gnn_probs > 0.5).astype(int)

xgb_probs = xgb_model.predict_proba(X_test)[:,1]
xgb_preds = (xgb_probs > 0.5).astype(int)
labels_test = y[test_idx]

# -------------------------------------------------
# 12. Safe Metrics
# -------------------------------------------------
def metrics(probs, preds, labels):
    roc_auc = roc_auc_score(labels, probs)
    try:
        p, r, _ = precision_recall_curve(labels, probs + 1e-12)
        idx = np.argsort(r)
        r, p = r[idx], p[idx]
        pr_auc = 0.0 if r.size == 0 or r[-1] == 0 else auc(r, p)
    except Exception:
        pr_auc = 0.0
    return {
        "ROC-AUC"   : roc_auc,
        "PR-AUC"    : pr_auc,
        "F1"        : f1_score(labels, preds, zero_division=0),
        "Precision" : precision_score(labels, preds, zero_division=0),
        "Recall"    : recall_score(labels, preds, zero_division=0)
    }

gnn_metrics = metrics(gnn_probs[test_idx], gnn_preds[test_idx], labels_test)
xgb_metrics = metrics(xgb_probs, xgb_preds, labels_test)

# -------------------------------------------------
# 13. UI – Dashboard
# -------------------------------------------------
st.markdown("## Model Performance")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### GraphSAGE GNN")
    for k, v in gnn_metrics.items():
        st.markdown(f"<div class='metric-card'><h4>{k}</h4><h2 style='color:#818cf8'>{v:.4f}</h2></div>", unsafe_allow_html=True)
with c2:
    st.markdown("### XGBoost")
    for k, v in xgb_metrics.items():
        st.markdown(f"<div class='metric-card'><h4>{k}</h4><h2 style='color:#34d399'>{v:.4f}</h2></div>", unsafe_allow_html=True)

# -------------------------------------------------
# 14. ROC / PR Curves
# -------------------------------------------------
fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall"))
fpr_g, tpr_g, _ = roc_curve(labels_test, gnn_probs[test_idx])
fpr_x, tpr_x, _ = roc_curve(labels_test, xgb_probs)
prec_g, rec_g, _ = precision_recall_curve(labels_test, gnn_probs[test_idx])
prec_x, rec_x, _ = precision_recall_curve(labels_test, xgb_probs)

fig.add_trace(go.Scatter(x=fpr_g, y=tpr_g, name=f"GNN ({gnn_metrics['ROC-AUC']:.3f})", line=dict(color="#818cf8")), row=1, col=1)
fig.add_trace(go.Scatter(x=fpr_x, y=tpr_x, name=f"XGB ({xgb_metrics['ROC-AUC']:.3f})", line=dict(color="#34d399")), row=1, col=1)
fig.add_trace(go.Scatter(x=rec_g, y=prec_g, name=f"GNN ({gnn_metrics['PR-AUC']:.3f})", line=dict(color="#818cf8")), row=1, col=2)
fig.add_trace(go.Scatter(x=rec_x, y=prec_x, name=f"XGB ({xgb_metrics['PR-AUC']:.3f})", line=dict(color="#34d399")), row=1, col=2)
fig.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# -------------------------------------------------
# 15. EXPLAINABILITY + ABOUT + PREDICT
# -------------------------------------------------
st.markdown("## Analysis & Prediction", unsafe_allow_html=True)

# Compute edge_inputs once
with st.spinner("Preparing edge features..."):
    with torch.no_grad():
        h = gnn_model.conv2(
            gnn_model.bn1(F.relu(gnn_model.conv1(data.x, data.edge_index))),
            data.edge_index
        )
        src_h = h[data.edge_index[0]]
        dst_h = h[data.edge_index[1]]
        type_emb = gnn_model.type_emb(data.type_idx)

    edge_inputs = np.hstack([
        src_h.cpu().numpy(),
        dst_h.cpu().numpy(),
        edge_numeric,
        type_emb.cpu().numpy()
    ])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["SHAP Explainer", "LIME Inspector", "About", "Predict"])

# =============================================
# TAB 1: SHAP
# =============================================
with tab1:
    st.markdown("### Interactive SHAP Force Plot")
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.spinner("Computing SHAP values..."):
            def shap_predict(x):
                with torch.no_grad():
                    x_t = torch.tensor(x, dtype=torch.float, device=data.x.device)
                    return F.softmax(gnn_model.edge_mlp(x_t), dim=1).cpu().numpy()[:, 1]

            bg = edge_inputs[train_idx[:50]]
            test_edge = edge_inputs[test_idx[0]:test_idx[0]+1]

            explainer = shap.KernelExplainer(shap_predict, bg)
            shap_vals = explainer.shap_values(test_edge, nsamples=100)

            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_vals,
                test_edge,
                show=False,
                matplotlib=False
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            st.components.v1.html(shap_html, height=450, scrolling=False)

    with col2:
        st.markdown("### Waterfall")
        fig, ax = plt.subplots(figsize=(6, 5))
        shap.waterfall_plot(
            shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value, data=test_edge[0]),
            show=False, max_display=10
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1e293b')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()

    st.markdown("### Global SHAP")
    with st.spinner("Computing global SHAP..."):
        shap_vals_all = explainer.shap_values(edge_inputs[test_idx[:30]], nsamples=100)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals_all, edge_inputs[test_idx[:30]], plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        st.pyplot(fig)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("Force Plot (HTML)", data=shap_html, file_name="shap_force.html", mime="text/html")
    with col_d2:
        buf.seek(0)
        st.download_button("Waterfall (PNG)", data=buf.read(), file_name="shap_waterfall.png", mime="image/png")

# =============================================
# TAB 2: LIME (White Text)
# =============================================
with tab2:
    st.markdown("### LIME Local Explanation")
    edge_to_explain = st.selectbox("Select Edge", test_idx[:20], format_func=lambda x: f"Edge {x}")
    sample_feat = edge_inputs[edge_to_explain:edge_to_explain+1]

    lime_expl = LimeTabularExplainer(
        edge_inputs[train_idx],
        mode="classification",
        feature_names=[f"feat_{i}" for i in range(edge_inputs.shape[1])],
        class_names=["Non-Fraud", "Fraud"]
    )

    def lime_predict(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float, device=data.x.device)
            return F.softmax(gnn_model.edge_mlp(x_t), dim=1).cpu().numpy()

    with st.spinner("Generating LIME..."):
        exp = lime_expl.explain_instance(sample_feat[0], lime_predict, num_features=10)

    lime_html = exp.as_html()
    dark_lime = lime_html.replace(
        "</head>",
        "<style>body{background:#0f172a;color:#ffffff !important;font-family:'Segoe UI'} "
        ".lime table{color:#ffffff !important;border-color:#6366f1}</style></head>"
    )
    st.components.v1.html(dark_lime, height=700, scrolling=True)

    probs = lime_predict(sample_feat)[0]
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Non-Fraud", f"{probs[0]:.1%}")
    with col_p2:
        st.metric("Fraud Risk", f"{probs[1]:.1%}")

    st.download_button("Download LIME", data=dark_lime, file_name=f"lime_edge_{edge_to_explain}.html", mime="text/html")

# =============================================
# TAB 3: ABOUT – Full Transaction Type Histogram
# =============================================
with tab3:
    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Transactions:** {len(df_bal):,}")
    with col2:
        st.write(f"**Fraud Rate:** {df_bal['isFraud'].mean():.2%}")

    st.markdown("### Transaction Type Distribution")
    type_counts = df_bal['type'].value_counts().sort_values(ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=type_counts.index,
        y=type_counts.values,
        text=type_counts.values,
        textposition='outside',
        marker_color='#6366f1',
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
    )])
    fig.update_layout(
        title="All Transaction Types",
        xaxis_title="Type",
        yaxis_title="Count",
        template="plotly_dark",
        height=500,
        margin=dict(t=80)
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# TAB 4: PREDICT – Real Fields (No Errors)
# =============================================
with tab4:
    st.markdown("### Live Fraud Prediction")
    st.info("Enter real transaction details → Click **Predict**")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            step = st.number_input("Step (Hour)", min_value=1, max_value=743, value=1, key="step")
            amount = st.number_input("Amount", min_value=0.0, value=9839.64, step=100.0, key="amt")
            nameOrig = st.text_input("Origin Account (nameOrig)", value="C1231006815", key="orig")
            oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=170136.0, step=100.0, key="oldorg")
            newbalanceOrig = st.number_input("Origin New Balance", min_value=0.0, value=160296.36, step=100.0, key="neworg")

        with col2:
            type_str = st.selectbox(
                "Transaction Type",
                options=le_type.classes_.tolist(),
                index=0,
                key="type"
            )
            nameDest = st.text_input("Destination Account (nameDest)", value="M1979787155", key="dest")
            oldbalanceDest = st.number_input("Dest Old Balance", min_value=0.0, value=0.0, step=100.0, key="olddest")
            newbalanceDest = st.number_input("Dest New Balance", min_value=0.0, value=0.0, step=100.0, key="newdest")

        submitted = st.form_submit_button("Predict Fraud Risk", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Predicting..."):
            # --- 1. Compute errors ---
            errOrig = newbalanceOrig + amount - oldbalanceOrg
            errDest = oldbalanceDest + amount - newbalanceDest

            # --- 2. Get node indices (or use mean if missing) ---
            acc_to_idx = {acc: i for i, acc in enumerate(pd.concat([df_bal['nameOrig'], df_bal['nameDest']]).unique())}
            src_idx = acc_to_idx.get(nameOrig, -1)
            dst_idx = acc_to_idx.get(nameDest, -1)

            with torch.no_grad():
                # Forward pass to get node embeddings
                h = gnn_model.conv2(
                    gnn_model.bn1(F.relu(gnn_model.conv1(data.x, data.edge_index))),
                    data.edge_index
                )

                # Use real embeddings if accounts exist, else mean
                if src_idx != -1:
                    src_emb = h[src_idx]
                else:
                    src_emb = h.mean(dim=0)

                if dst_idx != -1:
                    dst_emb = h[dst_idx]
                else:
                    dst_emb = h.mean(dim=0)

            # --- 3. Scale edge numeric ---
            edge_raw = np.array([[amount, errOrig, errDest, step]])
            scaler = StandardScaler().fit(edge_numeric)
            edge_scaled = scaler.transform(edge_raw)[0]

            # --- 4. Type embedding ---
            type_idx = np.where(le_type.classes_ == type_str)[0][0]
            type_emb_vec = gnn_model.type_emb.weight[type_idx].detach().cpu().numpy()

            # --- 5. Final input ---
            input_vec = np.concatenate([
                src_emb.cpu().numpy(),
                dst_emb.cpu().numpy(),
                edge_scaled,
                type_emb_vec
            ]).reshape(1, -1)

            # --- 6. Predict ---
            with torch.no_grad():
                logits = gnn_model.edge_mlp(
                    torch.tensor(input_vec, dtype=torch.float, device=data.x.device)
                )
                fraud_prob = F.softmax(logits, dim=1)[0, 1].item()

        # --- Results ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraud Probability", f"{fraud_prob:.1%}")
        with col2:
            status = "FRAUD" if fraud_prob > 0.5 else "SAFE"
            color = "red" if fraud_prob > 0.5 else "green"
            st.markdown(f"<h2 style='color:{color}; text-align:center; margin:0'>{status}</h2>", unsafe_allow_html=True)

        with st.expander("Input Summary"):
            st.json({
                "step": step,
                "type": type_str,
                "amount": amount,
                "nameOrig": nameOrig,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "nameDest": nameDest,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "errOrig": round(errOrig, 2),
                "errDest": round(errDest, 2)
            })


            
# =============================================
# FINAL METRICS
# =============================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### Final Metrics")
    for name, mets in [("GNN", gnn_metrics), ("XGBoost", xgb_metrics)]:
        st.markdown(f"**{name}**")
        for k, v in mets.items():
            color = "#34d399" if name == "XGBoost" else "#818cf8"
            st.markdown(f"<small style='color:{color}'>{k}: <b>{v:.4f}</b></small>", unsafe_allow_html=True)
        st.markdown("")

#st.success("FraudShield – Ready for Production")