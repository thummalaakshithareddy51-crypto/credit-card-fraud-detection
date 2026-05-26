import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detector", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Rajdhani', sans-serif;
}
.main { background-color: #0a0a0f; }
h1 {
    font-family: 'Share Tech Mono', monospace;
    color: #00ff88;
    text-align: center;
    font-size: 2.2rem;
    letter-spacing: 3px;
    text-shadow: 0 0 20px #00ff8855;
}
.subtitle {
    text-align: center;
    color: #666;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    margin-bottom: 2rem;
}
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc66);
    color: #0a0a0f;
    font-family: 'Share Tech Mono', monospace;
    font-weight: bold;
    letter-spacing: 2px;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    width: 100%;
    text-transform: uppercase;
}
.result-safe {
    background: #0a1f14;
    border: 1px solid #00ff88;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px #00ff8822;
}
.result-fraud {
    background: #1f0a0a;
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px #ff444422;
}
.result-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    letter-spacing: 3px;
}
.result-safe .result-title { color: #00ff88; }
.result-fraud .result-title { color: #ff4444; }
.result-desc { color: #888; font-size: 0.9rem; font-family: 'Share Tech Mono', monospace; }
.score-box {
    background: #111118;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
    margin-top: 1rem;
}
.score-label { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #555; letter-spacing: 2px; }
.score-value { font-family: 'Share Tech Mono', monospace; font-size: 1.4rem; color: #00ff88; }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    return model, pca

@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

model, pca = load_model()
df = load_data()

# Scale same way as training
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

st.markdown("<h1>🛡️ FRAUD DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>// REAL-TIME TRANSACTION ANALYSIS //</p>", unsafe_allow_html=True)

# ---------- Load Sample Buttons ----------
st.markdown("### Load a sample transaction")
col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Load Normal Transaction"):
        sample = df[df['Class'] == 0].sample(1)
        st.session_state['sample'] = sample

with col2:
    if st.button("🔴 Load Fraud Transaction"):
        sample = df[df['Class'] == 1].sample(1)
        st.session_state['sample'] = sample

# ---------- Show transaction and predict ----------
if 'sample' in st.session_state:
    sample = st.session_state['sample']
    
    st.markdown("---")
    st.markdown("### Transaction Details")
    
    # Show amount and time nicely
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Amount (scaled)", f"{sample['Amount'].values[0]:.4f}")
    with col2:
        st.metric("Time (scaled)", f"{sample['Time'].values[0]:.4f}")

    # Show V values
    with st.expander("View V1 - V28 features"):
        v_cols = [f'V{i}' for i in range(1, 29)]
        st.dataframe(sample[v_cols].T.rename(columns={sample.index[0]: 'Value'}))

    # Predict
    X_input = sample.drop('Class', axis=1)
    X_reduced = pca.transform(X_input)
    prediction = model.predict(X_reduced)[0]
    score = model.decision_function(X_reduced)[0]

    st.markdown("---")

    if prediction == 1:
        st.markdown("""
        <div class='result-safe'>
            <div style='font-size:3rem'>✅</div>
            <div class='result-title'>TRANSACTION NORMAL</div>
            <div class='result-desc'>No anomalous patterns detected.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-fraud'>
            <div style='font-size:3rem'>🚨</div>
            <div class='result-title'>SUSPICIOUS ACTIVITY</div>
            <div class='result-desc'>Anomalous patterns detected. Flagged for review.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='score-box'>
        <div class='score-label'>ANOMALY SCORE</div>
        <div class='score-value'>{score:.4f}</div>
        <div class='score-label'>more negative = more suspicious | threshold = 0.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Show actual label
    actual = "🔴 FRAUD" if sample['Class'].values[0] == 1 else "🟢 NORMAL"
    st.markdown(f"<br><center style='font-family:monospace; color:#444; font-size:0.8rem'>ACTUAL LABEL: {actual}</center>", unsafe_allow_html=True)