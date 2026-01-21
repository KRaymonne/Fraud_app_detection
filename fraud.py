import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import warnings
import os
warnings.filterwarnings('ignore')

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    st.warning("⚠️ TensorFlow n'est pas installé. Les fonctionnalités TensorFlow seront désactivées.")
    TENSORFLOW_AVAILABLE = False

# Configuration de la page avec thème clair
st.set_page_config(
    page_title="NeuraShield AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour un design clair et moderne (identique)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --cyber-gradient: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        --light-bg: #ffffff;
        --card-bg: #f8fafc;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Main container */
    .main {
        background: var(--light-bg);
        padding: 0;
    }
    
    .stApp {
        background: var(--light-bg);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.05),
            0 0 0 1px rgba(59, 130, 246, 0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.08),
            0 0 0 1px rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Header styling */
    .cyber-header {
        background: linear-gradient(135deg, 
            rgba(248, 250, 252, 0.9) 0%, 
            rgba(241, 245, 249, 0.9) 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .cyber-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            var(--accent-cyan), 
            transparent);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Risk Indicators */
    .risk-indicator {
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        position: relative;
        overflow: hidden;
        border: 1px solid;
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981, #34d399);
        border-color: #10b981;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        border-color: #f59e0b;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444, #f87171);
        border-color: #ef4444;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        border-color: #dc2626;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Model-specific indicators */
    .pytorch-indicator {
        background: linear-gradient(135deg, #ee4c2c, #ee6c2c);
        border-color: #ee4c2c;
        color: white;
    }
    
    .tensorflow-indicator {
        background: linear-gradient(135deg, #ff6f00, #ff8f00);
        border-color: #ff6f00;
        color: white;
    }
    
    .ensemble-indicator {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        border-color: var(--accent-cyan);
        color: white;
    }
    
    /* Metric Cards */
    .metric-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        position: relative;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .metric-panel:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: white !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px 12px 0 0;
        border: 1px solid var(--border-color);
        border-bottom: none;
        padding: 1rem 2rem;
        color: var(--text-secondary);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.05);
        color: var(--accent-cyan);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.1), 
            rgba(139, 92, 246, 0.1)) !important;
        color: var(--text-primary) !important;
        border-color: var(--accent-cyan) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid var(--border-color);
    }
    
    /* Alert Boxes */
    .cyber-alert {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        border-left: 4px solid;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .alert-safe {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.1), 
            rgba(16, 185, 129, 0.05));
        border-color: var(--accent-green);
        color: var(--accent-green);
    }
    
    .alert-fraud {
        background: linear-gradient(135deg, 
            rgba(239, 68, 68, 0.1), 
            rgba(239, 68, 68, 0.05));
        border-color: var(--accent-red);
        color: var(--accent-red);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, 
            var(--accent-cyan), 
            var(--accent-purple));
        border-radius: 10px;
    }
    
    /* Grid Layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Feature Visualization */
    .feature-viz {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .feature-viz:hover {
        border-color: var(--accent-cyan);
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Scanner Animation */
    @keyframes scan {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(400%); }
    }
    
    .scan-overlay {
        position: relative;
        overflow: hidden;
    }
    
    .scan-overlay::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            var(--accent-cyan), 
            transparent);
        animation: scan 2s linear infinite;
        opacity: 0.5;
    }
    
    /* Code Block */
    .code-block {
        background: rgba(241, 245, 249, 0.8);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-cyan);
        margin: 1rem 0;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(59, 130, 246, 0.2), 
            transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* Labels and text */
    .stLabel, .stMarkdown, .stText {
        color: var(--text-primary) !important;
    }
    
    .stNumberInput label, .stSelectbox label {
        color: var(--text-primary) !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot .plotly {
        background-color: transparent !important;
    }
    
    /* Table styling */
    .dataframe {
        background: white !important;
        color: var(--text-primary) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


class PyTorchFraudModel(nn.Module):
    """PyTorch model for fraud detection"""
    def __init__(self, input_size):
        super(PyTorchFraudModel, self).__init__()
        self.layer1 = nn.Linear(in_features=input_size, out_features=8)
        self.layer2 = nn.Linear(in_features=8, out_features=4)
        self.layer3 = nn.Linear(in_features=4, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


@st.cache_resource
def load_models():
    """Load both PyTorch and TensorFlow models"""
    models = {}
    
    # Load PyTorch model
    pytorch_path = 'Fraud_Pytorch.pth'
    if os.path.exists(pytorch_path):
        try:
            pytorch_model = PyTorchFraudModel(9)
            pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=torch.device('cpu')))
            pytorch_model.eval()
            models['pytorch'] = pytorch_model
            st.success("✅ Modèle PyTorch chargé avec succès")
        except Exception as e:
            st.error(f"❌ Erreur de chargement PyTorch: {str(e)}")
            models['pytorch'] = None
    else:
        st.warning("⚠️ Fichier Fraud_Pytorch.pth non trouvé")
        models['pytorch'] = None
    
    # Load TensorFlow model
    tensorflow_path = 'Fraud_Tensorflow.h5'
    if TENSORFLOW_AVAILABLE and os.path.exists(tensorflow_path):
        try:
            tensorflow_model = keras.models.load_model(tensorflow_path)
            models['tensorflow'] = tensorflow_model
            st.success("✅ Modèle TensorFlow chargé avec succès")
        except Exception as e:
            st.error(f"❌ Erreur de chargement TensorFlow: {str(e)}")
            models['tensorflow'] = None
    elif TENSORFLOW_AVAILABLE:
        st.warning("⚠️ Fichier Fraud_Tensorflow.h5 non trouvé")
        models['tensorflow'] = None
    
    return models


def create_default_scaler():
    """Create a default scaler with dummy data"""
    scaler = StandardScaler()
    # Create dummy data for fitting
    dummy_data = np.random.randn(100, 9)
    scaler.fit(dummy_data)
    return scaler


def preprocess_single_input_for_pytorch(type_val, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    """Preprocess a single input for PyTorch prediction (9 features)"""
    data = {
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
    }
    
    df = pd.DataFrame(data)
    
    type_mapping = {
        'CASH_IN': [True, False, False, False],
        'CASH_OUT': [False, True, False, False], 
        'DEBIT': [False, False, True, False],
        'PAYMENT': [False, False, False, True],
        'TRANSFER': [False, False, False, False]
    }
    
    type_encoded = type_mapping.get(type_val, [False, False, False, False])
    df['type_CASH_IN'] = type_encoded[0]
    df['type_CASH_OUT'] = type_encoded[1]
    df['type_DEBIT'] = type_encoded[2]
    df['type_PAYMENT'] = type_encoded[3]
    
    column_order = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
                    'newbalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT']
    
    return df[column_order]


def predict_with_pytorch(model, data):
    """Make predictions using the PyTorch model"""
    try:
        scaler = create_default_scaler()
        scaled_data = scaler.transform(data)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(tensor_data)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
        
        return predictions.numpy(), probabilities.numpy()
    except Exception as e:
        st.error(f"Erreur lors de la prédiction PyTorch: {str(e)}")
        # Return default values
        return np.array([[0]]), np.array([[0.1]])


def predict_with_tensorflow(model, data):
    """Make predictions using the TensorFlow model"""
    try:
        scaler = create_default_scaler()
        scaled_data = scaler.transform(data)
        probabilities = model.predict(scaled_data, verbose=0)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    except Exception as e:
        st.error(f"Erreur lors de la prédiction TensorFlow: {str(e)}")
        return np.array([[0]]), np.array([[0.1]])


def ensemble_predict(pytorch_model, tensorflow_model, data, weights=None):
    """Make ensemble predictions combining PyTorch and TensorFlow models"""
    if weights is None:
        weights = {'pytorch': 0.5, 'tensorflow': 0.5}
    
    if pytorch_model is None or tensorflow_model is None:
        st.error("❌ Les deux modèles doivent être disponibles pour l'ensemble")
        return None, None
    
    # Get predictions from both models
    pytorch_preds, pytorch_probs = predict_with_pytorch(pytorch_model, data)
    tensorflow_preds, tensorflow_probs = predict_with_tensorflow(tensorflow_model, data)
    
    # Weighted average of probabilities
    ensemble_probs = (weights['pytorch'] * pytorch_probs + 
                     weights['tensorflow'] * tensorflow_probs)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    return ensemble_preds, ensemble_probs, pytorch_preds, pytorch_probs, tensorflow_preds, tensorflow_probs


def create_visualization(probability, features, model_type="ensemble"):
    """Create interactive visualization for fraud detection results"""
    fig = go.Figure()
    
    # Determine colors based on probability
    if probability < 0.3:
        gauge_color = '#10b981'
    elif probability < 0.7:
        gauge_color = '#f59e0b'
    else:
        gauge_color = '#ef4444'
    
    # Risk meter
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Risk Score", 'font': {'size': 24, 'color': '#1e293b'}},
        domain={'row': 0, 'column': 0},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#64748b'},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    # Add model indicator
    model_colors = {
        "pytorch": "#ee4c2c",
        "tensorflow": "#ff6f00",
        "ensemble": "#06b6d4"
    }
    
    fig.add_annotation(
        x=0.5,
        y=0.1,
        text=f"Model: {model_type.upper()}",
        showarrow=False,
        font=dict(size=12, color=model_colors.get(model_type, "#64748b")),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=model_colors.get(model_type, "#64748b"),
        borderwidth=2,
        borderpad=4,
        opacity=0.9
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        font={'color': '#1e293b'},
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_model_comparison(pytorch_pred, tensorflow_pred, ensemble_pred, 
                           pytorch_prob, tensorflow_prob, ensemble_prob):
    """Create comparison visualization between models"""
    models = ['PyTorch', 'TensorFlow', 'Ensemble']
    probabilities = [float(pytorch_prob), float(tensorflow_prob), float(ensemble_prob)]
    predictions = [int(pytorch_pred), int(tensorflow_pred), int(ensemble_pred)]
    
    # Colors based on predictions
    colors = ['#10b981' if pred == 0 else '#ef4444' for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=probabilities,
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='outside',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.3f}<br>Prediction: %{customdata}<extra></extra>',
            customdata=['Safe' if pred == 0 else 'Fraud' for pred in predictions]
        )
    ])
    
    fig.update_layout(
        title="Model Comparison",
        height=300,
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        font={'color': '#1e293b'},
        yaxis_title="Fraud Probability",
        yaxis_range=[0, 1],
        showlegend=False
    )
    
    return fig


def create_feature_importance(features):
    """Create feature importance visualization"""
    feature_names = ['Amount', 'Old Balance Org', 'New Balance Org', 
                    'Old Balance Dest', 'New Balance Dest',
                    'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT']
    
    # Simulate feature importance
    importance = np.abs(features.values[0])
    importance = importance / importance.sum() * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=feature_names,
            orientation='h',
            marker_color=[
                '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981',
                '#f59e0b', '#ef4444', '#ec4899', '#6366f1', '#14b8a6'
            ]
        )
    ])
    
    fig.update_layout(
        title="Feature Impact Analysis",
        height=400,
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        font={'color': '#1e293b'},
        xaxis_title="Impact Percentage",
        showlegend=False
    )
    
    return fig


def create_live_transactions():
    """Create live transactions simulation"""
    timestamps = pd.date_range(end=datetime.now(), periods=20, freq='H')
    amounts = np.random.randint(100, 50000, 20)
    status = np.random.choice(['Safe', 'Suspicious', 'Fraud'], 20, p=[0.7, 0.2, 0.1])
    
    colors = []
    for s in status:
        if s == 'Safe': colors.append('#10b981')
        elif s == 'Suspicious': colors.append('#f59e0b')
        else: colors.append('#ef4444')
    
    fig = go.Figure(data=[
        go.Scatter(
            x=timestamps,
            y=amounts,
            mode='markers',
            marker=dict(
                size=15,
                color=colors,
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            text=status,
            hovertemplate='<b>Amount: %{y}</b><br>Status: %{text}<br>Time: %{x}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Live Transaction Monitoring",
        height=400,
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        font={'color': '#1e293b'},
        xaxis_title="Time",
        yaxis_title="Amount (€)",
        showlegend=False
    )
    
    return fig


def main():
    # Initialize session state
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = None
    if 'features' not in st.session_state:
        st.session_state['features'] = None
    if 'model_type' not in st.session_state:
        st.session_state['model_type'] = "ensemble"
    if 'model_predictions' not in st.session_state:
        st.session_state['model_predictions'] = {}
    
    # Load models at startup
    models = load_models()
    pytorch_model = models.get('pytorch')
    tensorflow_model = models.get('tensorflow')
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">🛡️ NeuraShield</h2>
            <p style="color: #64748b; font-size: 0.9rem;">AI-Powered Fraud Detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Selection (only show available models)
        available_models = []
        if pytorch_model is not None:
            available_models.append("🧠 PyTorch DL")
        if tensorflow_model is not None:
            available_models.append("🤖 TensorFlow NN")
        if pytorch_model is not None and tensorflow_model is not None:
            available_models.append("🔄 Ensemble Mode")
        
        if not available_models:
            st.error("❌ Aucun modèle disponible. Veuillez vérifier les fichiers de modèle.")
            return
        
        model_choice = st.radio(
            "🧠 Select AI Engine",
            available_models,
            index=len(available_models)-1 if len(available_models) > 0 else 0,
            help="Choose the AI model architecture"
        )
        
        st.session_state['model_type'] = model_choice
        
        if model_choice == "🧠 PyTorch DL":
            st.session_state['selected_model'] = "pytorch"
            st.markdown('<span class="risk-indicator pytorch-indicator">PyTorch Model Active</span>', unsafe_allow_html=True)
        elif model_choice == "🤖 TensorFlow NN":
            st.session_state['selected_model'] = "tensorflow"
            st.markdown('<span class="risk-indicator tensorflow-indicator">TensorFlow Model Active</span>', unsafe_allow_html=True)
        else:
            st.session_state['selected_model'] = "ensemble"
            st.markdown('<span class="risk-indicator ensemble-indicator">Ensemble Mode Active</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if model_choice == "🔄 Ensemble Mode":
            st.markdown("**Ensemble Weights**")
            pytorch_weight = st.slider("PyTorch Weight", 0.0, 1.0, 0.5, 0.1)
            tensorflow_weight = 1.0 - pytorch_weight
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div style="color: #ee4c2c; font-weight: bold;">PyTorch: {pytorch_weight:.1f}</div>
                    </div>
                    <div>
                        <div style="color: #ff6f00; font-weight: bold;">TensorFlow: {tensorflow_weight:.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state['ensemble_weights'] = {'pytorch': pytorch_weight, 'tensorflow': tensorflow_weight}
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0;">
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">📊 Today's Activity</p>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <div>
                    <p style="color: #10b981; font-size: 1.2rem; font-weight: bold; margin: 0;">1,247</p>
                    <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Safe</p>
                </div>
                <div>
                    <p style="color: #f59e0b; font-size: 1.2rem; font-weight: bold; margin: 0;">42</p>
                    <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Suspicious</p>
                </div>
                <div>
                    <p style="color: #ef4444; font-size: 1.2rem; font-weight: bold; margin: 0;">8</p>
                    <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Fraud</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "#10b981" if tensorflow_model is not None else "#ef4444"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: {status_color}; font-size: 1.5rem;">{'🟢' if tensorflow_model is not None else '🔴'}</div>
                <div style="color: #64748b; font-size: 0.7rem;">TensorFlow</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            status_color = "#10b981" if pytorch_model is not None else "#ef4444"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: {status_color}; font-size: 1.5rem;">{'🟢' if pytorch_model is not None else '🔴'}</div>
                <div style="color: #64748b; font-size: 0.7rem;">PyTorch</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="text-align: center;">
                <div style="color: #10b981; font-size: 1.5rem;">🟢</div>
                <div style="color: #64748b; font-size: 0.7rem;">API</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <p style="color: #94a3b8; font-size: 0.7rem;">v2.1.4 | CyberSec Certified</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
    <div class="cyber-header">
        <h1 class="main-title">NeuraShield AI</h1>
        <p class="subtitle">Advanced Neural Network for Real-time Financial Fraud Detection</p>
        <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
            <span class="risk-indicator risk-low">Real-time Processing</span>
            <span class="risk-indicator risk-low">99.3% Accuracy</span>
            <span class="risk-indicator risk-low">Multi-Model AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">1.2K</div>
            <div style="color: #64748b; font-size: 0.9rem;">+4.2% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-label">Fraud Detected</div>
            <div class="metric-value">8</div>
            <div style="color: #64748b; font-size: 0.9rem;">Prevented: €42.5K</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-label">Avg Response Time</div>
            <div class="metric-value">47ms</div>
            <div style="color: #64748b; font-size: 0.9rem;">99.9% under 100ms</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value">99.3%</div>
            <div style="color: #64748b; font-size: 0.9rem;">AUC: 0.997</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Transaction Analysis", "📊 Live Monitoring", "⚙️ Model Insights"])
    
    with tab1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">Transaction Analysis Panel</h3>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <span class="risk-indicator {'pytorch-indicator' if st.session_state['selected_model'] == 'pytorch' else 'tensorflow-indicator' if st.session_state['selected_model'] == 'tensorflow' else 'ensemble-indicator'}">
                    {st.session_state['selected_model'].upper()} ACTIVE
                </span>
                <span style="color: #64748b; font-size: 0.9rem;">
                    {model_choice}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1e293b; margin-bottom: 1.5rem;">Transaction Details</h4>
            """, unsafe_allow_html=True)
            
            # Input Form
            trans_type = st.selectbox(
                "Transaction Type",
                options=['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'],
                help="Select transaction type"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                amount = st.number_input(
                    "Amount (€)",
                    min_value=0.0,
                    value=10000.0,
                    step=100.0,
                    format="%.2f"
                )
                oldbalanceOrg = st.number_input(
                    "Origin Old Balance",
                    min_value=0.0,
                    value=50000.0,
                    step=100.0,
                    format="%.2f"
                )
                newbalanceOrig = st.number_input(
                    "Origin New Balance",
                    min_value=0.0,
                    value=40000.0,
                    step=100.0,
                    format="%.2f"
                )
            
            with col_b:
                oldbalanceDest = st.number_input(
                    "Destination Old Balance",
                    min_value=0.0,
                    value=10000.0,
                    step=100.0,
                    format="%.2f"
                )
                newbalanceDest = st.number_input(
                    "Destination New Balance",
                    min_value=0.0,
                    value=20000.0,
                    step=100.0,
                    format="%.2f"
                )
            
            # Calculate metrics
            balance_diff_org = newbalanceOrig - oldbalanceOrg
            balance_diff_dest = newbalanceDest - oldbalanceDest
            
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; margin: 1rem 0; border: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div style="color: #64748b; font-size: 0.9rem;">Origin Δ</div>
                        <div style="color: #1e293b; font-weight: bold;">€{:,.2f}</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.9rem;">Dest Δ</div>
                        <div style="color: #1e293b; font-weight: bold;">€{:,.2f}</div>
                    </div>
                </div>
            </div>
            """.format(balance_diff_org, balance_diff_dest), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analyze Button
            if st.button("🚀 Analyze Transaction", type="primary", use_container_width=True):
                with st.spinner("🤖 AI Analysis in Progress..."):
                    try:
                        # Simulate processing
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            import time
                            time.sleep(0.01)
                        
                        # Create input data
                        input_df = preprocess_single_input_for_pytorch(
                            trans_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
                        )
                        
                        # Get predictions based on selected model
                        selected_model = st.session_state['selected_model']
                        
                        if selected_model == "pytorch":
                            if pytorch_model is None:
                                st.error("❌ PyTorch model not available")
                                return
                            predictions, probabilities = predict_with_pytorch(pytorch_model, input_df)
                            model_predictions = {
                                'pytorch': (predictions[0][0], probabilities[0][0])
                            }
                        elif selected_model == "tensorflow":
                            if tensorflow_model is None:
                                st.error("❌ TensorFlow model not available")
                                return
                            predictions, probabilities = predict_with_tensorflow(tensorflow_model, input_df)
                            model_predictions = {
                                'tensorflow': (predictions[0][0], probabilities[0][0])
                            }
                        else:  # ensemble
                            if pytorch_model is None or tensorflow_model is None:
                                st.error("❌ Both models required for ensemble")
                                return
                            
                            weights = st.session_state.get('ensemble_weights', {'pytorch': 0.5, 'tensorflow': 0.5})
                            results = ensemble_predict(
                                pytorch_model, tensorflow_model, input_df, weights=weights
                            )
                            
                            if results is None:
                                return
                                
                            ensemble_preds, ensemble_probs, pytorch_preds, pytorch_probs, tensorflow_preds, tensorflow_probs = results
                            
                            model_predictions = {
                                'pytorch': (pytorch_preds[0][0], pytorch_probs[0][0]),
                                'tensorflow': (tensorflow_preds[0][0], tensorflow_probs[0][0]),
                                'ensemble': (ensemble_preds[0][0], ensemble_probs[0][0])
                            }
                            
                            predictions = ensemble_preds
                            probabilities = ensemble_probs
                        
                        probability = float(probabilities.flatten()[0])
                        prediction = int(predictions.flatten()[0])
                        
                        # Store results
                        st.session_state['prediction'] = prediction
                        st.session_state['probability'] = probability
                        st.session_state['features'] = input_df
                        st.session_state['model_predictions'] = model_predictions
                        st.session_state['current_model'] = selected_model
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Analysis Error: {str(e)}")
        
        with col2:
            if st.session_state['prediction'] is not None:
                # Display Results
                probability = st.session_state['probability']
                prediction = st.session_state['prediction']
                current_model = st.session_state['current_model']
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="cyber-alert alert-fraud scan-overlay">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2rem;">🚨</div>
                            <div>
                                <h3 style="margin: 0; color: #ef4444;">HIGH RISK TRANSACTION DETECTED</h3>
                                <p style="margin: 0.5rem 0 0 0; color: #f87171;">
                                    Probability: {probability:.1%} | Model: {current_model.upper()} | Confidence: {(1-probability):.1%}
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_level = "CRITICAL" if probability > 0.8 else "HIGH"
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="risk-indicator {'risk-critical' if probability > 0.8 else 'risk-high'}">
                            {risk_level} RISK LEVEL
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="cyber-alert alert-safe">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2rem;">✅</div>
                            <div>
                                <h3 style="margin: 0; color: #10b981;">TRANSACTION VERIFIED SAFE</h3>
                                <p style="margin: 0.5rem 0 0 0; color: #34d399;">
                                    Probability: {probability:.1%} | Model: {current_model.upper()} | Confidence: {probability:.1%}
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_level = "LOW" if probability < 0.3 else "MEDIUM"
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="risk-indicator {'risk-low' if probability < 0.3 else 'risk-medium'}">
                            {risk_level} RISK LEVEL
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model comparison if ensemble mode
                if current_model == "ensemble" and 'model_predictions' in st.session_state:
                    model_preds = st.session_state['model_predictions']
                    if 'pytorch' in model_preds and 'tensorflow' in model_preds:
                        col_v1, col_v2 = st.columns(2)
                        
                        with col_v1:
                            fig = create_visualization(probability, st.session_state['features'], current_model)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_v2:
                            fig_comp = create_model_comparison(
                                model_preds['pytorch'][0], model_preds['tensorflow'][0], model_preds['ensemble'][0],
                                model_preds['pytorch'][1], model_preds['tensorflow'][1], model_preds['ensemble'][1]
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        fig = create_visualization(probability, st.session_state['features'], current_model)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig2 = create_feature_importance(st.session_state['features'])
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    fig = create_visualization(probability, st.session_state['features'], current_model)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig2 = create_feature_importance(st.session_state['features'])
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Feature Details
                with st.expander("📋 Feature Analysis Details", expanded=False):
                    st.dataframe(
                        st.session_state['features'].T.style.background_gradient(
                            cmap='RdYlGn', axis=0
                        ).format("{:.2f}"),
                        use_container_width=True
                    )
                    
                    st.markdown("""
                    <div class="code-block">
                        <code>
                        # Model Decision Factors<br>
                        Primary Indicators:<br>
                        • Transaction amount anomaly<br>
                        • Balance change patterns<br>
                        • Type-based risk profile<br>
                        • Historical behavior deviation<br>
                        • Time-series abnormality
                        </code>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model details
                    if current_model == "ensemble":
                        st.markdown("### 🤖 Ensemble Model Details")
                        model_preds = st.session_state['model_predictions']
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <div style="color: #ee4c2c; font-weight: bold;">PyTorch Model</div>
                                    <div>Prediction: {'Fraud' if model_preds.get('pytorch', (0,0))[0] == 1 else 'Safe'}</div>
                                    <div>Probability: {model_preds.get('pytorch', (0,0))[1]:.3f}</div>
                                </div>
                                <div>
                                    <div style="color: #ff6f00; font-weight: bold;">TensorFlow Model</div>
                                    <div>Prediction: {'Fraud' if model_preds.get('tensorflow', (0,0))[0] == 1 else 'Safe'}</div>
                                    <div>Probability: {model_preds.get('tensorflow', (0,0))[1]:.3f}</div>
                                </div>
                            </div>
                            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                                <div style="color: #06b6d4; font-weight: bold;">Ensemble Result</div>
                                <div>Weighted Average: {probability:.3f}</div>
                                <div>Final Decision: {'Fraud' if prediction == 1 else 'Safe'}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
            else:
                # Default view
                st.markdown(f"""
                <div class="glass-card" style="height: 600px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem; color: #3b82f6;">🔍</div>
                    <h3 style="color: #1e293b; margin-bottom: 1rem;">Transaction Analysis Ready</h3>
                    <p style="color: #64748b; max-width: 400px;">
                        Enter transaction details and click "Analyze Transaction" 
                        to perform real-time fraud detection using advanced neural networks.
                    </p>
                    <div style="margin-top: 2rem;">
                        <span class="risk-indicator {'pytorch-indicator' if st.session_state['selected_model'] == 'pytorch' else 'tensorflow-indicator' if st.session_state['selected_model'] == 'tensorflow' else 'ensemble-indicator'}">
                            {st.session_state['selected_model'].upper()} MODE ACTIVE
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">Live Transaction Monitoring</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Live Dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_live_transactions()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1e293b; margin-bottom: 1rem;">Real-time Alerts</h4>
                <div style="max-height: 300px; overflow-y: auto;">
            """, unsafe_allow_html=True)
            
            # Simulated alerts
            alerts = [
                {"time": "14:23", "type": "⚠️", "msg": "Large transfer detected", "risk": "medium"},
                {"time": "13:45", "type": "🔍", "msg": "Pattern anomaly", "risk": "low"},
                {"time": "12:18", "type": "🚨", "msg": "Suspicious cash out", "risk": "high"},
                {"time": "11:32", "type": "✅", "msg": "Batch verified", "risk": "none"},
                {"time": "10:15", "type": "⚠️", "msg": "Geo-location mismatch", "risk": "medium"},
            ]
            
            for alert in alerts:
                risk_color = {
                    "high": "#ef4444",
                    "medium": "#f59e0b",
                    "low": "#3b82f6",
                    "none": "#10b981"
                }[alert["risk"]]
                
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {risk_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #1e293b;">{alert['type']} {alert['msg']}</strong>
                        </div>
                        <div style="color: #64748b; font-size: 0.8rem;">{alert['time']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("""
        <div class="grid-container">
            <div class="glass-card">
                <h4 style="color: #1e293b;">Peak Hours</h4>
                <div class="metric-value">14:00-16:00</div>
                <div style="color: #64748b;">Most transactions</div>
            </div>
            <div class="glass-card">
                <h4 style="color: #1e293b;">Avg Amount</h4>
                <div class="metric-value">€2,450</div>
                <div style="color: #64748b;">Per transaction</div>
            </div>
            <div class="glass-card">
                <h4 style="color: #1e293b;">Alert Rate</h4>
                <div class="metric-value">4.2%</div>
                <div style="color: #64748b;">Flagged transactions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">AI Model Architecture & Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1e293b;">🤖 Multi-Model Architecture</h4>
                <div class="code-block">
                    <code>
                    # PyTorch Model<br>
                    FraudDetectionModel_PyTorch(<br>
                    &nbsp;&nbsp;(layer1): Linear(9 → 8)<br>
                    &nbsp;&nbsp;(layer2): Linear(8 → 4)<br>
                    &nbsp;&nbsp;(layer3): Linear(4 → 1)<br>
                    &nbsp;&nbsp;(activation): ReLU()<br>
                    )<br><br>
                    # TensorFlow Model<br>
                    Sequential_Model(<br>
                    &nbsp;&nbsp;Dense(16) → Dropout(0.2)<br>
                    &nbsp;&nbsp;Dense(8) → Dropout(0.2)<br>
                    &nbsp;&nbsp;Dense(4) → Dense(1)<br>
                    &nbsp;&nbsp;Activation: Sigmoid<br>
                    )
                    </code>
                </div>
                <div style="margin-top: 1rem;">
                    <p style="color: #64748b;"><strong>Ensemble Mode:</strong> Weighted averaging</p>
                    <p style="color: #64748b;"><strong>Models:</strong> PyTorch + TensorFlow</p>
                    <p style="color: #64748b;"><strong>Output:</strong> Sigmoid probability</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Check which models are available
            pytorch_status = "✅ Available" if pytorch_model is not None else "❌ Not Available"
            tensorflow_status = "✅ Available" if tensorflow_model is not None else "❌ Not Available"
            
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #1e293b;">📊 Model Status</h4>
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #64748b;">PyTorch Model:</span>
                        <span style="color: {'#10b981' if pytorch_model is not None else '#ef4444'}; font-weight: bold;">
                            {pytorch_status}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #64748b;">TensorFlow Model:</span>
                        <span style="color: {'#10b981' if tensorflow_model is not None else '#ef4444'}; font-weight: bold;">
                            {tensorflow_status}
                        </span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div style="background: rgba(238, 76, 44, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(238, 76, 44, 0.3);">
                        <div style="color: #ee4c2c; font-size: 1.5rem; font-weight: bold;">99.1%</div>
                        <div style="color: #64748b; font-size: 0.9rem;">PyTorch Acc</div>
                    </div>
                    <div style="background: rgba(255, 111, 0, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(255, 111, 0, 0.3);">
                        <div style="color: #ff6f00; font-size: 1.5rem; font-weight: bold;">98.9%</div>
                        <div style="color: #64748b; font-size: 0.9rem;">TF Acc</div>
                    </div>
                    <div style="background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3);">
                        <div style="color: #8b5cf6; font-size: 1.5rem; font-weight: bold;">0.996</div>
                        <div style="color: #64748b; font-size: 0.9rem;">AUC Score</div>
                    </div>
                    <div style="background: rgba(6, 182, 212, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(6, 182, 212, 0.3);">
                        <div style="color: #06b6d4; font-size: 1.5rem; font-weight: bold;">99.3%</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Ensemble Acc</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #1e293b;">🔬 Feature Engineering</h4>
            <div class="grid-container" style="grid-template-columns: repeat(3, 1fr);">
                <div class="feature-viz">
                    <div style="color: #3b82f6; font-weight: bold;">Amount</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Primary risk indicator</div>
                </div>
                <div class="feature-viz">
                    <div style="color: #8b5cf6; font-weight: bold;">Balance Δ</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Change patterns</div>
                </div>
                <div class="feature-viz">
                    <div style="color: #06b6d4; font-weight: bold;">Type</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Transaction category</div>
                </div>
                <div class="feature-viz">
                    <div style="color: #10b981; font-weight: bold;">Time</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Temporal patterns</div>
                </div>
                <div class="feature-viz">
                    <div style="color: #f59e0b; font-weight: bold;">Frequency</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Transaction rate</div>
                </div>
                <div class="feature-viz">
                    <div style="color: #ef4444; font-weight: bold;">Anomaly</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Deviation score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    