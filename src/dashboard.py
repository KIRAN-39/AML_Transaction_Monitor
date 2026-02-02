"""
AML Transaction Monitoring Dashboard
Interactive Streamlit web interface for real-time monitoring and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Import our modules
from detection_rules import AMLDetectionRules
from risk_scorer import RiskScorer
from alert_manager import AlertManager
from feature_engineering import FeatureEngineer
from ml_models import AMLMLModels


# Page configuration
st.set_page_config(
    page_title="AML Transaction Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-medium {
        background-color: #fff9c4;
        border-left: 4px solid #ffc107;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = None
if 'alerts_df' not in st.session_state:
    st.session_state.alerts_df = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None


def load_sample_data():
    """Load sample transaction data"""
    try:
        df = pd.read_csv('transactions.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Sample data not found. Please run data_generator.py first.")
        return None


def run_detection(df, use_ml=False):
    """Run detection rules on transactions"""
    with st.spinner("🔍 Running detection analysis..."):
        # Rule-based detection
        detector = AMLDetectionRules()
        rule_results = detector.run_all_rules(df)
        
        # Combine and score
        scorer = RiskScorer()
        combined = scorer.combine_rule_results(rule_results)
        scored_df = scorer.calculate_cumulative_scores(combined)
        
        # ML detection if requested
        if use_ml and len(df) > 100:
            try:
                engineer = FeatureEngineer()
                X, feature_names, df_features = engineer.prepare_for_ml(df)
                
                ml_models = AMLMLModels()
                X_scaled = ml_models.preprocess_features(X, fit=True)
                ml_models.train_isolation_forest(X_scaled, contamination=0.05)
                
                predictions, anomaly_scores = ml_models.predict_isolation_forest(X_scaled)
                
                # Add ML results to scored dataframe
                ml_flagged = df_features[df_features['ml_prediction'] == -1]['transaction_id'].tolist()
                scored_df['ml_flagged'] = scored_df['transaction_id'].isin(ml_flagged)
                
            except Exception as e:
                st.warning(f"⚠️ ML detection skipped: {str(e)}")
        
        return scored_df


def home_page():
    """Home page with overview and quick stats"""
    st.markdown('<p class="main-header">🔍 AML Transaction Monitoring System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the AML Transaction Monitoring Dashboard
    
    This system helps detect suspicious transaction patterns using:
    - 🎯 **Rule-Based Detection** - 5 proven AML detection rules
    - 🤖 **Machine Learning** - Isolation Forest & K-Means clustering
    - 📊 **Risk Scoring** - Intelligent multi-factor assessment
    - 🗄️ **Alert Management** - Centralized database storage
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Quick Start")
        st.markdown("""
        1. **Load Data** - Upload CSV or use sample data
        2. **Run Detection** - Analyze transactions
        3. **Review Alerts** - Check flagged transactions
        4. **Export Reports** - Generate compliance reports
        """)
    
    with col2:
        st.markdown("#### 📈 System Capabilities")
        st.markdown("""
        - Real-time transaction monitoring
        - Multi-method detection (Rules + ML)
        - Risk-based alert prioritization
        - Interactive data visualization
        - Compliance-ready reporting
        """)
    
    # Quick stats if data is loaded
    if st.session_state.transactions_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset Overview")
        
        df = st.session_state.transactions_df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with col2:
            st.metric("Unique Users", f"{df['user_id'].nunique():,}")
        
        with col3:
            total_volume = df['amount'].sum()
            st.metric("Total Volume", f"${total_volume:,.0f}")
        
        with col4:
            if st.session_state.alerts_df is not None:
                alerts_count = len(st.session_state.alerts_df)
                st.metric("Alerts Generated", f"{alerts_count:,}", 
                         delta=f"{alerts_count/len(df)*100:.1f}% of transactions")
            else:
                st.metric("Alerts Generated", "Run detection first")


def transaction_monitor_page():
    """Transaction monitoring and detection page"""
    st.markdown('<p class="main-header">🔍 Transaction Monitor</p>', unsafe_allow_html=True)
    
    # File upload or load sample
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=['csv'])
    
    with col2:
        if st.button("Load Sample Data", type="primary"):
            df = load_sample_data()
            if df is not None:
                st.session_state.transactions_df = df
                st.success(f"✅ Loaded {len(df):,} transactions")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.transactions_df = df
        st.success(f"✅ Uploaded {len(df):,} transactions")
    
    # Display loaded data
    if st.session_state.transactions_df is not None:
        df = st.session_state.transactions_df
        
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Detection configuration
        st.markdown("---")
        st.markdown("### 🎯 Run Detection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_ml = st.checkbox("Enable ML Detection", value=False, 
                                help="Uses Isolation Forest for enhanced detection")
        
        with col2:
            st.write("")  # Spacer
        
        with col3:
            if st.button("🔍 Run Detection", type="primary", use_container_width=True):
                scored_df = run_detection(df, use_ml=use_ml)
                st.session_state.alerts_df = scored_df
                st.session_state.detection_results = True
                st.success(f"✅ Detection complete! Found {len(scored_df):,} alerts")
                st.rerun()
    
    else:
        st.info("👆 Please upload a CSV file or load sample data to begin")


def alert_dashboard_page():
    """Alert dashboard with filtering and details"""
    st.markdown('<p class="main-header">🚨 Alert Dashboard</p>', unsafe_allow_html=True)
    
    if st.session_state.alerts_df is None:
        st.warning("⚠️ No alerts available. Please run detection first.")
        return
    
    alerts_df = st.session_state.alerts_df
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    risk_dist = alerts_df['risk_level'].value_counts()
    
    with col1:
        critical = risk_dist.get('CRITICAL', 0)
        st.metric("🔴 Critical", critical, 
                 help="Immediate investigation required")
    
    with col2:
        high = risk_dist.get('HIGH', 0)
        st.metric("🟠 High Risk", high,
                 help="Priority investigation")
    
    with col3:
        medium = risk_dist.get('MEDIUM', 0)
        st.metric("🟡 Medium Risk", medium,
                 help="Standard review")
    
    with col4:
        low = risk_dist.get('LOW', 0)
        st.metric("🟢 Low Risk", low,
                 help="Routine monitoring")
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filter Alerts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "Risk Level",
            options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            default=['CRITICAL', 'HIGH']
        )
    
    with col2:
        min_score = st.slider("Minimum Risk Score", 0, 150, 60)
    
    with col3:
        search_user = st.text_input("Search User ID")
    
    # Apply filters
    filtered_df = alerts_df.copy()
    
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
    
    filtered_df = filtered_df[filtered_df['final_risk_score'] >= min_score]
    
    if search_user:
        filtered_df = filtered_df[filtered_df['user_id'].str.contains(search_user, case=False)]
    
    # Display alerts
    st.markdown(f"### 📋 Alerts ({len(filtered_df)} shown)")
    
    if len(filtered_df) > 0:
        # Sort by risk score
        filtered_df = filtered_df.sort_values('final_risk_score', ascending=False)
        
        # Display table
        display_cols = ['transaction_id', 'user_id', 'amount', 'timestamp', 
                       'rule_triggered', 'final_risk_score', 'risk_level']
        
        st.dataframe(
            filtered_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=400
        )
        
        # Transaction details
        st.markdown("---")
        st.markdown("### 🔎 Transaction Details")
        
        selected_tx = st.selectbox(
            "Select transaction to view details:",
            options=filtered_df['transaction_id'].tolist(),
            format_func=lambda x: f"{x} (Score: {filtered_df[filtered_df['transaction_id']==x]['final_risk_score'].values[0]})"
        )
        
        if selected_tx:
            tx_data = filtered_df[filtered_df['transaction_id'] == selected_tx].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Transaction Information")
                st.write(f"**Transaction ID:** {tx_data['transaction_id']}")
                st.write(f"**User ID:** {tx_data['user_id']}")
                st.write(f"**Amount:** ${tx_data['amount']:,.2f}")
                st.write(f"**Type:** {tx_data['transaction_type']}")
                st.write(f"**Timestamp:** {tx_data['timestamp']}")
            
            with col2:
                st.markdown("#### Risk Assessment")
                st.write(f"**Risk Score:** {tx_data['final_risk_score']}")
                st.write(f"**Risk Level:** {tx_data['risk_level']}")
                st.write(f"**Rules Triggered:** {tx_data['rule_triggered']}")
                
                # Risk level badge
                risk_level = tx_data['risk_level']
                if risk_level == 'CRITICAL':
                    st.error("🔴 CRITICAL - Immediate Action Required")
                elif risk_level == 'HIGH':
                    st.warning("🟠 HIGH - Priority Investigation")
                elif risk_level == 'MEDIUM':
                    st.info("🟡 MEDIUM - Standard Review")
                else:
                    st.success("🟢 LOW - Routine Monitoring")
    
    else:
        st.info("No alerts match the current filters")


def analytics_page():
    """Analytics and visualization page"""
    st.markdown('<p class="main-header">📊 Analytics Dashboard</p>', unsafe_allow_html=True)
    
    if st.session_state.alerts_df is None:
        st.warning("⚠️ No data available. Please run detection first.")
        return
    
    alerts_df = st.session_state.alerts_df
    transactions_df = st.session_state.transactions_df
    
    # Risk distribution
    st.markdown("### 🚦 Risk Level Distribution")
    
    risk_counts = alerts_df['risk_level'].value_counts()
    
    fig_risk = go.Figure(data=[
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=['#f44336', '#ff9800', '#ffc107', '#4caf50']),
            hole=0.4
        )
    ])
    
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Risk score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Risk Score Distribution")
        fig_scores = px.histogram(
            alerts_df, 
            x='final_risk_score',
            nbins=30,
            title="Distribution of Risk Scores"
        )
        fig_scores.update_layout(height=350)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Amount Distribution")
        fig_amounts = px.box(
            alerts_df,
            x='risk_level',
            y='amount',
            color='risk_level',
            title="Transaction Amounts by Risk Level",
            category_orders={'risk_level': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']}
        )
        fig_amounts.update_layout(height=350)
        st.plotly_chart(fig_amounts, use_container_width=True)
    
    # Timeline analysis
    if 'timestamp' in alerts_df.columns:
        st.markdown("### 📅 Alert Timeline")
        
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        alerts_df['date'] = alerts_df['timestamp'].dt.date
        
        daily_alerts = alerts_df.groupby('date').size().reset_index(name='count')
        
        fig_timeline = px.line(
            daily_alerts,
            x='date',
            y='count',
            title="Alerts Over Time"
        )
        fig_timeline.update_layout(height=350)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Top risky users
    st.markdown("### 👥 Top High-Risk Users")
    
    top_users = alerts_df.groupby('user_id').agg({
        'transaction_id': 'count',
        'final_risk_score': 'mean',
        'amount': 'sum'
    }).reset_index()
    
    top_users.columns = ['User ID', 'Alert Count', 'Avg Risk Score', 'Total Amount']
    top_users = top_users.sort_values('Alert Count', ascending=False).head(10)
    
    st.dataframe(top_users, use_container_width=True)


def settings_page():
    """Settings and configuration page"""
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Detection Rule Configuration")
    
    st.info("💡 Adjust thresholds to tune detection sensitivity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Rule Thresholds")
        
        high_value_threshold = st.number_input(
            "High-Value Threshold ($)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Transactions above this amount are flagged"
        )
        
        velocity_threshold = st.number_input(
            "Velocity Threshold (transactions/day)",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            help="Number of transactions in 24h to trigger alert"
        )
        
        structuring_min = st.number_input(
            "Structuring Min Amount ($)",
            min_value=5000,
            max_value=9500,
            value=9000,
            step=100
        )
    
    with col2:
        st.markdown("#### Risk Scoring")
        
        st.slider(
            "Low Risk Range",
            min_value=0,
            max_value=100,
            value=(0, 30),
            help="Score range for low-risk classification"
        )
        
        st.slider(
            "Medium Risk Range",
            min_value=0,
            max_value=100,
            value=(31, 60)
        )
        
        st.slider(
            "High Risk Range",
            min_value=0,
            max_value=150,
            value=(61, 90)
        )
    
    st.markdown("---")
    
    st.markdown("### 🤖 ML Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        contamination = st.slider(
            "Expected Anomaly Rate (%)",
            min_value=1,
            max_value=10,
            value=5,
            help="Expected percentage of fraudulent transactions"
        ) / 100
    
    with col2:
        n_clusters = st.slider(
            "Number of Clusters (K-Means)",
            min_value=3,
            max_value=10,
            value=5
        )
    
    if st.button("💾 Save Configuration"):
        st.success("✅ Configuration saved!")


def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "🔍 Transaction Monitor", "🚨 Alert Dashboard", 
         "📊 Analytics", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    
    # System info
    st.sidebar.markdown("### 📊 System Status")
    
    if st.session_state.transactions_df is not None:
        st.sidebar.success(f"✅ Data Loaded: {len(st.session_state.transactions_df):,} transactions")
    else:
        st.sidebar.warning("⚠️ No data loaded")
    
    if st.session_state.alerts_df is not None:
        st.sidebar.success(f"✅ Alerts: {len(st.session_state.alerts_df):,} flagged")
    else:
        st.sidebar.info("ℹ️ Run detection to generate alerts")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    **AML Transaction Monitor**
    
    Version 1.0
    
    Detection Methods:
    - Rule-Based
    - Isolation Forest
    - K-Means Clustering
    """)
    
    # Route to selected page
    if page == "🏠 Home":
        home_page()
    elif page == "🔍 Transaction Monitor":
        transaction_monitor_page()
    elif page == "🚨 Alert Dashboard":
        alert_dashboard_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()


if __name__ == "__main__":
    main()
