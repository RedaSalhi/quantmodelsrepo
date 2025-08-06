import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure page
st.set_page_config(
    page_title="Portfolio Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📊 Portfolio Analytics Hub</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("---")
    
    # Main page content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>🚀 Welcome to Your Financial Analytics Platform</h2>
            <p>A comprehensive toolkit for modern portfolio management and risk analysis, 
            built with cutting-edge financial theory and real-world data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Key Features")
        
        features = [
            ("📈 Portfolio Theory", "Modern Portfolio Theory implementation with efficient frontier analysis"),
            ("⚠️ Risk Analytics", "Comprehensive Value-at-Risk modeling with multiple methodologies"),
            ("💰 Asset Classes", "Multi-asset analysis covering equities, bonds, and commodities"),
            ("📊 Interactive Visualizations", "Dynamic charts and risk profiles"),
            ("🎓 Educational Resources", "Learn the theory behind the models")
        ]
        
        for title, desc in features:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📚 Quick Start Guide")
        
        st.markdown("""
        **1. Portfolio Theory** 📊
        - Build optimal portfolios
        - Analyze efficient frontier
        - Calculate Sharpe ratios
        
        **2. VaR Analytics** ⚠️
        - Estimate portfolio risk
        - Compare VaR methods
        - Stress test scenarios
        
        **3. Multi-Asset Analysis** 🌐
        - Equities analysis
        - Fixed income modeling
        - Commodity exposure
        
        **4. Educational Hub** 🎓
        - Learn methodologies
        - Understand assumptions
        - Access references
        """)
        
        st.markdown("### 🛠️ Data Sources")
        st.info("""
        - **Yahoo Finance**: Stock prices, commodities
        - **FRED**: Economic indicators, yields
        - **Real-time**: Live market data
        """)
    
    st.markdown("---")
    
    # Footer
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Start Portfolio Analysis", use_container_width=True):
            st.switch_page("pages/1_Portfolio_Theory.py")
    
    with col2:
        if st.button("⚠️ Risk Analysis", use_container_width=True):
            st.switch_page("pages/2_VaR_Analytics.py")
    
    with col3:
        if st.button("🎓 Learn More", use_container_width=True):
            st.switch_page("pages/4_Educational.py")
    
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Built with Streamlit • Modern Portfolio Theory • Risk Management</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
