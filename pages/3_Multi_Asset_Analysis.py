import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, MarketDataProcessor
from fixed_income import (BondPricer, YieldCurveAnalyzer, BondPortfolioAnalyzer, 
                         FixedIncomeVaR, FixedIncomeVisualizer)
from var_models import VaRCalculator, VaRVisualizer

st.set_page_config(
    page_title="Multi-Asset Analysis",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .asset-class-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .comparison-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🌐 Multi-Asset Analysis")
    st.markdown("Comprehensive analysis across equities, fixed income, and commodities.")
    
    # Initialize session state
    if 'multi_asset_data' not in st.session_state:
        st.session_state.multi_asset_data = {}
    
    data_loader = DataLoader()
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Asset Configuration")
    
    # Asset class selection
    selected_assets = st.sidebar.multiselect(
        "Select Asset Classes:",
        ["Equities", "Fixed Income", "Commodities"],
        default=["Equities", "Fixed Income", "Commodities"]
    )
    
    # Time period
    period = st.sidebar.selectbox(
        "Analysis Period:",
        ["1y", "2y", "3y", "5y"],
        index=1
    )
    
    # Load data for each asset class
    asset_data = {}
    
    if "Equities" in selected_assets:
        st.sidebar.markdown("### 📈 Equity Configuration")
        equity_portfolio = st.sidebar.selectbox(
            "Equity Portfolio:",
            ["Tech Portfolio", "Dividend Portfolio", "Sector ETFs", "S&P 500"],
            key="equity_portfolio"
        )
        
        if equity_portfolio == "S&P 500":
            equity_tickers = ["SPY"]
        elif equity_portfolio == "Tech Portfolio":
            equity_tickers = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"]
        elif equity_portfolio == "Dividend Portfolio":
            equity_tickers = ["JNJ", "PG", "KO", "PFE", "VZ"]
        else:  # Sector ETFs
            equity_tickers = ["XLF", "XLK", "XLE", "XLV", "XLI"]
        
        try:
            equity_prices = data_loader.get_stock_data(equity_tickers, period)
            if not equity_prices.empty:
                equity_returns = data_loader.calculate_returns(equity_prices)
                asset_data["Equities"] = {
                    'prices': equity_prices,
                    'returns': equity_returns,
                    'tickers': equity_tickers
                }
        except Exception as e:
            st.sidebar.error(f"Error loading equity data: {str(e)}")
    
    if "Fixed Income" in selected_assets:
        st.sidebar.markdown("### 💰 Fixed Income Configuration")
        bond_type = st.sidebar.selectbox(
            "Bond Analysis Type:",
            ["Treasury Yields", "Bond Portfolio", "ETFs"],
            key="bond_type"
        )
        
        try:
            if bond_type == "Treasury Yields":
                treasury_data = data_loader.get_treasury_rates()
                if not treasury_data.empty:
                    # Convert yield changes to returns
                    treasury_returns = treasury_data.diff().dropna() / 100
                    asset_data["Fixed Income"] = {
                        'prices': treasury_data,
                        'returns': treasury_returns,
                        'type': 'yields'
                    }
            elif bond_type == "ETFs":
                bond_etfs = ["TLT", "IEF", "SHY", "LQD", "HYG"]
                bond_prices = data_loader.get_stock_data(bond_etfs, period)
                if not bond_prices.empty:
                    bond_returns = data_loader.calculate_returns(bond_prices)
                    asset_data["Fixed Income"] = {
                        'prices': bond_prices,
                        'returns': bond_returns,
                        'tickers': bond_etfs,
                        'type': 'etfs'
                    }
            else:  # Bond Portfolio
                # Create sample bond portfolio
                sample_bonds = [
                    {'face_value': 1000, 'coupon_rate': 0.03, 'yield': 0.035, 'maturity': 5},
                    {'face_value': 1000, 'coupon_rate': 0.04, 'yield': 0.038, 'maturity': 10},
                    {'face_value': 1000, 'coupon_rate': 0.045, 'yield': 0.042, 'maturity': 20}
                ]
                asset_data["Fixed Income"] = {
                    'bonds': sample_bonds,
                    'type': 'portfolio'
                }
        except Exception as e:
            st.sidebar.error(f"Error loading fixed income data: {str(e)}")
    
    if "Commodities" in selected_assets:
        st.sidebar.markdown("### 🥇 Commodity Configuration")
        commodity_type = st.sidebar.selectbox(
            "Commodity Type:",
            ["All Commodities", "Precious Metals", "Energy"],
            key="commodity_type"
        )
        
        try:
            commodity_data = data_loader.get_commodity_data(period)
            if not commodity_data.empty:
                if commodity_type == "Precious Metals":
                    commodity_data = commodity_data[['Gold', 'Silver']].dropna()
                elif commodity_type == "Energy":
                    commodity_data = commodity_data[['Crude Oil', 'Natural Gas']].dropna()
                
                if not commodity_data.empty:
                    commodity_returns = data_loader.calculate_returns(commodity_data)
                    asset_data["Commodities"] = {
                        'prices': commodity_data,
                        'returns': commodity_returns,
                        'tickers': commodity_data.columns.tolist()
                    }
        except Exception as e:
            st.sidebar.error(f"Error loading commodity data: {str(e)}")
    
    st.session_state.multi_asset_data = asset_data
    
    # Main content
    if not asset_data:
        st.warning("Please select and configure at least one asset class.")
        return
    
    # Create tabs for different analyses
    tabs = ["🏠 Overview", "📊 Correlation Analysis", "⚠️ Risk Comparison", "💼 Portfolio Construction"]
    
    if "Fixed Income" in asset_data and asset_data["Fixed Income"].get('type') == 'yields':
        tabs.append("📈 Yield Curve Analysis")
    
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:  # Overview
        st.header("Multi-Asset Overview")
        
        # Display loaded asset classes
        cols = st.columns(len(asset_data))
        
        for i, (asset_class, data) in enumerate(asset_data.items()):
            with cols[i]:
                if asset_class == "Fixed Income" and data.get('type') == 'portfolio':
                    n_assets = len(data['bonds'])
                    asset_type = "Bond Portfolio"
                elif asset_class == "Fixed Income" and data.get('type') == 'yields':
                    n_assets = len(data['returns'].columns)
                    asset_type = "Treasury Yields"
                else:
                    n_assets = len(data.get('tickers', []))
                    asset_type = asset_class
                
                st.markdown(f"""
                <div class="asset-class-card">
                    <h3>{asset_class}</h3>
                    <p><b>{asset_type}</b></p>
                    <p>{n_assets} Assets</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Asset class statistics
        st.subheader("📊 Asset Class Statistics")
        
        stats_data = []
        
        for asset_class, data in asset_data.items():
            if 'returns' in data and not data['returns'].empty:
                returns = data['returns']
                
                if len(returns.columns) > 1:
                    # Equal-weighted portfolio for multi-asset classes
                    portfolio_returns = returns.mean(axis=1)
                else:
                    portfolio_returns = returns.iloc[:, 0]
                
                # Calculate statistics
                annual_return = portfolio_returns.mean() * 252
                annual_vol = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                max_drawdown = (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max()
                
                stats_data.append({
                    'Asset Class': asset_class,
                    'Annual Return': f"{annual_return:.2%}",
                    'Annual Volatility': f"{annual_vol:.2%}",
                    'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                    'Max Drawdown': f"{max_drawdown:.2%}",
                    'Observations': len(portfolio_returns)
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # Performance comparison chart
        if len(asset_data) > 1:
            st.subheader("📈 Cumulative Performance Comparison")
            
            import plotly.graph_objects as go
            
            fig_perf = go.Figure()
            
            for asset_class, data in asset_data.items():
                if 'returns' in data and not data['returns'].empty:
                    returns = data['returns']
                    
                    if len(returns.columns) > 1:
                        portfolio_returns = returns.mean(axis=1)
                    else:
                        portfolio_returns = returns.iloc[:, 0]
                    
                    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
                    
                    fig_perf.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns * 100,
                        mode='lines',
                        name=asset_class,
                        line=dict(width=2)
                    ))
            
            fig_perf.update_layout(
                title='Cumulative Returns by Asset Class',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
    
    with tab_objects[1]:  # Correlation Analysis
        st.header("Cross-Asset Correlation Analysis")
        
        # Combine returns from all asset classes
        all_returns = pd.DataFrame()
        
        for asset_class, data in asset_data.items():
            if 'returns' in data and not data['returns'].empty:
                returns = data['returns']
                
                # Rename columns with asset class prefix
                returns_renamed = returns.copy()
                returns_renamed.columns = [f"{asset_class}_{col}" for col in returns_renamed.columns]
                
                if all_returns.empty:
                    all_returns = returns_renamed
                else:
                    all_returns = all_returns.join(returns_renamed, how='outer')
        
        if not all_returns.empty:
            # Calculate correlation matrix
            correlation_matrix = all_returns.corr()
            
            # Plot correlation heatmap
            import plotly.express as px
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Cross-Asset Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            fig_corr.update_layout(
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation insights
            st.subheader("🔍 Correlation Insights")
            
            # Find highest and lowest correlations across asset classes
            cross_asset_corrs = []
            
            asset_classes = list(asset_data.keys())
            for i, asset1 in enumerate(asset_classes):
                for j, asset2 in enumerate(asset_classes):
                    if i < j:  # Avoid duplicates
                        asset1_cols = [col for col in correlation_matrix.columns if col.startswith(f"{asset1}_")]
                        asset2_cols = [col for col in correlation_matrix.columns if col.startswith(f"{asset2}_")]
                        
                        if asset1_cols and asset2_cols:
                            # Average correlation between asset classes
                            cross_corr_values = []
                            for col1 in asset1_cols:
                                for col2 in asset2_cols:
                                    cross_corr_values.append(correlation_matrix.loc[col1, col2])
                            
                            avg_corr = np.mean(cross_corr_values)
                            cross_asset_corrs.append({
                                'Asset Class Pair': f"{asset1} - {asset2}",
                                'Average Correlation': avg_corr
                            })
            
            if cross_asset_corrs:
                corr_df = pd.DataFrame(cross_asset_corrs)
                corr_df = corr_df.sort_values('Average Correlation', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Highest Correlation:**")
                    highest_corr = corr_df.iloc[0]
                    st.info(f"{highest_corr['Asset Class Pair']}: {highest_corr['Average Correlation']:.3f}")
                
                with col2:
                    st.markdown("**Lowest Correlation:**")
                    lowest_corr = corr_df.iloc[-1]
                    st.info(f"{lowest_corr['Asset Class Pair']}: {lowest_corr['Average Correlation']:.3f}")
                
                # Display full correlation table
                st.dataframe(
                    corr_df.style.format({'Average Correlation': '{:.3f}'}),
                    use_container_width=True
                )
    
    with tab_objects[2]:  # Risk Comparison
        st.header("Multi-Asset Risk Comparison")
        
        # Portfolio configuration for VaR analysis
        st.subheader("⚙️ Portfolio Configuration")
        
        available_assets = []
        for asset_class, data in asset_data.items():
            if 'returns' in data:
                for col in data['returns'].columns:
                    available_assets.append(f"{asset_class}_{col}")
        
        if available_assets:
            # Asset selection for risk analysis
            selected_for_risk = st.multiselect(
                "Select Assets for Risk Analysis:",
                available_assets,
                default=available_assets[:min(5, len(available_assets))]
            )
            
            if selected_for_risk:
                # Combine selected asset returns
                risk_returns = pd.DataFrame()
                
                for asset in selected_for_risk:
                    asset_class, asset_name = asset.split('_', 1)
                    if asset_class in asset_data and 'returns' in asset_data[asset_class]:
                        asset_return = asset_data[asset_class]['returns'][asset_name]
                        risk_returns[asset] = asset_return
                
                risk_returns = risk_returns.dropna()
                
                if not risk_returns.empty:
                    # Equal weights for risk analysis
                    n_assets = len(selected_for_risk)
                    equal_weights = np.array([1/n_assets] * n_assets)
                    
                    # Calculate VaR for the multi-asset portfolio
                    var_calculator = VaRCalculator(risk_returns, equal_weights)
                    var_calculator.portfolio_value = 1000000  # $1M portfolio
                    
                    # VaR calculation
                    confidence_levels = [0.01, 0.05, 0.10]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if st.button("🔥 Calculate Multi-Asset VaR"):
                            with st.spinner("Calculating multi-asset VaR..."):
                                parametric_var = var_calculator.parametric_var(confidence_levels)
                                historical_var = var_calculator.historical_var(confidence_levels)
                                monte_carlo_var = var_calculator.monte_carlo_var(confidence_levels)
                                
                                # VaR comparison visualization
                                visualizer = VaRVisualizer(var_calculator)
                                fig_var_comp = visualizer.plot_var_comparison(
                                    parametric_var, historical_var, monte_carlo_var
                                )
                                st.plotly_chart(fig_var_comp, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Selected Assets:**")
                        for asset in selected_for_risk:
                            asset_class, asset_name = asset.split('_', 1)
                            st.markdown(f"• {asset_class}: {asset_name}")
                        
                        st.markdown(f"**Portfolio Value:** $1,000,000")
                        st.markdown(f"**Weighting:** Equal ({1/n_assets:.1%} each)")
                    
                    # Individual asset risk metrics
                    st.subheader("📊 Individual Asset Risk Metrics")
                    
                    individual_risk_data = []
                    
                    for asset in selected_for_risk:
                        asset_returns = risk_returns[asset]
                        
                        # Calculate individual risk metrics
                        var_95 = np.percentile(asset_returns, 5)
                        var_99 = np.percentile(asset_returns, 1)
                        volatility = asset_returns.std() * np.sqrt(252)
                        
                        individual_risk_data.append({
                            'Asset': asset.replace('_', ' - '),
                            'Daily VaR (95%)': f"{var_95:.2%}",
                            'Daily VaR (99%)': f"{var_99:.2%}",
                            'Annual Volatility': f"{volatility:.2%}",
                            'Skewness': f"{asset_returns.skew():.3f}",
                            'Kurtosis': f"{asset_returns.kurtosis():.3f}"
                        })
                    
                    risk_metrics_df = pd.DataFrame(individual_risk_data)
                    st.dataframe(risk_metrics_df, use_container_width=True)
    
    with tab_objects[3]:  # Portfolio Construction
        st.header("Multi-Asset Portfolio Construction")
        
        # Create a diversified multi-asset portfolio
        if len(asset_data) >= 2:
            st.subheader("🏗️ Strategic Asset Allocation")
            
            # Asset allocation sliders
            st.markdown("**Set Asset Class Allocations:**")
            
            allocations = {}
            total_allocation = 0
            
            cols = st.columns(len(asset_data))
            
            for i, asset_class in enumerate(asset_data.keys()):
                with cols[i]:
                    default_allocation = 100 // len(asset_data)  # Equal allocation by default
                    allocation = st.slider(
                        f"{asset_class} (%)",
                        min_value=0,
                        max_value=100,
                        value=default_allocation,
                        step=5,
                        key=f"allocation_{asset_class}"
                    )
                    allocations[asset_class] = allocation
                    total_allocation += allocation
            
            # Normalize allocations if they don't sum to 100%
            if total_allocation > 0:
                normalized_allocations = {k: v/total_allocation for k, v in allocations.items()}
            else:
                normalized_allocations = allocations
            
            st.info(f"Total Allocation: {total_allocation}%")
            
            # Portfolio analysis
            if total_allocation > 0:
                st.subheader("📈 Portfolio Analysis")
                
                # Calculate portfolio returns
                portfolio_returns_components = []
                
                for asset_class, allocation in normalized_allocations.items():
                    if allocation > 0 and asset_class in asset_data and 'returns' in asset_data[asset_class]:
                        returns = asset_data[asset_class]['returns']
                        
                        # Use equal weights within each asset class
                        if len(returns.columns) > 1:
                            asset_class_return = returns.mean(axis=1)
                        else:
                            asset_class_return = returns.iloc[:, 0]
                        
                        weighted_return = asset_class_return * allocation
                        portfolio_returns_components.append(weighted_return)
                
                if portfolio_returns_components:
                    # Combine components
                    portfolio_returns = sum(portfolio_returns_components)
                    
                    # Portfolio statistics
                    annual_return = portfolio_returns.mean() * 252
                    annual_vol = portfolio_returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    # Display portfolio metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Expected Annual Return", f"{annual_return:.2%}")
                    
                    with col2:
                        st.metric("Annual Volatility", f"{annual_vol:.2%}")
                    
                    with col3:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                    
                    # Portfolio composition pie chart
                    import plotly.express as px
                    
                    fig_allocation = px.pie(
                        values=list(allocations.values()),
                        names=list(allocations.keys()),
                        title="Strategic Asset Allocation"
                    )
                    
                    st.plotly_chart(fig_allocation, use_container_width=True)
                    
                    # Performance analysis
                    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
                    
                    import plotly.graph_objects as go
                    
                    fig_performance = go.Figure()
                    
                    fig_performance.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns * 100,
                        mode='lines',
                        name='Multi-Asset Portfolio',
                        line=dict(color='blue', width=3)
                    ))
                    
                    fig_performance.update_layout(
                        title='Multi-Asset Portfolio Cumulative Returns',
                        xaxis_title='Date',
                        yaxis_title='Cumulative Return (%)',
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig_performance, use_container_width=True)
        
        else:
            st.info("Load at least 2 asset classes to construct a multi-asset portfolio.")
    
    # Yield Curve Analysis tab (if applicable)
    if len(tabs) > 4 and "Fixed Income" in asset_data and asset_data["Fixed Income"].get('type') == 'yields':
        with tab_objects[4]:  # Yield Curve Analysis
            st.header("Yield Curve Analysis")
            
            treasury_data = asset_data["Fixed Income"]['prices']
            
            # Initialize yield curve analyzer
            yield_analyzer = YieldCurveAnalyzer(treasury_data)
            visualizer = FixedIncomeVisualizer(yield_analyzer)
            
            # Current yield curve
            st.subheader("📊 Current Yield Curve")
            
            fig_yield_curve = visualizer.plot_yield_curve()
            st.plotly_chart(fig_yield_curve, use_container_width=True)
            
            # Yield curve metrics over time
            st.subheader("📈 Term Structure Evolution")
            
            fig_term_structure = visualizer.plot_term_structure_metrics()
            st.plotly_chart(fig_term_structure, use_container_width=True)
            
            # Current yield curve statistics
            latest_metrics = yield_analyzer.calculate_yield_curve_metrics()
            
            if latest_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Curve Level", f"{latest_metrics['level']:.2f}%")
                
                with col2:
                    st.metric("Slope (30Y-2Y)", f"{latest_metrics['slope']:.2f}%")
                
                with col3:
                    st.metric("Curvature", f"{latest_metrics['curvature']:.2f}")
                
                with col4:
                    st.metric("2Y-10Y Steepness", f"{latest_metrics['steepness_2_10']:.2f}%")
            
            # Fixed income VaR analysis
            if st.button("⚠️ Calculate Fixed Income VaR"):
                with st.spinner("Calculating fixed income VaR..."):
                    # Assume a bond portfolio with average duration of 7 years
                    portfolio_duration = 7.0
                    portfolio_convexity = 50.0  # Typical convexity for 7-year duration
                    
                    yield_changes = asset_data["Fixed Income"]['returns']
                    
                    fi_var = FixedIncomeVaR(yield_changes, portfolio_duration, portfolio_convexity)
                    
                    duration_var = fi_var.duration_based_var([0.01, 0.05, 0.10])
                    
                    st.subheader("⚠️ Fixed Income VaR Results")
                    
                    var_results_data = []
                    for var_level, var_data in duration_var.items():
                        var_results_data.append({
                            'Confidence Level': var_level.replace('VaR_', '').replace('%', '% VaR'),
                            'Yield Change': f"{var_data['yield_change']:.2f} bp",
                            'Price Change': f"{var_data['price_change']:.2%}",
                            'Portfolio VaR': f"${var_data['portfolio_var']:,.0f}"
                        })
                    
                    var_df = pd.DataFrame(var_results_data)
                    st.dataframe(var_df, use_container_width=True)
                    
                    st.info(f"""
                    **Assumptions:**
                    - Portfolio Duration: {portfolio_duration} years
                    - Portfolio Convexity: {portfolio_convexity}
                    - Portfolio Value: $1,000,000
                    """)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Multi-Asset Analysis • Cross-Asset Correlation • Integrated Risk Management</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
