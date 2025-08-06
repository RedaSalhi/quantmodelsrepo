import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, MarketDataProcessor
from portfolio import PortfolioOptimizer, PortfolioVisualizer, RiskMetrics

st.set_page_config(
    page_title="Portfolio Theory",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stAlert > div {
        background-color: #f0f2f6;
        border: 1px solid #d4dae4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 Modern Portfolio Theory")
    st.markdown("Build optimal portfolios using Markowitz optimization and efficient frontier analysis.")
    
    # Initialize session state
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    # Sidebar for configuration
    st.sidebar.title("🔧 Portfolio Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Sample Portfolios", "Custom Tickers", "Upload Data"]
    )
    
    data_loader = DataLoader()
    
    if data_source == "Sample Portfolios":
        sample_portfolios = data_loader.get_sample_portfolios()
        selected_portfolio = st.sidebar.selectbox(
            "Choose Portfolio:",
            list(sample_portfolios.keys())
        )
        tickers = sample_portfolios[selected_portfolio]
        st.sidebar.write(f"**Selected Assets:** {', '.join(tickers)}")
        
    elif data_source == "Custom Tickers":
        ticker_input = st.sidebar.text_area(
            "Enter Tickers (comma-separated):",
            value="AAPL,GOOGL,MSFT,TSLA,AMZN",
            help="Enter stock tickers separated by commas"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
    else:  # Upload Data
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file with price data",
            type=['csv'],
            help="CSV should have dates as index and asset prices as columns"
        )
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                tickers = uploaded_data.columns.tolist()
                st.sidebar.success(f"Loaded {len(tickers)} assets")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                tickers = []
        else:
            tickers = []
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Select Time Period:",
        ["1y", "2y", "3y", "5y", "max"],
        index=1
    )
    
    # Risk-free rate configuration
    st.sidebar.markdown("### Risk Parameters")
    
    rf_source = st.sidebar.radio(
        "Risk-Free Rate Source:",
        ["Auto (3M Treasury)", "Manual Input"]
    )
    
    if rf_source == "Manual Input":
        risk_free_rate = st.sidebar.number_input(
            "Risk-Free Rate (%):",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.2f"
        ) / 100
    else:
        risk_free_rate = data_loader.get_risk_free_rate()
        st.sidebar.info(f"Current 3M Treasury Rate: {risk_free_rate:.2%}")
    
    # Market portfolio selection
    market_portfolio_type = st.sidebar.radio(
        "Market Portfolio:",
        ["Max Sharpe Portfolio", "S&P 500 (SPY)"]
    )
    
    # Load and process data
    if tickers and len(tickers) >= 2:
        try:
            with st.spinner("Loading market data..."):
                if data_source == "Upload Data" and uploaded_file is not None:
                    price_data = uploaded_data
                else:
                    price_data = data_loader.get_stock_data(tickers, period)
                
                if price_data.empty:
                    st.error("No data available for selected assets.")
                    return
                
                # Clean data
                price_data = MarketDataProcessor.clean_data(price_data)
                returns_data = data_loader.calculate_returns(price_data)
                
                st.session_state.portfolio_data = {
                    'prices': price_data,
                    'returns': returns_data,
                    'tickers': tickers,
                    'risk_free_rate': risk_free_rate
                }
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    else:
        st.warning("Please select at least 2 assets to continue.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Optimization", "📊 Analysis", "⚠️ Risk Metrics", "📋 Summary"])
    
    with tab1:
        st.header("Portfolio Optimization")
        
        if st.session_state.portfolio_data is not None:
            returns_data = st.session_state.portfolio_data['returns']
            risk_free_rate = st.session_state.portfolio_data['risk_free_rate']
            
            # Initialize optimizer
            optimizer = PortfolioOptimizer(returns_data, risk_free_rate)
            visualizer = PortfolioVisualizer(optimizer)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 Run Optimization", type="primary"):
                    with st.spinner("Optimizing portfolios..."):
                        # Calculate optimal portfolios
                        min_vol_portfolio = optimizer.minimize_volatility()
                        max_sharpe_portfolio = optimizer.maximize_sharpe_ratio()
                        efficient_frontier = optimizer.efficient_frontier(n_portfolios=50)
                        
                        # Market portfolio
                        if market_portfolio_type == "S&P 500 (SPY)":
                            try:
                                spy_data = data_loader.get_stock_data(['SPY'], period)
                                spy_returns = data_loader.calculate_returns(spy_data)
                                spy_return, spy_vol, spy_sharpe = optimizer.portfolio_performance(np.array([1.0]))
                                market_portfolio = {
                                    'weights': np.array([1.0]),
                                    'expected_return': spy_return,
                                    'volatility': spy_vol,
                                    'sharpe_ratio': spy_sharpe,
                                    'success': True
                                }
                            except:
                                market_portfolio = max_sharpe_portfolio
                        else:
                            market_portfolio = max_sharpe_portfolio
                        
                        st.session_state.optimization_results = {
                            'min_vol': min_vol_portfolio,
                            'max_sharpe': max_sharpe_portfolio,
                            'efficient_frontier': efficient_frontier,
                            'market_portfolio': market_portfolio
                        }
                
                # Display results
                if st.session_state.optimization_results is not None:
                    results = st.session_state.optimization_results
                    
                    # Efficient frontier plot
                    fig_ef = visualizer.plot_efficient_frontier(
                        results['efficient_frontier'],
                        results['min_vol'],
                        results['max_sharpe'],
                        results['market_portfolio']
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)
            
            with col2:
                if st.session_state.optimization_results is not None:
                    results = st.session_state.optimization_results
                    
                    # Display portfolio metrics
                    st.subheader("🎯 Optimal Portfolios")
                    
                    # Minimum Variance Portfolio
                    if results['min_vol']['success']:
                        st.markdown("**Minimum Variance Portfolio**")
                        mv_port = results['min_vol']
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>Expected Return:</b> {mv_port['expected_return']:.2%}<br>
                            <b>Volatility:</b> {mv_port['volatility']:.2%}<br>
                            <b>Sharpe Ratio:</b> {mv_port['sharpe_ratio']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Maximum Sharpe Portfolio
                    if results['max_sharpe']['success']:
                        st.markdown("**Maximum Sharpe Ratio Portfolio**")
                        ms_port = results['max_sharpe']
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>Expected Return:</b> {ms_port['expected_return']:.2%}<br>
                            <b>Volatility:</b> {ms_port['volatility']:.2%}<br>
                            <b>Sharpe Ratio:</b> {ms_port['sharpe_ratio']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk-free rate info
                    st.info(f"Risk-Free Rate: {risk_free_rate:.2%}")
    
    with tab2:
        st.header("Portfolio Analysis")
        
        if st.session_state.optimization_results is not None:
            results = st.session_state.optimization_results
            returns_data = st.session_state.portfolio_data['returns']
            
            # Portfolio selection for detailed analysis
            portfolio_choice = st.selectbox(
                "Select Portfolio for Analysis:",
                ["Minimum Variance", "Maximum Sharpe", "Equal Weights"]
            )
            
            if portfolio_choice == "Minimum Variance":
                selected_portfolio = results['min_vol']
            elif portfolio_choice == "Maximum Sharpe":
                selected_portfolio = results['max_sharpe']
            else:  # Equal Weights
                n_assets = len(returns_data.columns)
                selected_portfolio = {
                    'weights': np.array([1/n_assets] * n_assets),
                    'success': True
                }
                ret, vol, sharpe = optimizer.portfolio_performance(selected_portfolio['weights'])
                selected_portfolio.update({
                    'expected_return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                })
            
            if selected_portfolio['success']:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Portfolio weights pie chart
                    fig_weights = visualizer.plot_portfolio_weights(
                        selected_portfolio['weights'], 
                        portfolio_choice
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
                
                with col2:
                    # Correlation matrix
                    fig_corr = visualizer.plot_correlation_matrix()
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Asset allocation table
                st.subheader("📊 Asset Allocation")
                weights_df = pd.DataFrame({
                    'Asset': returns_data.columns,
                    'Weight': selected_portfolio['weights'],
                    'Weight (%)': selected_portfolio['weights'] * 100
                }).sort_values('Weight', ascending=False)
                
                st.dataframe(weights_df.style.format({
                    'Weight': '{:.4f}',
                    'Weight (%)': '{:.2f}%'
                }), use_container_width=True)
        
        else:
            st.info("Run optimization first to see detailed analysis.")
    
    with tab3:
        st.header("Risk Metrics")
        
        if st.session_state.optimization_results is not None:
            results = st.session_state.optimization_results
            returns_data = st.session_state.portfolio_data['returns']
            
            # Portfolio selection
            portfolio_choice = st.selectbox(
                "Select Portfolio for Risk Analysis:",
                ["Minimum Variance", "Maximum Sharpe", "Equal Weights"],
                key="risk_portfolio_choice"
            )
            
            if portfolio_choice == "Minimum Variance":
                selected_portfolio = results['min_vol']
            elif portfolio_choice == "Maximum Sharpe":
                selected_portfolio = results['max_sharpe']
            else:
                n_assets = len(returns_data.columns)
                weights = np.array([1/n_assets] * n_assets)
                selected_portfolio = {'weights': weights, 'success': True}
            
            if selected_portfolio['success']:
                weights = selected_portfolio['weights']
                portfolio_returns = (returns_data * weights).sum(axis=1)
                
                # Calculate risk metrics
                var_95 = RiskMetrics.calculate_var(portfolio_returns, 0.05)
                var_99 = RiskMetrics.calculate_var(portfolio_returns, 0.01)
                cvar_95 = RiskMetrics.calculate_cvar(portfolio_returns, 0.05)
                cvar_99 = RiskMetrics.calculate_cvar(portfolio_returns, 0.01)
                max_dd_metrics = RiskMetrics.calculate_max_drawdown(portfolio_returns)
                
                # Display risk metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("VaR (95%)", f"{var_95:.2%}", help="Value at Risk at 95% confidence")
                
                with col2:
                    st.metric("VaR (99%)", f"{var_99:.2%}", help="Value at Risk at 99% confidence")
                
                with col3:
                    st.metric("CVaR (95%)", f"{cvar_95:.2%}", help="Conditional Value at Risk at 95%")
                
                with col4:
                    st.metric("CVaR (99%)", f"{cvar_99:.2%}", help="Conditional Value at Risk at 99%")
                
                # Additional risk metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Drawdown", f"{max_dd_metrics['max_drawdown']:.2%}")
                
                with col2:
                    st.metric("Current Drawdown", f"{max_dd_metrics['current_drawdown']:.2%}")
                
                with col3:
                    volatility = portfolio_returns.std() * np.sqrt(252)
                    st.metric("Annualized Volatility", f"{volatility:.2%}")
                
                # Returns distribution analysis
                fig_dist = visualizer.plot_returns_distribution()
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Asset statistics
                st.subheader("📈 Individual Asset Statistics")
                asset_stats = MarketDataProcessor.calculate_statistics(returns_data)
                asset_stats['Sharpe Ratio'] = (asset_stats['Mean'] - risk_free_rate) / asset_stats['Volatility']
                
                st.dataframe(asset_stats.style.format({
                    'Mean': '{:.2%}',
                    'Volatility': '{:.2%}',
                    'Skewness': '{:.3f}',
                    'Kurtosis': '{:.3f}',
                    'Min': '{:.2%}',
                    'Max': '{:.2%}',
                    'Sharpe Ratio': '{:.3f}'
                }), use_container_width=True)
        
        else:
            st.info("Run optimization first to see risk metrics.")
    
    with tab4:
        st.header("Portfolio Summary")
        
        if st.session_state.optimization_results is not None:
            results = st.session_state.optimization_results
            returns_data = st.session_state.portfolio_data['returns']
            
            # Executive summary
            st.subheader("📋 Executive Summary")
            
            # Portfolio comparison table
            portfolio_comparison = []
            
            if results['min_vol']['success']:
                mv_port = results['min_vol']
                portfolio_comparison.append({
                    'Portfolio': 'Minimum Variance',
                    'Expected Return': f"{mv_port['expected_return']:.2%}",
                    'Volatility': f"{mv_port['volatility']:.2%}",
                    'Sharpe Ratio': f"{mv_port['sharpe_ratio']:.3f}"
                })
            
            if results['max_sharpe']['success']:
                ms_port = results['max_sharpe']
                portfolio_comparison.append({
                    'Portfolio': 'Maximum Sharpe',
                    'Expected Return': f"{ms_port['expected_return']:.2%}",
                    'Volatility': f"{ms_port['volatility']:.2%}",
                    'Sharpe Ratio': f"{ms_port['sharpe_ratio']:.3f}"
                })
            
            # Equal weight portfolio
            n_assets = len(returns_data.columns)
            eq_weights = np.array([1/n_assets] * n_assets)
            eq_ret, eq_vol, eq_sharpe = optimizer.portfolio_performance(eq_weights)
            portfolio_comparison.append({
                'Portfolio': 'Equal Weights',
                'Expected Return': f"{eq_ret:.2%}",
                'Volatility': f"{eq_vol:.2%}",
                'Sharpe Ratio': f"{eq_sharpe:.3f}"
            })
            
            comparison_df = pd.DataFrame(portfolio_comparison)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Key insights
            st.subheader("🔍 Key Insights")
            
            if results['min_vol']['success'] and results['max_sharpe']['success']:
                mv_port = results['min_vol']
                ms_port = results['max_sharpe']
                
                insights = [
                    f"**Diversification Benefits:** The minimum variance portfolio reduces risk to {mv_port['volatility']:.1%}, compared to equal-weight volatility of {eq_vol:.1%}",
                    f"**Risk-Return Trade-off:** The maximum Sharpe ratio portfolio achieves {ms_port['sharpe_ratio']:.3f} Sharpe ratio with {ms_port['expected_return']:.1%} expected return",
                    f"**Risk-Free Rate Impact:** Current risk-free rate of {risk_free_rate:.1%} affects portfolio selection and Sharpe ratio calculations"
                ]
                
                # Portfolio concentration analysis
                mv_concentration = np.sum(mv_port['weights']**2)  # Herfindahl index
                ms_concentration = np.sum(ms_port['weights']**2)
                eq_concentration = np.sum(eq_weights**2)
                
                insights.append(f"**Concentration Analysis:** Min variance portfolio concentration: {mv_concentration:.3f}, Max Sharpe: {ms_concentration:.3f}, Equal weights: {eq_concentration:.3f}")
                
                for insight in insights:
                    st.markdown(f"• {insight}")
            
            # Export options
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Portfolio Weights"):
                    if results['max_sharpe']['success']:
                        weights_df = pd.DataFrame({
                            'Asset': returns_data.columns,
                            'Max_Sharpe_Weights': results['max_sharpe']['weights'],
                            'Min_Variance_Weights': results['min_vol']['weights'] if results['min_vol']['success'] else 0,
                            'Equal_Weights': eq_weights
                        })
                        csv = weights_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="portfolio_weights.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if st.button("📈 Download Efficient Frontier"):
                    if not results['efficient_frontier'].empty:
                        ef_csv = results['efficient_frontier'][['Return', 'Volatility', 'Sharpe_Ratio']].to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=ef_csv,
                            file_name="efficient_frontier.csv",
                            mime="text/csv"
                        )
        
        else:
            st.info("Run optimization first to see summary.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Modern Portfolio Theory • Markowitz Optimization • Efficient Frontier Analysis</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
