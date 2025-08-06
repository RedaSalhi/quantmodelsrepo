import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, MarketDataProcessor
from var_models import VaRCalculator, VaRVisualizer, ComponentVaR, StressTestScenarios

st.set_page_config(
    page_title="VaR Analytics",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .var-card {
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
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("⚠️ Value at Risk Analytics")
    st.markdown("Comprehensive VaR analysis using parametric, historical, and Monte Carlo methods.")
    
    # Initialize session state
    if 'var_data' not in st.session_state:
        st.session_state.var_data = None
    if 'var_results' not in st.session_state:
        st.session_state.var_results = None
    
    # Sidebar configuration
    st.sidebar.title("🔧 VaR Configuration")
    
    # Asset class selection
    asset_class = st.sidebar.selectbox(
        "Select Asset Class:",
        ["Equities", "Fixed Income", "Commodities", "Mixed Portfolio"]
    )
    
    data_loader = DataLoader()
    
    # Asset selection based on class
    if asset_class == "Equities":
        data_source = st.sidebar.radio(
            "Equity Data Source:",
            ["Sample Portfolios", "Custom Tickers"]
        )
        
        if data_source == "Sample Portfolios":
            sample_portfolios = data_loader.get_sample_portfolios()
            selected_portfolio = st.sidebar.selectbox(
                "Choose Portfolio:",
                list(sample_portfolios.keys())
            )
            tickers = sample_portfolios[selected_portfolio]
        else:
            ticker_input = st.sidebar.text_area(
                "Enter Tickers (comma-separated):",
                value="AAPL,GOOGL,MSFT,TSLA,JPM",
                help="Enter stock tickers"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    elif asset_class == "Fixed Income":
        st.sidebar.info("Using Treasury yield data from FRED")
        tickers = ["Treasury Yields"]  # Placeholder
    
    elif asset_class == "Commodities":
        st.sidebar.info("Using commodity futures data")
        tickers = ["Commodities"]  # Placeholder
    
    else:  # Mixed Portfolio
        equity_tickers = st.sidebar.text_input(
            "Equity Tickers:",
            value="SPY,QQQ,IWM"
        ).split(',')
        
        include_bonds = st.sidebar.checkbox("Include Bonds", value=True)
        include_commodities = st.sidebar.checkbox("Include Commodities", value=True)
        
        tickers = [t.strip() for t in equity_tickers if t.strip()]
        if include_bonds:
            tickers.extend(["TLT", "IEF"])  # Bond ETFs
        if include_commodities:
            tickers.extend(["GLD", "USO"])  # Commodity ETFs
    
    # Portfolio configuration
    st.sidebar.markdown("### Portfolio Settings")
    
    portfolio_value = st.sidebar.number_input(
        "Portfolio Value ($):",
        min_value=10000,
        value=1000000,
        step=10000,
        format="%d"
    )
    
    # Weight specification
    if len(tickers) > 1:
        weight_method = st.sidebar.radio(
            "Weight Method:",
            ["Equal Weights", "Custom Weights"]
        )
        
        if weight_method == "Custom Weights":
            st.sidebar.markdown("**Custom Weights:**")
            custom_weights = []
            for ticker in tickers:
                weight = st.sidebar.slider(
                    f"{ticker}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(tickers),
                    step=0.01,
                    format="%.2f"
                )
                custom_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(custom_weights)
            if total_weight > 0:
                weights = np.array(custom_weights) / total_weight
            else:
                weights = np.array([1.0/len(tickers)] * len(tickers))
            
            st.sidebar.info(f"Total Weight: {sum(weights):.2f}")
        else:
            weights = np.array([1.0/len(tickers)] * len(tickers))
    else:
        weights = np.array([1.0])
    
    # Time period and VaR parameters
    period = st.sidebar.selectbox(
        "Time Period:",
        ["1y", "2y", "3y", "5y"],
        index=1
    )
    
    confidence_levels = st.sidebar.multiselect(
        "Confidence Levels:",
        [0.01, 0.05, 0.10],
        default=[0.01, 0.05, 0.10],
        format_func=lambda x: f"{x:.0%}"
    )
    
    # Load and process data
    if tickers and len(tickers) >= 1:
        try:
            with st.spinner("Loading market data..."):
                if asset_class == "Fixed Income":
                    price_data = data_loader.get_treasury_rates()
                elif asset_class == "Commodities":
                    price_data = data_loader.get_commodity_data(period)
                else:
                    price_data = data_loader.get_stock_data(tickers, period)
                
                if price_data.empty:
                    st.error("No data available for selected assets.")
                    return
                
                # Clean data and calculate returns
                price_data = MarketDataProcessor.clean_data(price_data)
                
                if asset_class == "Fixed Income":
                    # For yields, calculate first differences instead of returns
                    returns_data = price_data.diff().dropna() / 100  # Convert bp to decimal
                else:
                    returns_data = data_loader.calculate_returns(price_data)
                
                st.session_state.var_data = {
                    'prices': price_data,
                    'returns': returns_data,
                    'weights': weights,
                    'portfolio_value': portfolio_value,
                    'asset_class': asset_class,
                    'tickers': tickers
                }
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    else:
        st.warning("Please select at least 1 asset to continue.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 VaR Calculation", 
        "📈 Visualization", 
        "🧪 Stress Testing", 
        "🔍 Component Analysis", 
        "✅ Backtesting"
    ])
    
    with tab1:
        st.header("VaR Calculation")
        
        if st.session_state.var_data is not None:
            var_data = st.session_state.var_data
            returns_data = var_data['returns']
            weights = var_data['weights']
            portfolio_value = var_data['portfolio_value']
            
            # Initialize VaR calculator
            var_calculator = VaRCalculator(returns_data, weights)
            var_calculator.portfolio_value = portfolio_value
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🔥 Calculate VaR", type="primary"):
                    with st.spinner("Calculating VaR using multiple methods..."):
                        # Calculate VaR using different methods
                        parametric_var = var_calculator.parametric_var(confidence_levels)
                        historical_var = var_calculator.historical_var(confidence_levels)
                        monte_carlo_var = var_calculator.monte_carlo_var(
                            confidence_levels, n_simulations=10000
                        )
                        
                        # Calculate Expected Shortfall
                        historical_es = var_calculator.calculate_expected_shortfall(
                            confidence_levels, method='historical'
                        )
                        
                        st.session_state.var_results = {
                            'parametric': parametric_var,
                            'historical': historical_var,
                            'monte_carlo': monte_carlo_var,
                            'expected_shortfall': historical_es
                        }
                
                # Display VaR results
                if st.session_state.var_results is not None:
                    results = st.session_state.var_results
                    
                    # VaR comparison chart
                    visualizer = VaRVisualizer(var_calculator)
                    fig_comparison = visualizer.plot_var_comparison(
                        results['parametric'],
                        results['historical'],
                        results['monte_carlo']
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col2:
                if st.session_state.var_results is not None:
                    results = st.session_state.var_results
                    
                    st.subheader("🎯 VaR Summary")
                    
                    # Display VaR values for each confidence level
                    for level in confidence_levels:
                        level_key = f'VaR_{int(level*100)}%'
                        
                        st.markdown(f"**{level:.0%} Confidence Level**")
                        
                        if level_key in results['parametric']:
                            param_val = results['parametric'][level_key]['dollar']
                            hist_val = results['historical'][level_key]['dollar']
                            mc_val = results['monte_carlo'][level_key]['dollar']
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <b>Parametric:</b> ${abs(param_val):,.0f}<br>
                                <b>Historical:</b> ${abs(hist_val):,.0f}<br>
                                <b>Monte Carlo:</b> ${abs(mc_val):,.0f}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Expected Shortfall
                    st.markdown("**Expected Shortfall (CVaR)**")
                    for level in confidence_levels:
                        level_key = f'ES_{int(level*100)}%'
                        if level_key in results['expected_shortfall']:
                            es_val = results['expected_shortfall'][level_key]['dollar']
                            st.markdown(f"• {level:.0%}: ${abs(es_val):,.0f}")
    
    with tab2:
        st.header("VaR Visualization")
        
        if st.session_state.var_data is not None:
            var_data = st.session_state.var_data
            returns_data = var_data['returns']
            weights = var_data['weights']
            
            var_calculator = VaRCalculator(returns_data, weights)
            visualizer = VaRVisualizer(var_calculator)
            
            # Portfolio return distribution
            st.subheader("📊 Return Distribution Analysis")
            
            selected_confidence = st.selectbox(
                "Select Confidence Level for Visualization:",
                confidence_levels,
                format_func=lambda x: f"{x:.0%}"
            )
            
            fig_dist = visualizer.plot_return_distribution(
                show_var=True,
                confidence_level=selected_confidence
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Time-varying VaR
            st.subheader("📈 Time-Varying Risk Metrics")
            
            window_size = st.slider(
                "Rolling Window Size (days):",
                min_value=30,
                max_value=500,
                value=252,
                step=10
            )
            
            fig_time_var = visualizer.plot_var_time_series(window_size)
            st.plotly_chart(fig_time_var, use_container_width=True)
            
            # Monte Carlo simulation
            st.subheader("🎲 Monte Carlo Simulation")
            
            col1, col2 = st.columns(2)
            with col1:
                n_sims = st.slider("Number of Simulations:", 100, 2000, 1000, 50)
            with col2:
                time_horizon = st.slider("Time Horizon (days):", 1, 60, 30, 1)
            
            fig_mc = visualizer.plot_monte_carlo_simulation(n_sims, time_horizon)
            st.plotly_chart(fig_mc, use_container_width=True)
    
    with tab3:
        st.header("Stress Testing")
        
        if st.session_state.var_data is not None:
            var_data = st.session_state.var_data
            returns_data = var_data['returns']
            weights = var_data['weights']
            portfolio_value = var_data['portfolio_value']
            
            # Predefined stress scenarios
            st.subheader("🌪️ Market Stress Scenarios")
            
            scenarios = StressTestScenarios.get_market_scenarios()
            
            selected_scenarios = st.multiselect(
                "Select Stress Scenarios:",
                list(scenarios.keys()),
                default=list(scenarios.keys())[:3]
            )
            
            if selected_scenarios:
                # Asset mapping for stress testing
                if var_data['asset_class'] in ['Equities', 'Mixed Portfolio']:
                    asset_mapping = {asset: 'equity' for asset in var_data['tickers']}
                    # Update mapping for known bond/commodity ETFs
                    bond_etfs = ['TLT', 'IEF', 'AGG', 'BND']
                    commodity_etfs = ['GLD', 'USO', 'SLV', 'DBA']
                    
                    for asset in var_data['tickers']:
                        if asset in bond_etfs:
                            asset_mapping[asset] = 'bond'
                        elif asset in commodity_etfs:
                            asset_mapping[asset] = 'commodity'
                else:
                    asset_mapping = None
                
                stress_results = {}
                
                for scenario_name in selected_scenarios:
                    scenario = scenarios[scenario_name]
                    portfolio_shock = StressTestScenarios.apply_scenario_to_portfolio(
                        returns_data, weights, scenario, asset_mapping
                    )
                    
                    portfolio_loss = portfolio_shock * portfolio_value
                    
                    stress_results[scenario_name] = {
                        'description': scenario['description'],
                        'portfolio_return': portfolio_shock,
                        'portfolio_loss': portfolio_loss,
                        'loss_percentage': portfolio_shock * 100
                    }
                
                # Display stress test results
                stress_df = pd.DataFrame([
                    {
                        'Scenario': name,
                        'Description': data['description'],
                        'Portfolio Return': f"{data['portfolio_return']:.2%}",
                        'Portfolio Loss': f"${abs(data['portfolio_loss']):,.0f}",
                        'Loss %': f"{abs(data['loss_percentage']):.2f}%"
                    }
                    for name, data in stress_results.items()
                ])
                
                st.dataframe(stress_df, use_container_width=True)
                
                # Visualization of stress test results
                import plotly.graph_objects as go
                
                fig_stress = go.Figure()
                
                scenarios_list = list(stress_results.keys())
                losses = [abs(stress_results[s]['portfolio_loss']) for s in scenarios_list]
                
                fig_stress.add_trace(go.Bar(
                    x=scenarios_list,
                    y=losses,
                    text=[f'${loss:,.0f}' for loss in losses],
                    textposition='auto',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_stress.update_layout(
                    title='Stress Testing Results - Portfolio Losses',
                    xaxis_title='Stress Scenario',
                    yaxis_title='Portfolio Loss ($)',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_stress, use_container_width=True)
            
            # Custom scenario builder
            st.subheader("🛠️ Custom Scenario Builder")
            
            with st.expander("Build Custom Stress Scenario"):
                col1, col2 = st.columns(2)
                
                with col1:
                    custom_equity_shock = st.slider(
                        "Equity Shock (%):",
                        -50, 50, -20, 1
                    ) / 100
                    
                    custom_bond_shock = st.slider(
                        "Bond Shock (%):",
                        -30, 30, 10, 1
                    ) / 100
                
                with col2:
                    custom_commodity_shock = st.slider(
                        "Commodity Shock (%):",
                        -50, 50, -15, 1
                    ) / 100
                    
                    custom_fx_shock = st.slider(
                        "FX Shock (%):",
                        -30, 30, 5, 1
                    ) / 100
                
                custom_scenario = {
                    'equity_shock': custom_equity_shock,
                    'bond_shock': custom_bond_shock,
                    'commodity_shock': custom_commodity_shock,
                    'fx_shock': custom_fx_shock
                }
                
                if st.button("🧪 Test Custom Scenario"):
                    custom_shock = StressTestScenarios.apply_scenario_to_portfolio(
                        returns_data, weights, custom_scenario, asset_mapping
                    )
                    custom_loss = custom_shock * portfolio_value
                    
                    st.markdown(f"""
                    <div class="var-card">
                        <h3>Custom Scenario Results</h3>
                        <p><b>Portfolio Return:</b> {custom_shock:.2%}</p>
                        <p><b>Portfolio Loss:</b> ${abs(custom_loss):,.0f}</p>
                        <p><b>Loss Percentage:</b> {abs(custom_shock)*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab4:
        st.header("Component VaR Analysis")
        
        if st.session_state.var_data is not None and len(var_data['tickers']) > 1:
            var_data = st.session_state.var_data
            returns_data = var_data['returns']
            weights = var_data['weights']
            
            # Component VaR calculation
            comp_var = ComponentVaR(returns_data, weights)
            
            confidence_level = st.selectbox(
                "Select Confidence Level for Component Analysis:",
                [0.01, 0.05, 0.10],
                index=1,
                format_func=lambda x: f"{x:.0%}",
                key="component_confidence"
            )
            
            if st.button("🔍 Calculate Component VaR"):
                with st.spinner("Calculating component contributions..."):
                    component_results = comp_var.calculate_component_var(confidence_level)
                    incremental_results = comp_var.calculate_incremental_var(confidence_level)
                    
                    # Display component VaR results
                    st.subheader("📊 Component VaR Breakdown")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Risk Contribution by Asset**")
                        
                        # Format component VaR table
                        display_df = component_results.copy()
                        display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2%}")
                        display_df['Component_VaR'] = display_df['Component_VaR'].apply(lambda x: f"{x:.4f}")
                        display_df['Percent_Contribution'] = display_df['Percent_Contribution'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(display_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Incremental VaR Analysis**")
                        
                        # Format incremental VaR table
                        inc_display_df = incremental_results.copy()
                        inc_display_df['Current_Weight'] = inc_display_df['Current_Weight'].apply(lambda x: f"{x:.2%}")
                        inc_display_df['Weight_Change'] = inc_display_df['Weight_Change'].apply(lambda x: f"{x:.2%}")
                        inc_display_df['Incremental_VaR'] = inc_display_df['Incremental_VaR'].apply(lambda x: f"{x:.4f}")
                        
                        st.dataframe(inc_display_df, use_container_width=True)
                    
                    # Component VaR visualization
                    import plotly.express as px
                    
                    fig_component = px.pie(
                        values=component_results['Percent_Contribution'].abs(),
                        names=component_results['Asset'],
                        title=f'Risk Contribution ({confidence_level:.0%} VaR)'
                    )
                    st.plotly_chart(fig_component, use_container_width=True)
        
        elif st.session_state.var_data is not None:
            st.info("Component VaR analysis requires multiple assets in the portfolio.")
        
        else:
            st.info("Load portfolio data first to perform component analysis.")
    
    with tab5:
        st.header("VaR Model Backtesting")
        
        if st.session_state.var_results is not None:
            var_data = st.session_state.var_data
            returns_data = var_data['returns']
            weights = var_data['weights']
            
            var_calculator = VaRCalculator(returns_data, weights)
            
            # Backtesting configuration
            st.subheader("⚙️ Backtesting Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                backtest_method = st.selectbox(
                    "VaR Method to Test:",
                    ["Historical", "Parametric", "Monte Carlo"]
                )
                
                backtest_period = st.slider(
                    "Backtest Period (days):",
                    min_value=100,
                    max_value=min(500, len(returns_data)),
                    value=250,
                    step=10
                )
            
            with col2:
                test_confidence = st.multiselect(
                    "Confidence Levels to Test:",
                    [0.01, 0.05, 0.10],
                    default=[0.05],
                    format_func=lambda x: f"{x:.0%}"
                )
            
            if st.button("🧪 Run Backtest") and test_confidence:
                with st.spinner("Running backtesting..."):
                    # Split data for backtesting
                    estimation_data = returns_data.iloc[:-backtest_period]
                    test_data = returns_data.iloc[-backtest_period:]
                    
                    # Calculate VaR on estimation period
                    backtest_calculator = VaRCalculator(estimation_data, weights)
                    
                    if backtest_method == "Historical":
                        var_estimates = backtest_calculator.historical_var(test_confidence)
                    elif backtest_method == "Parametric":
                        var_estimates = backtest_calculator.parametric_var(test_confidence)
                    else:  # Monte Carlo
                        var_estimates = backtest_calculator.monte_carlo_var(test_confidence)
                    
                    # Calculate portfolio returns for test period
                    test_portfolio_returns = (test_data * weights).sum(axis=1)
                    
                    # Perform Kupiec backtest
                    backtest_results = var_calculator.backtesting_kupiec(
                        var_estimates, test_portfolio_returns
                    )
                    
                    # Display backtest results
                    st.subheader("📊 Backtesting Results")
                    
                    backtest_df = pd.DataFrame([
                        {
                            'Confidence Level': level.replace('VaR_', '').replace('%', '% VaR'),
                            'Expected Violations': f"{data['expected_violations']:.1f}",
                            'Actual Violations': data['violations'],
                            'Violation Rate': f"{data['violation_rate']:.2%}",
                            'Expected Rate': f"{data['expected_rate']:.2%}",
                            'P-Value': f"{data['p_value']:.4f}",
                            'Model Adequate': "✅ Yes" if data['model_adequate'] else "❌ No"
                        }
                        for level, data in backtest_results.items()
                    ])
                    
                    st.dataframe(backtest_df, use_container_width=True)
                    
                    # Backtest visualization
                    visualizer = VaRVisualizer(var_calculator)
                    fig_backtest = visualizer.plot_backtest_results(backtest_results)
                    st.plotly_chart(fig_backtest, use_container_width=True)
                    
                    # Interpretation
                    st.subheader("🔍 Interpretation")
                    
                    interpretations = []
                    for level, data in backtest_results.items():
                        level_name = level.replace('VaR_', '').replace('%', '%')
                        if data['model_adequate']:
                            interpretations.append(
                                f"✅ **{level_name} VaR**: Model is adequate (p-value: {data['p_value']:.4f})"
                            )
                        else:
                            interpretations.append(
                                f"❌ **{level_name} VaR**: Model may be inadequate (p-value: {data['p_value']:.4f})"
                            )
                    
                    for interpretation in interpretations:
                        st.markdown(interpretation)
                    
                    st.info("""
                    **Kupiec Test Interpretation:**
                    - P-value > 0.05: Model is adequate (fail to reject null hypothesis)
                    - P-value ≤ 0.05: Model may be inadequate (reject null hypothesis)
                    - The test checks if the violation rate matches the expected rate
                    """)
        
        else:
            st.info("Calculate VaR first to perform backtesting.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Value at Risk Analytics • Risk Management • Stress Testing</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
