import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Educational Hub",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .theory-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .formula-box {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .assumption-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .method-comparison {
        background: white;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🎓 Educational Hub")
    st.markdown("Learn the theory, methods, and assumptions behind portfolio optimization and risk management.")
    
    # Create tabs for different educational topics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Portfolio Theory",
        "⚠️ Value at Risk",
        "💰 Fixed Income",
        "🧮 Mathematical Foundations",
        "📖 References & Sources"
    ])
    
    with tab1:
        st.header("Modern Portfolio Theory")
        
        # Introduction
        st.markdown("""
        <div class="theory-card">
            <h3>🏛️ Historical Background</h3>
            <p>Modern Portfolio Theory (MPT) was introduced by Harry Markowitz in 1952 in his paper 
            "Portfolio Selection". This groundbreaking work earned him the Nobel Prize in Economics in 1990 
            and revolutionized investment management by providing a mathematical framework for portfolio construction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Concepts
        st.subheader("🔑 Key Concepts")
        
        concepts = [
            ("**Diversification**", "The practice of spreading investments across various assets to reduce risk. 'Don't put all your eggs in one basket.'"),
            ("**Efficient Frontier**", "The set of optimal portfolios offering the highest expected return for each level of risk."),
            ("**Risk-Return Trade-off**", "The principle that potential return rises with an increase in risk."),
            ("**Correlation**", "A statistical measure that determines how assets move in relation to each other."),
            ("**Sharpe Ratio**", "A measure of risk-adjusted return, calculated as (Return - Risk-free Rate) / Volatility.")
        ]
        
        for concept, description in concepts:
            st.markdown(f"• {concept}: {description}")
        
        # Mathematical Framework
        st.subheader("🧮 Mathematical Framework")
        
        st.markdown("**Expected Portfolio Return:**")
        st.markdown("""
        <div class="formula-box">
        E(Rp) = Σ wi × E(Ri)
        
        Where:
        • E(Rp) = Expected portfolio return
        • wi = Weight of asset i in the portfolio
        • E(Ri) = Expected return of asset i
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Portfolio Variance:**")
        st.markdown("""
        <div class="formula-box">
        σ²p = Σ Σ wi × wj × σij
        
        Where:
        • σ²p = Portfolio variance
        • wi, wj = Weights of assets i and j
        • σij = Covariance between assets i and j
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sharpe Ratio:**")
        st.markdown("""
        <div class="formula-box">
        Sharpe Ratio = (E(Rp) - Rf) / σp
        
        Where:
        • E(Rp) = Expected portfolio return
        • Rf = Risk-free rate
        • σp = Portfolio standard deviation
        </div>
        """, unsafe_allow_html=True)
        
        # Assumptions
        st.subheader("⚠️ Key Assumptions")
        
        assumptions = [
            "Investors are rational and risk-averse",
            "Returns are normally distributed",
            "Correlations and volatilities are constant over time",
            "No transaction costs or taxes",
            "All information is freely available to all investors",
            "Investors can borrow and lend at the risk-free rate"
        ]
        
        for assumption in assumptions:
            st.markdown(f"""
            <div class="assumption-box">
                • {assumption}
            </div>
            """, unsafe_allow_html=True)
        
        # Limitations
        st.subheader("⚡ Limitations and Criticisms")
        
        limitations = [
            ("**Historical Data Dependency**", "MPT relies on historical data, which may not predict future performance."),
            ("**Normal Distribution Assumption**", "Real asset returns often exhibit fat tails and skewness."),
            ("**Static Nature**", "The model assumes constant correlations and volatilities."),
            ("**Optimization Sensitivity**", "Small changes in inputs can lead to dramatically different optimal portfolios."),
            ("**Behavioral Factors**", "Ignores investor psychology and behavioral biases.")
        ]
        
        for limitation, description in limitations:
            st.markdown(f"• {limitation}: {description}")
        
        # Interactive Example
        st.subheader("🎯 Interactive Example: Two-Asset Portfolio")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Asset Parameters:**")
            r1 = st.slider("Asset 1 Expected Return (%)", 5, 20, 10) / 100
            r2 = st.slider("Asset 2 Expected Return (%)", 5, 20, 15) / 100
            vol1 = st.slider("Asset 1 Volatility (%)", 10, 40, 20) / 100
            vol2 = st.slider("Asset 2 Volatility (%)", 10, 40, 30) / 100
            corr = st.slider("Correlation", -1.0, 1.0, 0.3, 0.1)
            
            weight1 = st.slider("Weight in Asset 1", 0.0, 1.0, 0.6, 0.05)
            weight2 = 1 - weight1
        
        with col2:
            # Calculate portfolio metrics
            portfolio_return = weight1 * r1 + weight2 * r2
            portfolio_variance = (weight1**2 * vol1**2 + weight2**2 * vol2**2 + 
                                2 * weight1 * weight2 * vol1 * vol2 * corr)
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # Generate efficient frontier
            weights_1 = np.linspace(0, 1, 100)
            weights_2 = 1 - weights_1
            
            ef_returns = weights_1 * r1 + weights_2 * r2
            ef_variances = (weights_1**2 * vol1**2 + weights_2**2 * vol2**2 + 
                           2 * weights_1 * weights_2 * vol1 * vol2 * corr)
            ef_vols = np.sqrt(ef_variances)
            
            # Plot efficient frontier
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ef_vols * 100,
                y=ef_returns * 100,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=[vol1 * 100, vol2 * 100],
                y=[r1 * 100, r2 * 100],
                mode='markers',
                name='Individual Assets',
                marker=dict(color='red', size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=[portfolio_vol * 100],
                y=[portfolio_return * 100],
                mode='markers',
                name='Current Portfolio',
                marker=dict(color='green', size=12, symbol='diamond')
            ))
            
            fig.update_layout(
                title='Two-Asset Efficient Frontier',
                xaxis_title='Volatility (%)',
                yaxis_title='Expected Return (%)',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.metric("Portfolio Return", f"{portfolio_return:.2%}")
            st.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
            st.metric("Weight Asset 1", f"{weight1:.1%}")
            st.metric("Weight Asset 2", f"{weight2:.1%}")
    
    with tab2:
        st.header("Value at Risk (VaR)")
        
        # Introduction
        st.markdown("""
        <div class="theory-card">
            <h3>🎯 What is Value at Risk?</h3>
            <p>Value at Risk (VaR) is a statistical measure that quantifies the potential loss in value 
            of a portfolio over a defined period for a given confidence interval. It answers the question: 
            "What is the maximum loss we can expect with X% confidence over the next Y days?"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VaR Methods Comparison
        st.subheader("🔬 VaR Methodologies")
        
        methods_data = [
            {
                'Method': 'Parametric (Variance-Covariance)',
                'Description': 'Assumes returns follow a normal distribution',
                'Pros': 'Simple, fast computation; Provides smooth estimates',
                'Cons': 'Strong normality assumption; Poor for fat-tailed distributions',
                'Best Use': 'Well-diversified portfolios with normal-like returns'
            },
            {
                'Method': 'Historical Simulation',
                'Description': 'Uses actual historical return distribution',
                'Pros': 'No distributional assumptions; Captures actual tail behavior',
                'Cons': 'Assumes past represents future; Limited by historical data',
                'Best Use': 'When historical data is representative and sufficient'
            },
            {
                'Method': 'Monte Carlo Simulation',
                'Description': 'Generates random scenarios based on assumed distributions',
                'Pros': 'Flexible distributional assumptions; Can model complex portfolios',
                'Cons': 'Computationally intensive; Requires model specification',
                'Best Use': 'Complex portfolios with non-linear payoffs'
            }
        ]
        
        for method_data in methods_data:
            st.markdown(f"""
            <div class="method-comparison">
                <h4>{method_data['Method']}</h4>
                <p><strong>Description:</strong> {method_data['Description']}</p>
                <p><strong>Pros:</strong> {method_data['Pros']}</p>
                <p><strong>Cons:</strong> {method_data['Cons']}</p>
                <p><strong>Best Use:</strong> {method_data['Best Use']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mathematical Formulations
        st.subheader("📐 Mathematical Formulations")
        
        st.markdown("**Parametric VaR (Normal Distribution):**")
        st.markdown("""
        <div class="formula-box">
        VaR = μ + σ × Φ⁻¹(α)
        
        Where:
        • μ = Expected return (usually 0 for daily)
        • σ = Portfolio standard deviation
        • Φ⁻¹(α) = Inverse normal cumulative distribution function
        • α = Confidence level (e.g., 0.05 for 95% confidence)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Historical VaR:**")
        st.markdown("""
        <div class="formula-box">
        VaR = Percentile(Historical Returns, α × 100)
        
        Where:
        • Historical Returns = Sorted historical return observations
        • α = Confidence level (e.g., 0.05 for 95% confidence)
        </div>
        """, unsafe_allow_html=True)
        
        # Expected Shortfall
        st.subheader("📊 Expected Shortfall (Conditional VaR)")
        
        st.markdown("""
        Expected Shortfall (ES) measures the average loss beyond the VaR threshold. 
        It provides information about tail risk that VaR doesn't capture.
        """)
        
        st.markdown("""
        <div class="formula-box">
        ES(α) = E[Loss | Loss > VaR(α)]
        
        This is the expected value of losses that exceed the VaR threshold.
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive VaR Example
        st.subheader("🎮 Interactive VaR Calculation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Distribution Parameters:**")
            mean_return = st.slider("Daily Mean Return (%)", -1.0, 1.0, 0.0, 0.1) / 100
            volatility = st.slider("Daily Volatility (%)", 0.5, 5.0, 2.0, 0.1) / 100
            confidence = st.selectbox("Confidence Level", [90, 95, 99])
            
            portfolio_value = st.number_input("Portfolio Value ($)", 
                                            min_value=100000, 
                                            value=1000000, 
                                            step=100000,
                                            format="%d")
        
        with col2:
            # Generate sample data
            np.random.seed(42)
            n_samples = 1000
            sample_returns = np.random.normal(mean_return, volatility, n_samples)
            
            # Calculate VaR
            alpha = (100 - confidence) / 100
            parametric_var = mean_return + volatility * np.percentile(np.random.standard_normal(10000), alpha * 100)
            historical_var = np.percentile(sample_returns, alpha * 100)
            
            # Convert to dollar amounts
            parametric_var_dollar = parametric_var * portfolio_value
            historical_var_dollar = historical_var * portfolio_value
            
            # Plot distribution with VaR
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=sample_returns * 100,
                nbinsx=50,
                name='Return Distribution',
                opacity=0.7,
                histnorm='probability density'
            ))
            
            # VaR lines
            fig.add_vline(x=parametric_var * 100, 
                         line_dash="dash", 
                         line_color="red",
                         annotation_text=f"Parametric VaR ({confidence}%)")
            
            fig.add_vline(x=historical_var * 100, 
                         line_dash="dot", 
                         line_color="orange",
                         annotation_text=f"Historical VaR ({confidence}%)")
            
            fig.update_layout(
                title='VaR Illustration',
                xaxis_title='Daily Return (%)',
                yaxis_title='Density',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display results
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Parametric VaR", f"${abs(parametric_var_dollar):,.0f}")
            with col_b:
                st.metric("Historical VaR", f"${abs(historical_var_dollar):,.0f}")
        
        # Backtesting
        st.subheader("✅ Model Validation: Backtesting")
        
        st.markdown("""
        Backtesting is crucial for validating VaR models. The most common test is the **Kupiec Test**, 
        which checks if the number of VaR violations matches the expected frequency.
        """)
        
        st.markdown("""
        <div class="formula-box">
        Kupiec Test Statistic = 2 × ln[(1-p)^(T-N) × p^N] - 2 × ln[(1-x)^(T-N) × x^N]
        
        Where:
        • p = Theoretical violation rate (e.g., 0.05 for 95% VaR)
        • x = Observed violation rate
        • T = Total observations
        • N = Number of violations
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Interpretation:**
        - If p-value > 0.05: Model is adequate (violations match expected frequency)
        - If p-value ≤ 0.05: Model may be inadequate (too many or too few violations)
        """)
    
    with tab3:
        st.header("Fixed Income Analytics")
        
        # Bond Basics
        st.markdown("""
        <div class="theory-card">
            <h3>💰 Fixed Income Fundamentals</h3>
            <p>Fixed income securities are debt instruments that pay fixed periodic payments 
            and return principal at maturity. Understanding their price sensitivity to interest 
            rate changes is crucial for portfolio management and risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Duration and Convexity
        st.subheader("⏱️ Duration and Convexity")
        
        st.markdown("**Modified Duration:**")
        st.markdown("""
        <div class="formula-box">
        Modified Duration = Macaulay Duration / (1 + YTM/n)
        
        Price Change ≈ -Modified Duration × ΔYield
        
        Where:
        • YTM = Yield to Maturity
        • n = Number of compounding periods per year
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Convexity:**")
        st.markdown("""
        <div class="formula-box">
        Price Change ≈ -Duration × ΔYield + 0.5 × Convexity × (ΔYield)²
        
        Convexity improves duration's price approximation for larger yield changes.
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Bond Example
        st.subheader("🔧 Interactive Bond Price Calculator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            face_value = st.number_input("Face Value ($)", value=1000, step=100)
            coupon_rate = st.slider("Coupon Rate (%)", 0.0, 10.0, 4.0, 0.5) / 100
            years_to_maturity = st.slider("Years to Maturity", 1, 30, 10)
            ytm = st.slider("Yield to Maturity (%)", 1.0, 10.0, 5.0, 0.1) / 100
        
        with col2:
            # Calculate bond metrics
            from math import pow
            
            # Bond price calculation
            periods = years_to_maturity * 2  # Semi-annual
            coupon_payment = (coupon_rate * face_value) / 2
            discount_rate = ytm / 2
            
            if discount_rate == 0:
                bond_price = face_value + coupon_payment * periods
            else:
                pv_coupons = coupon_payment * (1 - pow(1 + discount_rate, -periods)) / discount_rate
                pv_principal = face_value / pow(1 + discount_rate, periods)
                bond_price = pv_coupons + pv_principal
            
            # Duration calculation (simplified Macaulay duration)
            cash_flows = []
            times = []
            present_values = []
            
            for t in range(1, periods + 1):
                if t < periods:
                    cf = coupon_payment
                else:
                    cf = coupon_payment + face_value
                
                pv = cf / pow(1 + discount_rate, t)
                weight = pv / bond_price
                time_years = t / 2  # Convert to years
                
                cash_flows.append(cf)
                times.append(time_years)
                present_values.append(pv)
            
            macaulay_duration = sum(pv * t for pv, t in zip([pv/bond_price for pv in present_values], times))
            modified_duration = macaulay_duration / (1 + ytm / 2)
            
            # Display metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Bond Price", f"${bond_price:.2f}")
                st.metric("Modified Duration", f"{modified_duration:.2f} years")
            
            with col_b:
                current_yield = (coupon_rate * face_value) / bond_price
                st.metric("Current Yield", f"{current_yield:.2%}")
                st.metric("Macaulay Duration", f"{macaulay_duration:.2f} years")
            
            # Price sensitivity chart
            yield_changes = np.linspace(-0.03, 0.03, 50)
            bond_prices = []
            duration_estimates = []
            
            for dy in yield_changes:
                new_ytm = ytm + dy
                new_discount_rate = new_ytm / 2
                
                if new_discount_rate == 0:
                    new_price = face_value + coupon_payment * periods
                else:
                    new_pv_coupons = coupon_payment * (1 - pow(1 + new_discount_rate, -periods)) / new_discount_rate
                    new_pv_principal = face_value / pow(1 + new_discount_rate, periods)
                    new_price = new_pv_coupons + new_pv_principal
                
                bond_prices.append(new_price)
                
                # Duration approximation
                duration_price = bond_price * (1 - modified_duration * dy)
                duration_estimates.append(duration_price)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=yield_changes * 100,
                y=bond_prices,
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=yield_changes * 100,
                y=duration_estimates,
                mode='lines',
                name='Duration Approximation',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Bond Price Sensitivity to Yield Changes',
                xaxis_title='Yield Change (%)',
                yaxis_title='Bond Price ($)',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Yield Curve
        st.subheader("📈 Yield Curve Analysis")
        
        st.markdown("""
        The yield curve shows the relationship between interest rates and time to maturity. 
        Common yield curve shapes include:
        """)
        
        shapes = [
            ("**Normal (Upward Sloping)**", "Long-term rates higher than short-term rates"),
            ("**Inverted (Downward Sloping)**", "Short-term rates higher than long-term rates"),
            ("**Flat**", "Similar rates across all maturities"),
            ("**Humped**", "Medium-term rates highest")
        ]
        
        for shape, description in shapes:
            st.markdown(f"• {shape}: {description}")
    
    with tab4:
        st.header("Mathematical Foundations")
        
        # Statistics Review
        st.subheader("📊 Statistical Concepts")
        
        st.markdown("**Probability Distributions:**")
        
        # Interactive distribution comparison
        col1, col2 = st.columns([1, 2])
        
        with col1:
            distribution = st.selectbox("Select Distribution", 
                                      ["Normal", "Student's t", "Log-normal"])
            if distribution == "Student's t":
                df = st.slider("Degrees of Freedom", 1, 30, 5)
            elif distribution == "Log-normal":
                sigma = st.slider("Sigma", 0.1, 2.0, 0.5, 0.1)
        
        with col2:
            x = np.linspace(-4, 4, 1000)
            
            if distribution == "Normal":
                from scipy.stats import norm
                y = norm.pdf(x, 0, 1)
                title = "Standard Normal Distribution"
            elif distribution == "Student's t":
                from scipy.stats import t
                y = t.pdf(x, df)
                title = f"Student's t Distribution (df={df})"
            else:  # Log-normal
                from scipy.stats import lognorm
                x_pos = np.linspace(0.1, 5, 1000)
                y = lognorm.pdf(x_pos, sigma)
                x = x_pos
                title = f"Log-normal Distribution (σ={sigma})"
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', 
                                   name=distribution,
                                   line=dict(width=3)))
            
            fig.update_layout(
                title=title,
                xaxis_title='Value',
                yaxis_title='Probability Density',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Optimization Methods
        st.subheader("🎯 Optimization Techniques")
        
        st.markdown("**Lagrange Multipliers for Portfolio Optimization:**")
        st.markdown("""
        <div class="formula-box">
        Minimize: ½ w'Σw
        Subject to: w'μ = μₚ and w'1 = 1
        
        Solution: w* = (Σ⁻¹μλ₁ + Σ⁻¹1λ₂) / (1'Σ⁻¹1)
        
        Where:
        • w = weight vector
        • Σ = covariance matrix
        • μ = expected return vector
        • λ₁, λ₂ = Lagrange multipliers
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Measures
        st.subheader("📏 Risk Measures Comparison")
        
        risk_measures = [
            {
                'Measure': 'Variance/Volatility',
                'Formula': 'σ² = E[(R - μ)²]',
                'Pros': 'Well-understood, mathematically tractable',
                'Cons': 'Treats upside and downside equally'
            },
            {
                'Measure': 'Value at Risk (VaR)',
                'Formula': 'P(Loss > VaR) = α',
                'Pros': 'Intuitive, regulatory standard',
                'Cons': 'Not subadditive, ignores tail losses'
            },
            {
                'Measure': 'Expected Shortfall',
                'Formula': 'ES = E[Loss | Loss > VaR]',
                'Pros': 'Coherent risk measure, captures tail risk',
                'Cons': 'More complex to compute and interpret'
            },
            {
                'Measure': 'Maximum Drawdown',
                'Formula': 'MDD = max(Peak - Trough) / Peak',
                'Pros': 'Shows worst-case scenario',
                'Cons': 'Backward-looking, path-dependent'
            }
        ]
        
        for measure in risk_measures:
            st.markdown(f"""
            <div class="method-comparison">
                <h4>{measure['Measure']}</h4>
                <p><strong>Formula:</strong> {measure['Formula']}</p>
                <p><strong>Pros:</strong> {measure['Pros']}</p>
                <p><strong>Cons:</strong> {measure['Cons']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Matrix Operations
        st.subheader("🔢 Key Matrix Operations")
        
        st.markdown("""
        **Covariance Matrix Properties:**
        - Symmetric: Σᵢⱼ = Σⱼᵢ
        - Positive semi-definite: w'Σw ≥ 0 for all w
        - Diagonal elements are variances: Σᵢᵢ = σᵢ²
        - Off-diagonal elements are covariances: Σᵢⱼ = σᵢⱼ
        """)
        
        st.markdown("""
        **Portfolio Variance Decomposition:**
        """)
        st.markdown("""
        <div class="formula-box">
        σₚ² = Σᵢ wᵢ²σᵢ² + Σᵢ≠ⱼ wᵢwⱼσᵢⱼ
        
        Individual Risk + Interaction Risk
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.header("References & Sources")
        
        # Academic References
        st.subheader("📚 Academic References")
        
        references = [
            {
                'category': 'Portfolio Theory',
                'papers': [
                    "Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.",
                    "Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. The Journal of Finance, 19(3), 425-442.",
                    "Tobin, J. (1958). Liquidity Preference as Behavior Towards Risk. The Review of Economic Studies, 25(2), 65-86.",
                    "Black, F., & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal, 48(5), 28-43."
                ]
            },
            {
                'category': 'Risk Management',
                'papers': [
                    "Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk. 3rd Edition, McGraw-Hill.",
                    "Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999). Coherent Measures of Risk. Mathematical Finance, 9(3), 203-228.",
                    "Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. The Journal of Derivatives, 3(2), 73-84."
                ]
            },
            {
                'category': 'Fixed Income',
                'papers': [
                    "Macaulay, F. R. (1938). Some Theoretical Problems Suggested by the Movements of Interest Rates. NBER.",
                    "Fisher, L., & Weil, R. L. (1971). Coping with the Risk of Interest-Rate Fluctuations. The Journal of Business, 44(4), 408-431.",
                    "Litterman, R., & Scheinkman, J. (1991). Common Factors Affecting Bond Returns. The Journal of Fixed Income, 1(1), 54-61."
                ]
            }
        ]
        
        for ref_group in references:
            st.markdown(f"**{ref_group['category']}:**")
            for paper in ref_group['papers']:
                st.markdown(f"• {paper}")
            st.markdown("")
        
        # Data Sources
        st.subheader("🌐 Data Sources Used")
        
        data_sources = [
            {
                'Source': 'Yahoo Finance (yfinance)',
                'Description': 'Historical stock prices, ETF data, and market indices',
                'URL': 'https://finance.yahoo.com',
                'Access': 'Free API via Python yfinance library'
            },
            {
                'Source': 'FRED (Federal Reserve Economic Data)',
                'Description': 'Treasury yields, economic indicators, and monetary data',
                'URL': 'https://fred.stlouisfed.org',
                'Access': 'Free API via pandas-datareader'
            },
            {
                'Source': 'Sample Portfolios',
                'Description': 'Curated portfolios for educational purposes',
                'URL': 'Internal',
                'Access': 'Pre-defined in application'
            }
        ]
        
        for source in data_sources:
            st.markdown(f"""
            <div class="method-comparison">
                <h4>{source['Source']}</h4>
                <p><strong>Description:</strong> {source['Description']}</p>
                <p><strong>URL:</strong> {source['URL']}</p>
                <p><strong>Access:</strong> {source['Access']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation Details
        st.subheader("🛠️ Implementation Details")
        
        implementation_notes = [
            "**Optimization Library**: SciPy's minimize function with SLSQP method",
            "**Statistical Calculations**: NumPy and Pandas for matrix operations",
            "**Visualization**: Plotly for interactive charts and analysis",
            "**Data Processing**: Pandas for time series analysis and data cleaning",
            "**Distribution Functions**: SciPy.stats for probability calculations"
        ]
        
        for note in implementation_notes:
            st.markdown(f"• {note}")
        
        # Assumptions and Limitations
        st.subheader("⚠️ Important Assumptions and Limitations")
        
        st.markdown("""
        <div class="assumption-box">
        <h4>🔴 Key Limitations to Remember:</h4>
        <ul>
            <li><strong>Historical Data Dependency:</strong> All models rely on historical data which may not predict future performance</li>
            <li><strong>Normal Distribution Assumptions:</strong> Many models assume normal distributions, but real returns often have fat tails</li>
            <li><strong>Constant Parameters:</strong> Models assume correlations and volatilities remain constant over time</li>
            <li><strong>No Transaction Costs:</strong> Real trading involves costs that can significantly impact performance</li>
            <li><strong>Liquidity Assumptions:</strong> Models assume assets can be bought/sold instantly at market prices</li>
            <li><strong>Model Risk:</strong> All models are simplifications and may not capture all market dynamics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Further Reading
        st.subheader("📖 Recommended Further Reading")
        
        books = [
            "**'Quantitative Portfolio Management'** by Chincarini & Kim - Comprehensive coverage of modern portfolio theory",
            "**'Active Portfolio Management'** by Grinold & Kahn - Advanced techniques for active management",
            "**'Risk Management and Financial Institutions'** by John Hull - Practical risk management approaches",
            "**'Fixed Income Mathematics'** by Frank Fabozzi - Detailed bond mathematics and analytics",
            "**'The Handbook of Portfolio Mathematics'** by Ralph Vince - Mathematical foundations of portfolio construction"
        ]
        
        for book in books:
            st.markdown(f"• {book}")
        
        # Online Resources
        st.subheader("🌍 Online Resources")
        
        online_resources = [
            "**CFA Institute**: Professional standards and educational materials",
            "**Quantlib**: Open-source quantitative finance library",
            "**SSRN**: Academic papers and research in finance",
            "**Federal Reserve Publications**: Economic research and data",
            "**Bank for International Settlements**: Risk management guidelines"
        ]
        
        for resource in online_resources:
            st.markdown(f"• {resource}")
        
        # Contact and Feedback
        st.subheader("💬 Feedback and Contributions")
        
        st.info("""
        This educational platform is designed to help users understand the theoretical foundations 
        behind portfolio optimization and risk management. For feedback, suggestions, or contributions:
        
        • Review the methodology assumptions carefully before applying to real portfolios
        • Consider consulting with financial professionals for investment decisions
        • Use this tool for educational and research purposes
        • Remember that past performance does not guarantee future results
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Educational Hub • Theory & Practice • Continuous Learning</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
