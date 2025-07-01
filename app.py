import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add the models directory to the path
sys.path.append('models')

# Import your existing models
try:
    from models.black_scholes.sde import BlackScholesSDE
    from models.black_scholes.pde import BlackScholesPDE
    from models.stochastic_volatility.heston import HestonModel
    from models.stochastic_volatility.sabr import SABRModel
    from models.jump_models.jump_diffusion import JumpDiffusionModel
    from models.local_volatility.local_vol import LocalVolatilityModel
    from models.hull_white.hull_white import HullWhiteModel
    from models.hull_white.extensions import HullWhiteExtensions
except ImportError as e:
    st.error(f"Model import error: {e}")
    st.info("Please ensure all __init__.py files are created in the model directories")

# Black-Scholes Option Pricing Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calculate option Greeks (sensitivities)"""
    if T <= 0:
        return {
            'call_delta': 1 if S > K else 0, 'put_delta': -1 if S < K else 0,
            'gamma': 0, 'call_theta': 0, 'put_theta': 0,
            'vega': 0, 'call_rho': 0, 'put_rho': 0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta: price sensitivity to underlying price
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    
    # Gamma: delta sensitivity to underlying price
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta: price sensitivity to time decay (per day)
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega: price sensitivity to volatility (per 1% vol change)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho: price sensitivity to interest rate (per 1% rate change)
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'call_theta': call_theta, 'put_theta': put_theta,
        'vega': vega, 'call_rho': call_rho, 'put_rho': put_rho
    }

# Utility function for plotting paths
def plot_simulation_paths(time_steps, paths, title, ylabel="Price", show_mean=True, max_displayed_paths=50):
    """Create an interactive plot for Monte Carlo simulation paths"""
    fig = go.Figure()
    
    # Display a subset of paths for better performance
    n_paths_to_show = min(max_displayed_paths, paths.shape[1])
    for i in range(n_paths_to_show):
        fig.add_trace(go.Scatter(
            x=time_steps, y=paths[:, i],
            mode='lines', line=dict(width=0.8, color='lightblue'),
            showlegend=False, opacity=0.6,
            hovertemplate=f'Path {i+1}<br>Time: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>'
        ))
    
    # Add mean path if requested and we have multiple paths
    if show_mean and paths.shape[1] > 1:
        mean_values = np.mean(paths, axis=1)
        fig.add_trace(go.Scatter(
            x=time_steps, y=mean_values,
            mode='lines', line=dict(color='red', width=3),
            name='Average Path',
            hovertemplate=f'Average<br>Time: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (years)",
        yaxis_title=ylabel,
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="Quantitative Finance Models", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Quantitative Finance Models Dashboard")
    st.markdown("An interactive exploration of mathematical models in finance using your existing implementations")
    
    # Sidebar navigation
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a financial model to explore:",
        [
            "Welcome",
            "Black-Scholes SDE", 
            "Black-Scholes PDE", 
            "Heston Stochastic Volatility", 
            "SABR Model",
            "Merton Jump Diffusion", 
            "Local Volatility", 
            "Hull-White Interest Rate", 
            "Option Pricing & Greeks"
        ]
    )
    
    if model_choice == "Welcome":
        show_welcome_page()
    elif model_choice == "Black-Scholes SDE":
        show_black_scholes_sde()
    elif model_choice == "Black-Scholes PDE":
        show_black_scholes_pde()
    elif model_choice == "Heston Stochastic Volatility":
        show_heston_model()
    elif model_choice == "SABR Model":
        show_sabr_model()
    elif model_choice == "Merton Jump Diffusion":
        show_jump_diffusion()
    elif model_choice == "Local Volatility":
        show_local_volatility()
    elif model_choice == "Hull-White Interest Rate":
        show_hull_white()
    elif model_choice == "Option Pricing & Greeks":
        show_option_pricing()

def show_welcome_page():
    st.header("Welcome to the Quantitative Finance Dashboard")
    
    st.markdown("""
    This dashboard brings your quantitative finance models to life through interactive simulations 
    and visualizations. Each model represents decades of mathematical finance research, now at your fingertips.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Models")
        st.markdown("""
        **Asset Price Models:**
        - **Black-Scholes SDE**: The foundation of modern options theory
        - **Black-Scholes PDE**: Numerical solution to the famous partial differential equation
        - **Heston Model**: Captures the stochastic nature of volatility
        - **SABR Model**: Popular for interest rate derivatives
        - **Jump Diffusion**: Accounts for sudden market movements
        - **Local Volatility**: State-dependent volatility modeling
        
        **Interest Rate Models:**
        - **Hull-White**: Mean-reverting short rate model
        
        **Options Analysis:**
        - **Complete Greeks**: All option sensitivities in one place
        """)
    
    with col2:
        st.subheader("What Makes This Special")
        st.markdown("""
        **Real Implementation**: These aren't toy models - they use your actual production code
        
        **Interactive Parameters**: Adjust model parameters and see immediate results
        
        **Professional Visualizations**: Publication-quality charts that reveal model behavior
        
        **Educational Value**: Understand how changing parameters affects outcomes
        
        **Monte Carlo Power**: Run thousands of simulations to understand probability distributions
        """)
    
    st.subheader("Getting Started")
    st.markdown("""
    1. **Select a model** from the sidebar - each has its own story to tell
    2. **Adjust parameters** to see how sensitive the model is to your assumptions  
    3. **Run simulations** and watch the mathematical magic unfold
    4. **Explore results** with interactive charts that respond to your curiosity
    
    *Tip: Start with Black-Scholes SDE if you're new to quantitative finance - it's the gateway to understanding all the others.*
    """)
    
    # Show model structure
    with st.expander("Model Code Structure"):
        st.code("""
models/
├── black_scholes/
│   ├── sde.py          # Stochastic Differential Equation implementation
│   └── pde.py          # Partial Differential Equation solver
├── stochastic_volatility/
│   ├── heston.py       # Heston stochastic volatility model
│   └── sabr.py         # SABR model for interest rates
├── jump_models/
│   └── jump_diffusion.py  # Merton jump diffusion model
├── local_volatility/
│   └── local_vol.py    # Local volatility (Dupire) model
└── hull_white/
    ├── hull_white.py   # Hull-White interest rate model
    └── extensions.py   # Additional Hull-White features
        """)

def show_black_scholes_sde():
    st.header("Black-Scholes Stochastic Differential Equation")
    st.markdown("""
    The Black-Scholes SDE describes how stock prices evolve through time, combining 
    deterministic drift with random volatility. This is the mathematical foundation 
    that revolutionized options trading.
    """)
    
    st.latex(r'dS_t = \mu S_t dt + \sigma S_t dW_t')
    st.code("from models.black_scholes.sde import BlackScholesSDE")
    
    # Common simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("Number of simulation paths", 10, 1000, 100, 10)
    T = st.sidebar.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        spot = st.number_input("Initial stock price ($)", value=100.0, min_value=1.0, step=1.0)
        mu = st.slider("Drift rate", -0.5, 0.5, 0.05, 0.01)
        sigma = st.slider("Volatility", 0.01, 1.0, 0.2, 0.01)
        dt_choice = st.selectbox("Time step frequency", [1/252, 1/52, 1/12], index=0, 
                               format_func=lambda x: f"Daily (1/{int(1/x)})" if x == 1/252 else f"1/{int(1/x)} periods per year")
        
        run_simulation = st.button("Run Monte Carlo Simulation", type="primary")
    
    if run_simulation:
        try:
            with st.spinner("Running simulation..."):
                model = BlackScholesSDE(spot, mu, sigma, dt_choice)
                paths = model.simulate_paths(T, n_paths)
                
                time_steps = np.linspace(0, T, paths.shape[0])
                
                with col2:
                    fig = plot_simulation_paths(time_steps, paths, 
                                              f"Black-Scholes Stock Price Paths ({n_paths:,} simulations)", 
                                              "Stock Price ($)")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Statistical analysis
            st.subheader("Simulation Results")
            final_prices = paths[-1, :]
            
            # Key statistics
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Average final price", f"${np.mean(final_prices):.2f}")
            with stats_col2:
                st.metric("Standard deviation", f"${np.std(final_prices):.2f}")
            with stats_col3:
                st.metric("Minimum price", f"${np.min(final_prices):.2f}")
            with stats_col4:
                st.metric("Maximum price", f"${np.max(final_prices):.2f}")
            
            # Distribution analysis
            col_hist1, col_hist2 = st.columns(2)
            
            with col_hist1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=final_prices, nbinsx=50, name="Final price distribution",
                                              marker_color='lightblue', opacity=0.7))
                fig_hist.update_layout(title="Distribution of Final Stock Prices", 
                                     xaxis_title="Final Price ($)", yaxis_title="Frequency",
                                     template='plotly_white')
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_hist2:
                # Calculate probabilities
                prob_above_initial = np.mean(final_prices > spot) * 100
                prob_double = np.mean(final_prices > 2 * spot) * 100
                prob_loss_50 = np.mean(final_prices < 0.5 * spot) * 100
                
                st.markdown("**Probability Analysis:**")
                st.write(f"• Probability of gain: {prob_above_initial:.1f}%")
                st.write(f"• Probability of doubling: {prob_double:.1f}%")  
                st.write(f"• Probability of 50%+ loss: {prob_loss_50:.1f}%")
                
                theoretical_mean = spot * np.exp(mu * T)
                theoretical_std = spot * np.exp(mu * T) * np.sqrt(np.exp(sigma**2 * T) - 1)
                
                st.markdown("**Theoretical vs Simulated:**")
                st.write(f"• Expected final price: ${theoretical_mean:.2f}")
                st.write(f"• Simulated mean: ${np.mean(final_prices):.2f}")
                st.write(f"• Expected std dev: ${theoretical_std:.2f}")
                st.write(f"• Simulated std dev: ${np.std(final_prices):.2f}")
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")

def show_black_scholes_pde():
    st.header("Black-Scholes Partial Differential Equation Solver")
    st.markdown("""
    Rather than simulating random paths, we can solve the Black-Scholes PDE directly 
    to find the exact option price. This numerical approach shows how the mathematical 
    theory translates into practical computation.
    """)
    
    st.latex(r'\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0')
    st.code("from models.black_scholes.pde import BlackScholesPDE")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("PDE Parameters")
        S_max = st.number_input("Maximum asset price for grid", value=200.0, min_value=50.0, step=10.0)
        K = st.number_input("Strike price", value=100.0, min_value=1.0, step=1.0)
        T_pde = st.slider("Time to maturity", 0.1, 2.0, 0.25, 0.05)
        r = st.slider("Risk-free interest rate", 0.0, 0.2, 0.05, 0.001)
        sigma = st.slider("Volatility", 0.01, 1.0, 0.2, 0.01)
        
        st.subheader("Numerical Settings")
        M = st.slider("Price grid points", 50, 200, 100, 10)
        N = st.slider("Time steps", 100, 2000, 1000, 100)
        
        solve_pde = st.button("Solve PDE Numerically", type="primary")
    
    if solve_pde:
        try:
            with st.spinner("Solving the Black-Scholes PDE..."):
                pde_solver = BlackScholesPDE(S_max, K, T_pde, r, sigma, M, N)
                S, V = pde_solver.solve_call()
                
                with col2:
                    # Option value surface
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=S, y=V, mode='lines', name='Call option value (PDE)',
                                           line=dict(color='blue', width=3)))
                    fig.add_vline(x=K, line_dash="dash", annotation_text="Strike price", 
                                line_color="red")
                    
                    # Add intrinsic value for comparison
                    intrinsic = np.maximum(S - K, 0)
                    fig.add_trace(go.Scatter(x=S, y=intrinsic, mode='lines', name='Intrinsic value',
                                           line=dict(color='gray', dash='dot')))
                    
                    fig.update_layout(
                        title="Call Option Value vs Stock Price",
                        xaxis_title="Stock Price ($)", yaxis_title="Option Value ($)", 
                        height=500, template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Compare with analytical solution
            st.subheader("Numerical vs Analytical Comparison")
            analytical_prices = [black_scholes_call(s, K, T_pde, r, sigma) for s in S]
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=S, y=V, name='PDE Solution', 
                                            line=dict(color='blue', width=2)))
                fig_comp.add_trace(go.Scatter(x=S, y=analytical_prices, name='Analytical Formula', 
                                            line=dict(color='red', dash='dash', width=2)))
                fig_comp.update_layout(
                    title="PDE vs Analytical Solution", 
                    xaxis_title="Stock Price ($)", yaxis_title="Option Value ($)",
                    template='plotly_white'
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col_comp2:
                # Error analysis
                error = np.abs(np.array(V) - np.array(analytical_prices))
                max_error = np.max(error)
                mean_error = np.mean(error)
                rmse = np.sqrt(np.mean(error**2))
                
                st.markdown("**Accuracy Metrics:**")
                st.write(f"• Maximum absolute error: ${max_error:.6f}")
                st.write(f"• Mean absolute error: ${mean_error:.6f}")
                st.write(f"• Root mean square error: ${rmse:.6f}")
                
                # Error plot
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(x=S, y=error, mode='lines',
                                             line=dict(color='orange', width=2)))
                fig_error.update_layout(
                    title="Numerical Error vs Stock Price",
                    xaxis_title="Stock Price ($)", yaxis_title="Absolute Error ($)",
                    template='plotly_white'
                )
                st.plotly_chart(fig_error, use_container_width=True)
                
        except Exception as e:
            st.error(f"PDE solution failed: {e}")

def show_heston_model():
    st.header("Heston Stochastic Volatility Model")
    st.markdown("""
    The Heston model captures a crucial market reality: volatility itself is random. 
    This creates more realistic price dynamics and explains phenomena like volatility clustering 
    and the volatility smile in options markets.
    """)
    
    st.latex(r'''
    \begin{align}
    dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_1^t \\
    dv_t &= \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_2^t \\
    d\langle W_1, W_2 \rangle_t &= \rho dt
    \end{align}
    ''')
    st.code("from models.stochastic_volatility.heston import HestonModel")
    
    # Common simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("Number of simulation paths", 10, 1000, 100, 10)
    T = st.sidebar.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        spot = st.number_input("Initial stock price", value=100.0, min_value=1.0, step=1.0)
        mu = st.slider("Stock drift rate", -0.5, 0.5, 0.05, 0.01)
        
        st.markdown("**Volatility Process:**")
        v0 = st.slider("Initial variance", 0.01, 1.0, 0.04, 0.01)
        kappa = st.slider("Mean reversion speed", 0.1, 10.0, 2.0, 0.1)
        theta = st.slider("Long-term variance", 0.01, 1.0, 0.04, 0.01)
        xi = st.slider("Volatility of volatility", 0.01, 2.0, 0.3, 0.01)
        rho = st.slider("Correlation", -1.0, 1.0, -0.7, 0.1)
        
        dt_choice = st.selectbox("Time step", [1/252, 1/52], index=0, 
                               format_func=lambda x: "Daily" if x == 1/252 else "Weekly")
        
        run_heston = st.button("Run Heston Simulation", type="primary")
    
    if run_heston:
        try:
            with st.spinner("Simulating correlated stock and volatility paths..."):
                model = HestonModel(spot, mu, v0, kappa, theta, xi, rho, dt_choice)
                S_paths, v_paths = model.simulate_paths(T, n_paths)
                
                time_steps = np.linspace(0, T, S_paths.shape[0])
                
                with col2:
                    # Create dual-axis subplots
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Stock Price Evolution', 'Variance Evolution'),
                        vertical_spacing=0.12
                    )
                    
                    # Stock price paths
                    sample_size = min(25, n_paths)
                    for i in range(sample_size):
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=S_paths[:, i],
                            mode='lines', line=dict(width=0.7, color='lightblue'),
                            showlegend=False, opacity=0.6
                        ), row=1, col=1)
                    
                    # Variance paths
                    for i in range(sample_size):
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=v_paths[:, i],
                            mode='lines', line=dict(width=0.7, color='lightcoral'),
                            showlegend=False, opacity=0.6
                        ), row=2, col=1)
                    
                    # Mean paths for reference
                    fig.add_trace(go.Scatter(
                        x=time_steps, y=np.mean(S_paths, axis=1),
                        mode='lines', line=dict(color='darkblue', width=3),
                        name='Average price', showlegend=True
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=time_steps, y=np.mean(v_paths, axis=1),
                        mode='lines', line=dict(color='darkred', width=3),
                        name='Average variance', showlegend=True
                    ), row=2, col=1)
                    
                    fig.update_layout(height=700, title="Heston Model: Stock Price and Variance Dynamics",
                                    template='plotly_white')
                    fig.update_xaxes(title_text="Time (years)", row=2, col=1)
                    fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="Variance", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Statistical analysis
            st.subheader("Model Insights")
            final_prices = S_paths[-1, :]
            final_variances = v_paths[-1, :]
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.markdown("**Price Statistics:**")
                st.write(f"• Average final price: ${np.mean(final_prices):.2f}")
                st.write(f"• Price volatility: {np.std(final_prices)/np.mean(final_prices)*100:.1f}%")
                st.write(f"• Price range: ${np.min(final_prices):.2f} - ${np.max(final_prices):.2f}")
            
            with insight_col2:
                st.markdown("**Volatility Statistics:**")
                final_vol = np.sqrt(final_variances)
                st.write(f"• Average final volatility: {np.mean(final_vol):.1%}")
                st.write(f"• Volatility of volatility: {np.std(final_vol):.1%}")
                st.write(f"• Long-term target: {np.sqrt(theta):.1%}")
            
            with insight_col3:
                st.markdown("**Correlation Effects:**")
                realized_corr = np.corrcoef(
                    np.diff(S_paths, axis=0).flatten(), 
                    np.diff(v_paths, axis=0).flatten()
                )[0,1]
                st.write(f"• Target correlation: {rho:.2f}")
                st.write(f"• Realized correlation: {realized_corr:.2f}")
                
                leverage_effect = "Negative" if rho < 0 else "Positive" if rho > 0 else "None"
                st.write(f"• Leverage effect: {leverage_effect}")
                
        except Exception as e:
            st.error(f"Heston simulation failed: {e}")

def show_sabr_model():
    st.header("SABR Stochastic Alpha Beta Rho Model")
    st.markdown("""
    SABR is the industry standard for modeling interest rate derivatives. It's particularly 
    good at capturing the volatility smile and is widely used for pricing caps, floors, 
    and swaptions in fixed income markets.
    """)
    
    st.latex(r'''
    \begin{align}
    dF_t &= \sigma_t F_t^\beta dW_1^t \\
    d\sigma_t &= \nu \sigma_t dW_2^t \\
    d\langle W_1, W_2 \rangle_t &= \rho dt
    \end{align}
    ''')
    st.code("from models.stochastic_volatility.sabr import SABRModel")
    
    # Common simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("Number of simulation paths", 10, 1000, 100, 10)
    T = st.sidebar.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("SABR Parameters")
        F0 = st.number_input("Initial forward rate", value=0.05, min_value=0.001, max_value=0.5, 
                            step=0.001, format="%.3f")
        alpha = st.slider("Initial volatility", 0.01, 1.0, 0.2, 0.01)
        beta = st.slider("Elasticity parameter", 0.0, 1.0, 0.5, 0.1)
        rho = st.slider("Correlation", -1.0, 1.0, -0.3, 0.1)
        nu = st.slider("Volatility of volatility", 0.01, 2.0, 0.3, 0.01)
        
        dt_choice = st.selectbox("Time step", [1/252, 1/52], index=0, 
                               format_func=lambda x: "Daily" if x == 1/252 else "Weekly")
        
        run_sabr = st.button("Run SABR Simulation", type="primary")
    
    if run_sabr:
        try:
            with st.spinner("Simulating SABR forward rate and volatility dynamics..."):
                model = SABRModel(F0, alpha, beta, rho, nu, dt_choice)
                F_paths, sigma_paths = model.simulate_paths(T, n_paths)
                
                time_steps = np.linspace(0, T, F_paths.shape[0])
                
                with col2:
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Forward Rate Evolution', 'Volatility Evolution'),
                        vertical_spacing=0.12
                    )
                    
                    # Forward rate paths
                    sample_size = min(25, n_paths)
                    for i in range(sample_size):
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=F_paths[:, i] * 100,
                            mode='lines', line=dict(width=0.7, color='lightgreen'),
                            showlegend=False, opacity=0.6
                        ), row=1, col=1)
                    
                    # Volatility paths
                    for i in range(sample_size):
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=sigma_paths[:, i],
                            mode='lines', line=dict(width=0.7, color='lightcoral'),
                            showlegend=False, opacity=0.6
                        ), row=2, col=1)
                    
                    # Average paths
                    fig.add_trace(go.Scatter(
                        x=time_steps, y=np.mean(F_paths, axis=1) * 100,
                        mode='lines', line=dict(color='darkgreen', width=3),
                        name='Average forward rate', showlegend=True
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=time_steps, y=np.mean(sigma_paths, axis=1),
                        mode='lines', line=dict(color='darkred', width=3),
                        name='Average volatility', showlegend=True
                    ), row=2, col=1)
                    
                    fig.update_layout(height=700, title="SABR Model: Forward Rate and Volatility Dynamics",
                                    template='plotly_white')
                    fig.update_xaxes(title_text="Time (years)", row=2, col=1)
                    fig.update_yaxes(title_text="Forward Rate (%)", row=1, col=1)
                    fig.update_yaxes(title_text="Volatility", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            # Analysis specific to SABR
            st.subheader("SABR Model Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                final_forwards = F_paths[-1, :] * 100
                final_vols = sigma_paths[-1, :]
                
                st.markdown("**Forward Rate Statistics:**")
                st.write(f"• Initial rate: {F0*100:.3f}%")
                st.write(f"• Average final rate: {np.mean(final_forwards):.3f}%")
                st.write(f"• Rate standard deviation: {np.std(final_forwards):.3f}%")
                st.write(f"• Rate range: {np.min(final_forwards):.3f}% - {np.max(final_forwards):.3f}%")
            
            with analysis_col2:
                st.markdown("**Volatility Behavior:**")
                st.write(f"• Initial volatility: {alpha:.3f}")
                st.write(f"• Average final volatility: {np.mean(final_vols):.3f}")
                st.write(f"• Volatility standard deviation: {np.std(final_vols):.3f}")
                st.write(f"• Beta elasticity effect: {'High' if beta > 0.7 else 'Medium' if beta > 0.3 else 'Low'}")
                
        except Exception as e:
            st.error(f"SABR simulation failed: {e}")

def show_jump_diffusion():
    st.header("Merton Jump Diffusion Model")
    st.markdown("""
    Markets don't always move smoothly. The Merton model adds sudden jumps to the 
    traditional Black-Scholes framework, capturing events like earnings announcements, 
    market crashes, or breaking news that cause discontinuous price movements.
    """)
    
    st.latex(r'\frac{dS_t}{S_t} = (\mu - \lambda k) dt + \sigma dW_t + J dN_t')
    st.markdown("where $N_t$ is a Poisson process with intensity $\lambda$ and $J$ represents jump sizes")
    st.code("from models.jump_models.jump_diffusion import JumpDiffusionModel")
    
    # Common simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("Number of simulation paths", 10, 1000, 100, 10)
    T = st.sidebar.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        spot = st.number_input("Initial stock price", value=100.0, min_value=1.0, step=1.0)
        mu = st.slider("Drift rate", -0.5, 0.5, 0.05, 0.01)
        sigma = st.slider("Diffusion volatility", 0.01, 1.0, 0.2, 0.01)
        
        st.markdown("**Jump Parameters:**")
        lamb = st.slider("Jump intensity", 0.0, 10.0, 1.0, 0.1)
        mu_j = st.slider("Average jump size", -0.5, 0.5, -0.1, 0.01)
        sigma_j = st.slider("Jump size volatility", 0.01, 1.0, 0.3, 0.01)
        
        dt_choice = st.selectbox("Time step", [1/252, 1/52], index=0, 
                               format_func=lambda x: "Daily" if x == 1/252 else "Weekly")
        
        run_jump = st.button("Run Jump Diffusion Simulation", type="primary")
    
    if run_jump:
        try:
            with st.spinner("Simulating stock paths with random jumps..."):
                model = JumpDiffusionModel(spot, mu, sigma, lamb, mu_j, sigma_j, dt_choice)
                paths = model.simulate_paths(T, n_paths)
                
                time_steps = np.linspace(0, T, paths.shape[0])
                
                with col2:
                    fig = plot_simulation_paths(time_steps, paths, 
                                              f"Jump Diffusion Paths ({n_paths:,} simulations)", 
                                              "Stock Price ($)")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Jump analysis
            st.subheader("Jump Analysis")
            
            jump_col1, jump_col2 = st.columns(2)
            
            with jump_col1:
                # Calculate daily returns to identify jumps
                returns = np.diff(np.log(paths), axis=0)
                
                # Simple jump detection: returns > 3 standard deviations
                return_std = np.std(returns)
                jump_threshold = 3 * return_std
                
                large_moves = np.abs(returns) > jump_threshold
                
                st.markdown("**Jump Statistics:**")
                st.write(f"• Expected jumps per year: {lamb:.1f}")
                st.write(f"• Observed large moves per path: {np.sum(large_moves)/n_paths:.2f}")
                st.write(f"• Jump detection threshold: ±{jump_threshold:.3f}")
                st.write(f"• Average jump size (when they occur): {mu_j:.2f}")
                
                # Expected vs actual final distribution
                final_prices = paths[-1, :]
                st.write(f"• Price range: ${np.min(final_prices):.2f} - ${np.max(final_prices):.2f}")
            
            with jump_col2:
                # Distribution comparison
                fig_dist = go.Figure()
                
                # Histogram of final prices
                fig_dist.add_trace(go.Histogram(x=final_prices, nbinsx=50, 
                                              name="Jump diffusion", opacity=0.7,
                                              marker_color='orange'))
                
                # Compare with pure Black-Scholes (no jumps)
                bs_model = BlackScholesSDE(spot, mu + lamb * mu_j, sigma, dt_choice)
                bs_paths = bs_model.simulate_paths(T, n_paths)
                bs_final = bs_paths[-1, :]
                
                fig_dist.add_trace(go.Histogram(x=bs_final, nbinsx=50, 
                                              name="Pure Black-Scholes", opacity=0.5,
                                              marker_color='blue'))
                
                fig_dist.update_layout(title="Final Price Distribution: Jump vs No-Jump",
                                     xaxis_title="Final Price ($)", yaxis_title="Frequency",
                                     template='plotly_white', barmode='overlay')
                st.plotly_chart(fig_dist, use_container_width=True)
                
        except Exception as e:
            st.error(f"Jump diffusion simulation failed: {e}")

def show_local_volatility():
    st.header("Local Volatility Model (Dupire)")
    st.markdown("""
    Local volatility models allow volatility to depend on both the current stock price 
    and time. This flexibility helps match observed market prices of options across 
    different strikes and maturities, creating a more realistic pricing framework.
    """)
    
    st.latex(r'dS_t = \sigma(S_t, t) S_t dW_t')
    st.code("from models.local_volatility.local_vol import LocalVolatilityModel")
    
    # Common simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("Number of simulation paths", 10, 1000, 100, 10)
    T = st.sidebar.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        spot = st.number_input("Initial stock price", value=100.0, min_value=1.0, step=1.0)
        
        volatility_type = st.selectbox("Volatility surface type", 
                                     ["Constant", "Time-dependent", "Level-dependent", "Both time and level"])
        
        if volatility_type == "Constant":
            base_vol = st.slider("Constant volatility", 0.1, 1.0, 0.2, 0.01)
            vol_surface = lambda S, t: np.full_like(S, base_vol) if hasattr(S, '__len__') else base_vol
            
        elif volatility_type == "Time-dependent":
            base_vol = st.slider("Base volatility", 0.1, 1.0, 0.2, 0.01)
            time_decay = st.slider("Time decay factor", -2.0, 2.0, -0.5, 0.1)
            vol_surface = lambda S, t: base_vol * np.exp(time_decay * t)
            
        elif volatility_type == "Level-dependent":
            base_vol = st.slider("Base volatility", 0.1, 1.0, 0.2, 0.01)
            level_slope = st.slider("Level sensitivity", -2.0, 2.0, 0.5, 0.1)
            vol_surface = lambda S, t: base_vol * (1 + level_slope * (S / spot - 1))
            
        else:  # Both time and level dependent
            base_vol = st.slider("Base volatility", 0.1, 1.0, 0.2, 0.01)
            time_decay = st.slider("Time decay", -1.0, 1.0, -0.2, 0.1)
            level_slope = st.slider("Level sensitivity", -1.0, 1.0, 0.3, 0.1)
            vol_surface = lambda S, t: base_vol * np.exp(time_decay * t) * (1 + level_slope * (S / spot - 1))
        
        dt_choice = st.selectbox("Time step", [1/252, 1/52], index=0, 
                               format_func=lambda x: "Daily" if x == 1/252 else "Weekly")
        
        run_local_vol = st.button("Run Local Volatility Simulation", type="primary")
    
    if run_local_vol:
        try:
            with st.spinner("Simulating paths with state-dependent volatility..."):
                model = LocalVolatilityModel(spot, vol_surface, dt_choice)
                paths = model.simulate_paths(T, n_paths)
                
                time_steps = np.linspace(0, T, paths.shape[0])
                
                with col2:
                    fig = plot_simulation_paths(time_steps, paths, 
                                              f"Local Volatility Paths ({n_paths:,} simulations)", 
                                              "Stock Price ($)")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Volatility surface visualization
            st.subheader("Local Volatility Surface")
            
            surface_col1, surface_col2 = st.columns(2)
            
            with surface_col1:
                # Create surface data
                S_range = np.linspace(spot * 0.5, spot * 1.5, 50)
                t_range = np.linspace(0.01, T, 50)
                S_grid, t_grid = np.meshgrid(S_range, t_range)
                
                # Calculate volatility surface
                vol_grid = np.zeros_like(S_grid)
                for i in range(len(t_range)):
                    for j in range(len(S_range)):
                        vol_grid[i, j] = vol_surface(S_range[j], t_range[i])
                
                # Create 3D surface plot
                fig_surface = go.Figure(data=[go.Surface(z=vol_grid, x=S_grid, y=t_grid,
                                                       colorscale='Viridis')])
                fig_surface.update_layout(
                    title='Local Volatility Surface σ(S,t)',
                    scene=dict(
                        xaxis_title='Stock Price ($)',
                        yaxis_title='Time (years)',
                        zaxis_title='Volatility σ(S,t)'
                    ),
                    height=500
                )
                st.plotly_chart(fig_surface, use_container_width=True)
            
            with surface_col2:
                # Slice through the surface
                mid_time = T / 2
                vol_at_mid_time = [vol_surface(S, mid_time) for S in S_range]
                
                fig_slice = go.Figure()
                fig_slice.add_trace(go.Scatter(x=S_range, y=vol_at_mid_time,
                                             mode='lines', line=dict(width=3, color='blue'),
                                             name=f'Volatility at t={mid_time:.2f}'))
                fig_slice.add_vline(x=spot, line_dash="dash", annotation_text="Initial price")
                fig_slice.update_layout(
                    title=f"Volatility Smile at t = {mid_time:.2f} years",
                    xaxis_title="Stock Price ($)",
                    yaxis_title="Local Volatility",
                    template='plotly_white'
                )
                st.plotly_chart(fig_slice, use_container_width=True)
                
                # Statistical analysis
                final_prices = paths[-1, :]
                st.markdown("**Path Statistics:**")
                st.write(f"• Average final price: ${np.mean(final_prices):.2f}")
                st.write(f"• Price standard deviation: ${np.std(final_prices):.2f}")
                st.write(f"• Skewness: {pd.Series(final_prices).skew():.3f}")
                st.write(f"• Kurtosis: {pd.Series(final_prices).kurtosis():.3f}")
                
        except Exception as e:
            st.error(f"Local volatility simulation failed: {e}")

def show_hull_white():
    st.header("Hull-White One-Factor Interest Rate Model")
    st.markdown("""
    Interest rates are mean-reverting - they don't trend indefinitely up or down but 
    tend to return to long-term averages. The Hull-White model captures this behavior 
    and is fundamental for pricing interest rate derivatives and managing duration risk.
    """)
    
    st.latex(r'dr_t = [\theta(t) - a \cdot r_t] dt + \sigma dW_t')
    st.code("from models.hull_white.hull_white import HullWhiteModel")
    
    # Common simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("Number of simulation paths", 10, 1000, 100, 10)
    T = st.sidebar.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        r0 = st.slider("Initial interest rate", 0.0, 0.15, 0.05, 0.001, format="%.3f")
        a = st.slider("Mean reversion speed", 0.1, 5.0, 1.0, 0.1)
        sigma = st.slider("Interest rate volatility", 0.001, 0.1, 0.02, 0.001, format="%.3f")
        theta = st.slider("Long-term rate level", 0.0, 0.15, 0.05, 0.001, format="%.3f")
        
        dt_choice = st.selectbox("Time step", [1/252, 1/52], index=0, 
                               format_func=lambda x: "Daily" if x == 1/252 else "Weekly")
        
        run_hw = st.button("Run Interest Rate Simulation", type="primary")
    
    if run_hw:
        try:
            with st.spinner("Simulating interest rate paths with mean reversion..."):
                model = HullWhiteModel(r0, a, sigma, theta, dt_choice)
                rates = model.simulate_short_rate(T, n_paths)
                
                time_steps = np.linspace(0, T, rates.shape[0])
                
                with col2:
                    # Convert to percentage for display
                    rates_percent = rates * 100
                    fig = plot_simulation_paths(time_steps, rates_percent, 
                                              f"Hull-White Interest Rate Paths ({n_paths:,} simulations)", 
                                              "Interest Rate (%)")
                    
                    # Add long-term mean line
                    fig.add_hline(y=theta*100, line_dash="dash", line_color="red",
                                annotation_text="Long-term mean")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Bond pricing analysis
            st.subheader("Zero-Coupon Bond Analysis")
            
            bond_col1, bond_col2, bond_col3 = st.columns(3)
            
            with bond_col1:
                st.markdown("**Current Conditions:**")
                current_rate = st.slider("Current short rate", 0.0, 0.15, r0, 0.001, format="%.3f")
                bond_maturity = st.slider("Bond maturity (years)", 0.1, 10.0, 2.0, 0.1)
                current_time = st.slider("Current time", 0.0, min(T, bond_maturity-0.1), 0.0, 0.1)
            
            # Calculate bond price and yield
            bond_price = model.zero_coupon_bond_price(current_rate, bond_maturity, current_time)
            bond_yield = -np.log(bond_price) / (bond_maturity - current_time)
            
            with bond_col2:
                st.markdown("**Bond Valuation:**")
                st.write(f"• Bond price: ${bond_price:.4f}")
                st.write(f"• Bond yield: {bond_yield*100:.3f}%")
                st.write(f"• Yield spread: {(bond_yield - current_rate)*100:.1f} bps")
                
                # Duration approximation
                duration_approx = bond_maturity - current_time
                st.write(f"• Approximate duration: {duration_approx:.2f} years")
            
            with bond_col3:
                # Yield curve
                maturities = np.linspace(0.1, 10, 50)
                bond_prices = [model.zero_coupon_bond_price(current_rate, mat, current_time) 
                             for mat in maturities if mat > current_time]
                valid_maturities = [mat for mat in maturities if mat > current_time]
                yields = [-np.log(price) / (mat - current_time) 
                        for price, mat in zip(bond_prices, valid_maturities)]
                
                fig_yield = go.Figure()
                fig_yield.add_trace(go.Scatter(x=valid_maturities, y=np.array(yields) * 100, 
                                             mode='lines', line=dict(width=3, color='green'),
                                             name='Yield curve'))
                fig_yield.add_hline(y=theta*100, line_dash="dot", line_color="red",
                                  annotation_text="Long-term rate")
                fig_yield.update_layout(title="Zero-Coupon Yield Curve", 
                                      xaxis_title="Maturity (years)", 
                                      yaxis_title="Yield (%)",
                                      template='plotly_white')
                st.plotly_chart(fig_yield, use_container_width=True)
            
            # Rate statistics
            st.subheader("Interest Rate Behavior Analysis")
            
            final_rates = rates[-1, :] * 100
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.markdown("**Rate Statistics:**")
                st.write(f"• Initial rate: {r0*100:.3f}%")
                st.write(f"• Average final rate: {np.mean(final_rates):.3f}%")
                st.write(f"• Rate standard deviation: {np.std(final_rates):.3f}%")
                st.write(f"• Rate range: {np.min(final_rates):.3f}% - {np.max(final_rates):.3f}%")
                
            with stats_col2:
                st.markdown("**Mean Reversion Analysis:**")
                st.write(f"• Target rate: {theta*100:.3f}%")
                st.write(f"• Reversion speed: {a:.2f}")
                st.write(f"• Half-life: {np.log(2)/a:.2f} years")
                
                pct_near_target = np.mean(np.abs(final_rates - theta*100) < 1.0) * 100
                st.write(f"• % of paths within 1% of target: {pct_near_target:.1f}%")
                
        except Exception as e:
            st.error(f"Hull-White simulation failed: {e}")

def show_option_pricing():
    st.header("Black-Scholes Option Pricing and Risk Sensitivities")
    st.markdown("""
    Options are complex instruments, but the Black-Scholes model gives us both prices 
    and risk sensitivities (the "Greeks"). Understanding these sensitivities is crucial 
    for managing options portfolios and understanding how they respond to market changes.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Option Specifications")
        S = st.number_input("Current stock price", value=100.0, min_value=1.0, step=1.0)
        K = st.number_input("Strike price", value=100.0, min_value=1.0, step=1.0)
        T_option = st.slider("Time to expiration (years)", 0.01, 2.0, 0.25, 0.01)
        r = st.slider("Risk-free rate", 0.0, 0.15, 0.05, 0.001, format="%.3f")
        sigma = st.slider("Implied volatility", 0.05, 1.5, 0.25, 0.01)
        
        # Calculate prices and Greeks
        call_price = black_scholes_call(S, K, T_option, r, sigma)
        put_price = black_scholes_put(S, K, T_option, r, sigma)
        greeks = calculate_greeks(S, K, T_option, r, sigma)
        
        st.subheader("Option Values")
        st.metric("Call option price", f"${call_price:.3f}")
        st.metric("Put option price", f"${put_price:.3f}")
        
        # Put-call parity check
        parity_check = call_price - put_price - (S - K * np.exp(-r * T_option))
        parity_status = "✓ Valid" if abs(parity_check) < 0.0001 else "✗ Invalid"
        st.metric("Put-call parity", parity_status, f"Error: {parity_check:.6f}")
    
    with col2:
        # Greeks explanation and values
        st.subheader("The Greeks: Risk Sensitivities")
        
        greeks_col1, greeks_col2 = st.columns(2)
        
        with greeks_col1:
            st.markdown("**First-order Greeks:**")
            st.metric("Delta (Call)", f"{greeks['call_delta']:.4f}")
            st.metric("Delta (Put)", f"{greeks['put_delta']:.4f}")
            st.metric("Vega", f"{greeks['vega']:.3f}")
            st.metric("Rho (Call)", f"{greeks['call_rho']:.3f}")
        
        with greeks_col2:
            st.markdown("**Higher-order Greeks:**")
            st.metric("Gamma", f"{greeks['gamma']:.4f}")
            st.metric("Theta (Call)", f"{greeks['call_theta']:.3f}")
            st.metric("Theta (Put)", f"{greeks['put_theta']:.3f}")
            st.metric("Rho (Put)", f"{greeks['put_rho']:.3f}")
    
    # Sensitivity analysis
    st.subheader("Interactive Sensitivity Analysis")
    
    # Stock price sensitivity
    spot_range = np.linspace(S * 0.6, S * 1.4, 100)
    call_prices_spot = [black_scholes_call(s, K, T_option, r, sigma) for s in spot_range]
    put_prices_spot = [black_scholes_put(s, K, T_option, r, sigma) for s in spot_range]
    
    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(x=spot_range, y=call_prices_spot, name='Call Option', 
                                line=dict(color='green', width=3)))
    fig_spot.add_trace(go.Scatter(x=spot_range, y=put_prices_spot, name='Put Option', 
                                line=dict(color='red', width=3)))
    
    # Add payoff diagrams
    call_payoff = np.maximum(spot_range - K, 0)
    put_payoff = np.maximum(K - spot_range, 0)
    fig_spot.add_trace(go.Scatter(x=spot_range, y=call_payoff, name='Call Payoff at Expiry',
                                line=dict(color='lightgreen', dash='dot')))
    fig_spot.add_trace(go.Scatter(x=spot_range, y=put_payoff, name='Put Payoff at Expiry',
                                line=dict(color='lightcoral', dash='dot')))
    
    fig_spot.add_vline(x=S, line_dash="dash", annotation_text="Current Price", line_color="blue")
    fig_spot.add_vline(x=K, line_dash="dot", annotation_text="Strike", line_color="orange")
    fig_spot.update_layout(title="Option Values vs Stock Price", 
                         xaxis_title="Stock Price ($)", yaxis_title="Option Value ($)",
                         template='plotly_white')
    st.plotly_chart(fig_spot, use_container_width=True)
    
    # Create tabs for different sensitivity analyses
    tab1, tab2, tab3 = st.tabs(["Volatility Sensitivity", "Time Decay", "Greeks Surface"])
    
    with tab1:
        # Volatility sensitivity
        vol_range = np.linspace(0.05, 1.0, 100)
        call_prices_vol = [black_scholes_call(S, K, T_option, r, v) for v in vol_range]
        put_prices_vol = [black_scholes_put(S, K, T_option, r, v) for v in vol_range]
        
        fig_vol.add_trace(go.Scatter(x=vol_range*100, y=put_prices_vol, name='Put Option', 
                                   line=dict(color='red', width=3)))
        fig_vol.add_vline(x=sigma*100, line_dash="dash", annotation_text="Current Volatility", 
                        line_color="blue")
        fig_vol.update_layout(title="Option Values vs Volatility", 
                            xaxis_title="Volatility (%)", yaxis_title="Option Value ($)",
                            template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.markdown("""
        **Volatility Impact:** Higher volatility increases option values because it increases 
        the probability of large price movements, benefiting option holders who have limited downside 
        but unlimited upside potential.
        """)
    
    with tab2:
        # Time decay analysis
        if T_option > 0.01:
            time_range = np.linspace(0.01, T_option, 100)
            call_prices_time = [black_scholes_call(S, K, t, r, sigma) for t in time_range]
            put_prices_time = [black_scholes_put(S, K, t, r, sigma) for t in time_range]
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=time_range*365, y=call_prices_time, name='Call Option', 
                                        line=dict(color='green', width=3)))
            fig_time.add_trace(go.Scatter(x=time_range*365, y=put_prices_time, name='Put Option', 
                                        line=dict(color='red', width=3)))
            fig_time.add_vline(x=T_option*365, line_dash="dash", annotation_text="Current TTM", 
                             line_color="blue")
            fig_time.update_layout(title="Option Values vs Time to Maturity", 
                                 xaxis_title="Days to Expiration", yaxis_title="Option Value ($)",
                                 template='plotly_white')
            st.plotly_chart(fig_time, use_container_width=True)
            
            st.markdown("""
            **Time Decay (Theta):** Options lose value as time passes, all else being equal. 
            This decay accelerates as expiration approaches, which is why theta is often called 
            the "enemy" of option buyers.
            """)
    
    with tab3:
        # Greeks surface - Delta surface
        spot_grid = np.linspace(S * 0.7, S * 1.3, 30)
        time_grid = np.linspace(0.02, T_option, 30)
        
        S_mesh, T_mesh = np.meshgrid(spot_grid, time_grid)
        delta_surface = np.zeros_like(S_mesh)
        
        for i, t in enumerate(time_grid):
            for j, s in enumerate(spot_grid):
                greeks_point = calculate_greeks(s, K, t, r, sigma)
                delta_surface[i, j] = greeks_point['call_delta']
        
        fig_delta_surface = go.Figure(data=[go.Surface(z=delta_surface, x=S_mesh, y=T_mesh*365,
                                                     colorscale='RdYlBu')])
        fig_delta_surface.update_layout(
            title='Call Delta Surface',
            scene=dict(
                xaxis_title='Stock Price ($)',
                yaxis_title='Days to Expiration',
                zaxis_title='Delta'
            ),
            height=500
        )
        st.plotly_chart(fig_delta_surface, use_container_width=True)
        
        st.markdown("""
        **Delta Surface:** This 3D visualization shows how delta (price sensitivity) changes 
        with both stock price and time to expiration. Notice how delta approaches 1 for 
        deep in-the-money calls and 0 for deep out-of-the-money calls.
        """)
    
    # Portfolio analysis section
    st.subheader("Portfolio Risk Analysis")
    
    portfolio_col1, portfolio_col2 = st.columns(2)
    
    with portfolio_col1:
        st.markdown("**Hypothetical Portfolio:**")
        n_calls = st.number_input("Number of call options", value=10, min_value=0, step=1)
        n_puts = st.number_input("Number of put options", value=-5, step=1)
        n_shares = st.number_input("Number of shares", value=100, step=1)
    
    with portfolio_col2:
        # Calculate portfolio Greeks
        portfolio_value = (n_calls * call_price + n_puts * put_price + n_shares * S)
        portfolio_delta = (n_calls * greeks['call_delta'] + 
                         n_puts * greeks['put_delta'] + n_shares)
        portfolio_gamma = (n_calls + n_puts) * greeks['gamma']
        portfolio_theta = (n_calls * greeks['call_theta'] + n_puts * greeks['put_theta'])
        portfolio_vega = (n_calls + n_puts) * greeks['vega']
        
        st.markdown("**Portfolio Risk Metrics:**")
        st.write(f"• Total portfolio value: ${portfolio_value:,.2f}")
        st.write(f"• Portfolio delta: {portfolio_delta:.2f}")
        st.write(f"• Portfolio gamma: {portfolio_gamma:.4f}")
        st.write(f"• Portfolio theta: ${portfolio_theta:.2f}/day")
        st.write(f"• Portfolio vega: ${portfolio_vega:.2f}")
        
        # Risk interpretation
        st.markdown("**Risk Interpretation:**")
        if abs(portfolio_delta) < 0.1:
            st.success("Portfolio is approximately delta-neutral")
        elif portfolio_delta > 0:
            st.info(f"Portfolio gains ${portfolio_delta:.2f} per $1 stock increase")
        else:
            st.warning(f"Portfolio loses ${abs(portfolio_delta):.2f} per $1 stock increase")

# Footer with educational content
def show_footer():
    st.markdown("---")
    st.markdown("""
    ### About This Dashboard
    
    This interactive dashboard demonstrates the power of quantitative finance models in understanding 
    market behavior and pricing financial instruments. Each model represents decades of academic 
    research and practical application in financial markets.
    
    **Educational Purpose:** These models help us understand how financial markets work, how to 
    price complex instruments, and how to manage risk in portfolios.
    
    **Real-World Application:** Professional traders, risk managers, and quants use these exact 
    models every day to make multi-million dollar decisions in global financial markets.
    
    **Remember:** All models are approximations of reality. They're powerful tools for understanding 
    and prediction, but they should always be used with awareness of their limitations and assumptions.
    """)

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your model imports and ensure all dependencies are installed.") = go.Figure()
        fig_vol.add_trace(go.Scatter(x=vol_range*100, y=call_prices_vol, name='Call Option', 
                                   line=dict(color='green', width=3)))
