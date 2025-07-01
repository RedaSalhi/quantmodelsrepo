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
from models.black_scholes.sde import BlackScholesSDE
from models.black_scholes.pde import BlackScholesPDE
from models.stochastic_volatility.heston import HestonModel
from models.stochastic_volatility.sabr import SABRModel
from models.jump_models.jump_diffusion import JumpDiffusionModel
from models.local_volatility.local_vol import LocalVolatilityModel
from models.hull_white.hull_white import HullWhiteModel
from models.hull_white.extensions import HullWhiteExtensions

# Black-Scholes Option Pricing Functions
def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option pricing formula"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option pricing formula"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calculate option Greeks"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2))
    put_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    # Rho
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'call_theta': call_theta, 'put_theta': put_theta,
        'vega': vega, 'call_rho': call_rho, 'put_rho': put_rho
    }

# Streamlit App
def main():
    st.set_page_config(page_title="Quantitative Models Dashboard", layout="wide")
    
    st.title("🏛️ Quantitative Finance Models Dashboard")
    st.markdown("Interactive dashboard using your existing quantitative finance models")
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose a model:",
        ["Black-Scholes SDE", "Black-Scholes PDE", "Heston Model", "SABR Model", 
         "Jump Diffusion", "Local Volatility", "Hull-White", "Option Pricing & Greeks"]
    )
    
    # Common parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameters")
    n_paths = st.sidebar.slider("Number of Paths", 10, 1000, 100)
    T = st.sidebar.slider("Time Horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    if model_type == "Black-Scholes SDE":
        st.header("📈 Black-Scholes SDE Model")
        st.markdown("**Model:** dS = μS dt + σS dW")
        st.code("from models.black_scholes.sde import BlackScholesSDE")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            spot = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
            dt = st.selectbox("Time Step", [1/252, 1/52, 1/12], index=0, format_func=lambda x: f"1/{int(1/x)} (daily)" if x == 1/252 else f"1/{int(1/x)}")
        
        if st.button("Simulate SDE Paths"):
            # Use your existing BlackScholesSDE class
            model = BlackScholesSDE(spot, mu, sigma, dt)
            paths = model.simulate_paths(T, n_paths)
            
            time_steps = np.linspace(0, T, paths.shape[0])
            
            fig = go.Figure()
            
            # Add sample paths
            sample_paths = min(50, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ))
            
            # Add mean path
            mean_path = np.mean(paths, axis=1)
            fig.add_trace(go.Scatter(
                x=time_steps, y=mean_path,
                mode='lines', line=dict(color='red', width=3),
                name='Mean Path'
            ))
            
            fig.update_layout(
                title=f"Black-Scholes SDE Simulation ({n_paths} paths)",
                xaxis_title="Time (years)", yaxis_title="Stock Price", height=500
            )
            
            with col2:
                st.plotly_chart(fig_spot, use_container_width=True)
        
        # Volatility sensitivity
        vol_range = np.linspace(0.1, 1.0, 50)
        call_prices_vol = [black_scholes_call(S, K, T_option, r, v) for v in vol_range]
        put_prices_vol = [black_scholes_put(S, K, T_option, r, v) for v in vol_range]
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=vol_range, y=call_prices_vol, name='Call', line=dict(color='green')))
        fig_vol.add_trace(go.Scatter(x=vol_range, y=put_prices_vol, name='Put', line=dict(color='red')))
        fig_vol.add_vline(x=sigma, line_dash="dash", annotation_text="Current Vol")
        fig_vol.update_layout(title="Option Price vs Volatility", xaxis_title="Volatility", yaxis_title="Option Price")
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Time decay analysis
        time_range = np.linspace(0.01, T_option, 50)
        call_prices_time = [black_scholes_call(S, K, t, r, sigma) for t in time_range]
        put_prices_time = [black_scholes_put(S, K, t, r, sigma) for t in time_range]
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=time_range, y=call_prices_time, name='Call', line=dict(color='green')))
        fig_time.add_trace(go.Scatter(x=time_range, y=put_prices_time, name='Put', line=dict(color='red')))
        fig_time.add_vline(x=T_option, line_dash="dash", annotation_text="Current TTM")
        fig_time.update_layout(title="Option Price vs Time to Maturity", xaxis_title="Time to Maturity", yaxis_title="Option Price")
        
        st.plotly_chart(fig_time, use_container_width=True)

if __name__ == "__main__":
    main(), use_container_width=True)
            
            # Statistics
            st.subheader("📊 Path Statistics")
            final_prices = paths[-1, :]
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
            with stats_col2:
                st.metric("Std Dev", f"${np.std(final_prices):.2f}")
            with stats_col3:
                st.metric("Min Price", f"${np.min(final_prices):.2f}")
            with stats_col4:
                st.metric("Max Price", f"${np.max(final_prices):.2f}")

    elif model_type == "Black-Scholes PDE":
        st.header("🧮 Black-Scholes PDE Solver")
        st.markdown("**Model:** Finite difference solution to Black-Scholes PDE")
        st.code("from models.black_scholes.pde import BlackScholesPDE")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PDE Parameters")
            S_max = st.number_input("Max Asset Price", value=200.0, min_value=50.0)
            K = st.number_input("Strike Price", value=100.0, min_value=1.0)
            T_pde = st.slider("Time to Maturity", 0.1, 2.0, 0.25, 0.05)
            r = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, 0.001)
            sigma = st.slider("Volatility", 0.01, 1.0, 0.2, 0.01)
            M = st.slider("Asset Price Steps", 50, 200, 100)
            N = st.slider("Time Steps", 100, 2000, 1000)
        
        if st.button("Solve PDE"):
            # Use your existing BlackScholesPDE class
            pde_solver = BlackScholesPDE(S_max, K, T_pde, r, sigma, M, N)
            S, V = pde_solver.solve_call()
            
            # Plot option price vs spot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=S, y=V, mode='lines', name='Call Option Price'))
            fig.add_vline(x=K, line_dash="dash", annotation_text="Strike Price")
            fig.update_layout(
                title="Call Option Price vs Spot Price (PDE Solution)",
                xaxis_title="Spot Price", yaxis_title="Option Price", height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show analytical vs numerical comparison
            analytical_prices = [black_scholes_call(s, K, T_pde, r, sigma) for s in S]
            
            st.subheader("📊 PDE vs Analytical Comparison")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=S, y=V, name='PDE Solution', line=dict(color='blue')))
            fig_comp.add_trace(go.Scatter(x=S, y=analytical_prices, name='Analytical', line=dict(color='red', dash='dash')))
            fig_comp.update_layout(title="PDE vs Analytical Solution", xaxis_title="Spot Price", yaxis_title="Option Price")
            st.plotly_chart(fig_comp, use_container_width=True)

    elif model_type == "Heston Model":
        st.header("📊 Heston Stochastic Volatility Model")
        st.markdown("**Model:** dS = μS dt + √v S dW₁, dv = κ(θ-v) dt + ξ√v dW₂")
        st.code("from models.stochastic_volatility.heston import HestonModel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            spot = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
            v0 = st.slider("Initial Variance (v₀)", 0.01, 1.0, 0.04, 0.01)
            kappa = st.slider("Mean Reversion Speed (κ)", 0.1, 10.0, 2.0, 0.1)
            theta = st.slider("Long-term Variance (θ)", 0.01, 1.0, 0.04, 0.01)
            xi = st.slider("Vol of Vol (ξ)", 0.01, 2.0, 0.3, 0.01)
            rho = st.slider("Correlation (ρ)", -1.0, 1.0, -0.7, 0.1)
            dt = st.selectbox("Time Step", [1/252, 1/52], index=0, format_func=lambda x: f"1/{int(1/x)}")
        
        if st.button("Simulate Heston Paths"):
            # Use your existing HestonModel class
            model = HestonModel(spot, mu, v0, kappa, theta, xi, rho, dt)
            S_paths, v_paths = model.simulate_paths(T, n_paths)
            
            time_steps = np.linspace(0, T, S_paths.shape[0])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price Paths', 'Variance Paths'),
                vertical_spacing=0.1
            )
            
            # Stock price paths
            sample_paths = min(20, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=S_paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ), row=1, col=1)
            
            # Variance paths
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=v_paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ), row=2, col=1)
            
            fig.update_layout(height=600, title="Heston Model Simulation")
            fig.update_xaxes(title_text="Time (years)", row=2, col=1)
            fig.update_yaxes(title_text="Stock Price", row=1, col=1)
            fig.update_yaxes(title_text="Variance", row=2, col=1)
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)

    elif model_type == "SABR Model":
        st.header("📈 SABR Stochastic Volatility Model")
        st.markdown("**Model:** dF = σ F^β dW₁, dσ = νσ dW₂")
        st.code("from models.stochastic_volatility.sabr import SABRModel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            F0 = st.number_input("Initial Forward (F₀)", value=100.0, min_value=1.0)
            alpha = st.slider("Initial Volatility (α)", 0.01, 1.0, 0.2, 0.01)
            beta = st.slider("Elasticity (β)", 0.0, 1.0, 0.5, 0.1)
            rho = st.slider("Correlation (ρ)", -1.0, 1.0, -0.3, 0.1)
            nu = st.slider("Vol of Vol (ν)", 0.01, 2.0, 0.3, 0.01)
            dt = st.selectbox("Time Step", [1/252, 1/52], index=0, format_func=lambda x: f"1/{int(1/x)}")
        
        if st.button("Simulate SABR Paths"):
            # Use your existing SABRModel class
            model = SABRModel(F0, alpha, beta, rho, nu, dt)
            F_paths, sigma_paths = model.simulate_paths(T, n_paths)
            
            time_steps = np.linspace(0, T, F_paths.shape[0])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Forward Price Paths', 'Volatility Paths'),
                vertical_spacing=0.1
            )
            
            # Forward paths
            sample_paths = min(20, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=F_paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ), row=1, col=1)
            
            # Volatility paths
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=sigma_paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ), row=2, col=1)
            
            fig.update_layout(height=600, title="SABR Model Simulation")
            fig.update_xaxes(title_text="Time (years)", row=2, col=1)
            fig.update_yaxes(title_text="Forward Price", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)

    elif model_type == "Jump Diffusion":
        st.header("⚡ Merton Jump Diffusion Model")
        st.markdown("**Model:** dS/S = (μ - λk) dt + σ dW + J dq")
        st.code("from models.jump_models.jump_diffusion import JumpDiffusionModel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            spot = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
            sigma = st.slider("Diffusion Vol (σ)", 0.01, 1.0, 0.2, 0.01)
            lamb = st.slider("Jump Intensity (λ)", 0.0, 10.0, 1.0, 0.1)
            mu_j = st.slider("Mean Jump Size (μⱼ)", -0.5, 0.5, -0.1, 0.01)
            sigma_j = st.slider("Jump Vol (σⱼ)", 0.01, 1.0, 0.3, 0.01)
            dt = st.selectbox("Time Step", [1/252, 1/52], index=0, format_func=lambda x: f"1/{int(1/x)}")
        
        if st.button("Simulate Jump Diffusion"):
            # Use your existing JumpDiffusionModel class
            model = JumpDiffusionModel(spot, mu, sigma, lamb, mu_j, sigma_j, dt)
            paths = model.simulate_paths(T, n_paths)
            
            time_steps = np.linspace(0, T, paths.shape[0])
            
            fig = go.Figure()
            
            sample_paths = min(50, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ))
            
            # Add mean path
            mean_path = np.mean(paths, axis=1)
            fig.add_trace(go.Scatter(
                x=time_steps, y=mean_path,
                mode='lines', line=dict(color='red', width=3),
                name='Mean Path'
            ))
            
            fig.update_layout(
                title=f"Jump Diffusion Simulation ({n_paths} paths)",
                xaxis_title="Time (years)", yaxis_title="Stock Price", height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)

    elif model_type == "Local Volatility":
        st.header("🌊 Local Volatility Model")
        st.markdown("**Model:** dS = σ(S,t) S dW")
        st.code("from models.local_volatility.local_vol import LocalVolatilityModel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            spot = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=1.0)
            vol_type = st.selectbox("Volatility Surface Type", ["Constant", "Time-dependent", "Level-dependent"])
            
            if vol_type == "Constant":
                base_vol = st.slider("Constant Volatility", 0.1, 1.0, 0.2, 0.01)
                local_vol_surface = lambda S, t: base_vol
            elif vol_type == "Time-dependent":
                base_vol = st.slider("Base Volatility", 0.1, 1.0, 0.2, 0.01)
                time_decay = st.slider("Time Decay Factor", 0.0, 2.0, 0.5, 0.1)
                local_vol_surface = lambda S, t: base_vol * np.exp(-time_decay * t)
            else:  # Level-dependent
                base_vol = st.slider("Base Volatility", 0.1, 1.0, 0.2, 0.01)
                level_factor = st.slider("Level Factor", 0.0, 2.0, 0.5, 0.1)
                local_vol_surface = lambda S, t: base_vol * (1 + level_factor * (S / spot - 1))
            
            dt = st.selectbox("Time Step", [1/252, 1/52], index=0, format_func=lambda x: f"1/{int(1/x)}")
        
        if st.button("Simulate Local Vol Paths"):
            # Use your existing LocalVolatilityModel class
            model = LocalVolatilityModel(spot, local_vol_surface, dt)
            paths = model.simulate_paths(T, n_paths)
            
            time_steps = np.linspace(0, T, paths.shape[0])
            
            fig = go.Figure()
            
            sample_paths = min(50, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=paths[:, i],
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ))
            
            # Add mean path
            mean_path = np.mean(paths, axis=1)
            fig.add_trace(go.Scatter(
                x=time_steps, y=mean_path,
                mode='lines', line=dict(color='red', width=3),
                name='Mean Path'
            ))
            
            fig.update_layout(
                title=f"Local Volatility Simulation ({n_paths} paths)",
                xaxis_title="Time (years)", yaxis_title="Stock Price", height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)

    elif model_type == "Hull-White":
        st.header("🏦 Hull-White Interest Rate Model")
        st.markdown("**Model:** dr = [θ(t) - ar] dt + σ dW")
        st.code("from models.hull_white.hull_white import HullWhiteModel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            r0 = st.slider("Initial Rate (r₀)", 0.0, 0.2, 0.05, 0.001)
            a = st.slider("Mean Reversion (a)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.001, 0.1, 0.02, 0.001)
            theta = st.slider("Long-term Rate (θ)", 0.0, 0.2, 0.05, 0.001)
            dt = st.selectbox("Time Step", [1/252, 1/52], index=0, format_func=lambda x: f"1/{int(1/x)}")
        
        if st.button("Simulate Interest Rates"):
            # Use your existing HullWhiteModel class
            model = HullWhiteModel(r0, a, sigma, theta, dt)
            rates = model.simulate_short_rate(T, n_paths)
            
            time_steps = np.linspace(0, T, rates.shape[0])
            
            fig = go.Figure()
            
            sample_paths = min(50, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=rates[:, i] * 100,
                    mode='lines', line=dict(width=0.5),
                    showlegend=False, opacity=0.6
                ))
            
            # Add mean path
            mean_rates = np.mean(rates, axis=1)
            fig.add_trace(go.Scatter(
                x=time_steps, y=mean_rates * 100,
                mode='lines', line=dict(color='red', width=3),
                name='Mean Rate'
            ))
            
            fig.update_layout(
                title=f"Hull-White Interest Rate Simulation ({n_paths} paths)",
                xaxis_title="Time (years)", yaxis_title="Interest Rate (%)", height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
            
            # Zero Coupon Bond Pricing
            st.subheader("💰 Zero Coupon Bond Pricing")
            bond_col1, bond_col2 = st.columns(2)
            
            with bond_col1:
                current_rate = st.slider("Current Short Rate", 0.0, 0.2, r0, 0.001)
                bond_maturity = st.slider("Bond Maturity", 0.1, 10.0, 1.0, 0.1)
                current_time = st.slider("Current Time", 0.0, T, 0.0, 0.1)
            
            bond_price = model.zero_coupon_bond_price(current_rate, bond_maturity, current_time)
            
            with bond_col2:
                st.metric("Zero Coupon Bond Price", f"{bond_price:.4f}")
                st.metric("Implied Yield", f"{(-np.log(bond_price) / bond_maturity * 100):.2f}%")

    elif model_type == "Option Pricing & Greeks":
        st.header("💰 Black-Scholes Option Pricing & Greeks")
        st.markdown("**Complete option analysis with Greeks calculation**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Option Parameters")
            S = st.number_input("Current Stock Price (S)", value=100.0, min_value=1.0)
            K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0)
            T_option = st.slider("Time to Expiry (years)", 0.01, 5.0, 0.25, 0.01)
            r = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, 0.001)
            sigma = st.slider("Volatility", 0.01, 2.0, 0.2, 0.01)
        
        # Calculate option prices and Greeks
        call_price = black_scholes_call(S, K, T_option, r, sigma)
        put_price = black_scholes_put(S, K, T_option, r, sigma)
        greeks = calculate_greeks(S, K, T_option, r, sigma)
        
        with col2:
            st.subheader("Option Prices")
            st.metric("Call Option Price", f"${call_price:.2f}")
            st.metric("Put Option Price", f"${put_price:.2f}")
            st.metric("Call-Put Parity Check", f"${call_price - put_price - (S - K * np.exp(-r * T_option)):.6f}")
        
        # Greeks Display
        st.subheader("📊 The Greeks")
        greek_col1, greek_col2, greek_col3, greek_col4 = st.columns(4)
        
        with greek_col1:
            st.metric("Call Delta", f"{greeks['call_delta']:.3f}")
            st.metric("Put Delta", f"{greeks['put_delta']:.3f}")
        
        with greek_col2:
            st.metric("Gamma", f"{greeks['gamma']:.3f}")
            st.metric("Vega", f"{greeks['vega']:.2f}")
        
        with greek_col3:
            st.metric("Call Theta", f"{greeks['call_theta']:.2f}")
            st.metric("Put Theta", f"{greeks['put_theta']:.2f}")
        
        with greek_col4:
            st.metric("Call Rho", f"{greeks['call_rho']:.2f}")
            st.metric("Put Rho", f"{greeks['put_rho']:.2f}")
        
        # Sensitivity Analysis
        st.subheader("📈 Sensitivity Analysis")
        
        # Price sensitivity to spot
        spot_range = np.linspace(S * 0.7, S * 1.3, 50)
        call_prices = [black_scholes_call(s, K, T_option, r, sigma) for s in spot_range]
        put_prices = [black_scholes_put(s, K, T_option, r, sigma) for s in spot_range]
        
        fig_spot = go.Figure()
        fig_spot.add_trace(go.Scatter(x=spot_range, y=call_prices, name='Call', line=dict(color='green')))
        fig_spot.add_trace(go.Scatter(x=spot_range, y=put_prices, name='Put', line=dict(color='red')))
        fig_spot.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
        fig_spot.update_layout(title="Option Price vs Spot Price", xaxis_title="Spot Price", yaxis_title="Option Price")
        
        st.plotly_chart(fig
