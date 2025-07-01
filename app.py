import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import your models (assuming they're in the same directory structure)
# You may need to adjust these imports based on your actual file structure
import sys
import os

# Add the models directory to the path
sys.path.append('models')

# Model implementations (copied from your files)
class BlackScholesSDE:
    def __init__(self, spot, mu, sigma, dt=1/252):
        self.spot = spot
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        N = int(T / self.dt)
        paths = np.zeros((N + 1, n_paths))
        paths[0] = self.spot

        for t in range(1, N + 1):
            z = np.random.standard_normal(n_paths)
            paths[t] = paths[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt +
                self.sigma * np.sqrt(self.dt) * z
            )
        return paths

class HestonModel:
    def __init__(self, spot, mu, v0, kappa, theta, xi, rho, dt=1/252):
        self.spot = spot
        self.mu = mu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        N = int(T / self.dt)
        S = np.zeros((N + 1, n_paths))
        v = np.zeros((N + 1, n_paths))
        S[0] = self.spot
        v[0] = self.v0

        for t in range(1, N + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            W1 = z1
            W2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2

            v_prev = np.maximum(v[t - 1], 0)
            v[t] = v_prev + self.kappa * (self.theta - v_prev) * self.dt + \
                   self.xi * np.sqrt(v_prev * self.dt) * W2
            v[t] = np.maximum(v[t], 0)

            S[t] = S[t - 1] * np.exp(
                (self.mu - 0.5 * v_prev) * self.dt + np.sqrt(v_prev * self.dt) * W1
            )
        return S, v

class JumpDiffusionModel:
    def __init__(self, spot, mu, sigma, lamb, mu_j, sigma_j, dt=1/252):
        self.spot = spot
        self.mu = mu
        self.sigma = sigma
        self.lamb = lamb
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.dt = dt

    def simulate_paths(self, T, n_paths):
        N = int(T / self.dt)
        paths = np.zeros((N + 1, n_paths))
        paths[0] = self.spot

        for t in range(1, N + 1):
            z = np.random.standard_normal(n_paths)
            jumps = np.random.poisson(self.lamb * self.dt, n_paths)
            jump_sizes = np.random.normal(self.mu_j, self.sigma_j, n_paths)
            J = np.exp(jump_sizes) - 1

            paths[t] = paths[t - 1] * np.exp(
                (self.mu - self.lamb * self.mu_j - 0.5 * self.sigma ** 2) * self.dt +
                self.sigma * np.sqrt(self.dt) * z
            ) * (1 + J * jumps)

        return paths

class HullWhiteModel:
    def __init__(self, r0, a, sigma, theta=0.0, dt=1/252):
        self.r0 = r0
        self.a = a
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def theta_t(self, t):
        if callable(self.theta):
            return self.theta(t)
        return self.theta

    def simulate_short_rate(self, T, n_paths):
        N = int(T / self.dt)
        r = np.zeros((N + 1, n_paths))
        r[0] = self.r0

        for t in range(1, N + 1):
            time = t * self.dt
            z = np.random.standard_normal(n_paths)
            r[t] = r[t - 1] + (self.theta_t(time) - self.a * r[t - 1]) * self.dt + \
                   self.sigma * np.sqrt(self.dt) * z

        return r

# Black-Scholes Option Pricing
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Streamlit App
def main():
    st.set_page_config(page_title="Quantitative Models Dashboard", layout="wide")
    
    st.title("🏛️ Quantitative Finance Models Dashboard")
    st.markdown("Interactive dashboard for exploring various quantitative finance models")
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose a model:",
        ["Black-Scholes", "Heston Model", "Jump Diffusion", "Hull-White", "Option Pricing"]
    )
    
    # Common parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameters")
    n_paths = st.sidebar.slider("Number of Paths", 10, 1000, 100)
    T = st.sidebar.slider("Time Horizon (years)", 0.1, 5.0, 1.0, 0.1)
    
    if model_type == "Black-Scholes":
        st.header("📈 Black-Scholes Model")
        st.markdown("Geometric Brownian Motion: dS = μS dt + σS dW")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            spot = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
        
        if st.button("Simulate Paths"):
            model = BlackScholesSDE(spot, mu, sigma)
            paths = model.simulate_paths(T, n_paths)
            
            # Create time axis
            time_steps = np.linspace(0, T, paths.shape[0])
            
            # Plot using Plotly
            fig = go.Figure()
            
            # Add a sample of paths for visualization (max 50 for performance)
            sample_paths = min(50, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, 
                    y=paths[:, i],
                    mode='lines',
                    line=dict(width=0.5),
                    showlegend=False,
                    opacity=0.6
                ))
            
            # Add mean path
            mean_path = np.mean(paths, axis=1)
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=mean_path,
                mode='lines',
                line=dict(color='red', width=3),
                name='Mean Path'
            ))
            
            fig.update_layout(
                title=f"Black-Scholes Simulation ({n_paths} paths)",
                xaxis_title="Time (years)",
                yaxis_title="Stock Price",
                height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
            
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
    
    elif model_type == "Heston Model":
        st.header("📊 Heston Stochastic Volatility Model")
        st.markdown("dS = μS dt + √v S dW₁, dv = κ(θ-v) dt + ξ√v dW₂")
        
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
        
        if st.button("Simulate Heston Paths"):
            model = HestonModel(spot, mu, v0, kappa, theta, xi, rho)
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
    
    elif model_type == "Jump Diffusion":
        st.header("⚡ Merton Jump Diffusion Model")
        st.markdown("dS/S = (μ - λk) dt + σ dW + J dq")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            spot = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
            sigma = st.slider("Diffusion Vol (σ)", 0.01, 1.0, 0.2, 0.01)
            lamb = st.slider("Jump Intensity (λ)", 0.0, 10.0, 1.0, 0.1)
            mu_j = st.slider("Mean Jump Size (μⱼ)", -0.5, 0.5, -0.1, 0.01)
            sigma_j = st.slider("Jump Vol (σⱼ)", 0.01, 1.0, 0.3, 0.01)
        
        if st.button("Simulate Jump Diffusion"):
            model = JumpDiffusionModel(spot, mu, sigma, lamb, mu_j, sigma_j)
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
                xaxis_title="Time (years)",
                yaxis_title="Stock Price",
                height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
    
    elif model_type == "Hull-White":
        st.header("🏦 Hull-White Interest Rate Model")
        st.markdown("dr = [θ(t) - ar] dt + σ dW")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            r0 = st.slider("Initial Rate (r₀)", 0.0, 0.2, 0.05, 0.001)
            a = st.slider("Mean Reversion (a)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.001, 0.1, 0.02, 0.001)
            theta = st.slider("Long-term Rate (θ)", 0.0, 0.2, 0.05, 0.001)
        
        if st.button("Simulate Interest Rates"):
            model = HullWhiteModel(r0, a, sigma, theta)
            rates = model.simulate_short_rate(T, n_paths)
            
            time_steps = np.linspace(0, T, rates.shape[0])
            
            fig = go.Figure()
            
            sample_paths = min(50, n_paths)
            for i in range(sample_paths):
                fig.add_trace(go.Scatter(
                    x=time_steps, y=rates[:, i] * 100,  # Convert to percentage
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
                xaxis_title="Time (years)",
                yaxis_title="Interest Rate (%)",
                height=500
            )
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
    
    elif model_type == "Option Pricing":
        st.header("💰 Black-Scholes Option Pricing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Option Parameters")
            S = st.number_input("Current Stock Price (S)", value=100.0, min_value=1.0)
            K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0)
            T_option = st.slider("Time to Expiry (years)", 0.01, 5.0, 0.25, 0.01)
            r = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, 0.001)
            sigma = st.slider("Volatility", 0.01, 2.0, 0.2, 0.01)
        
        # Calculate option prices
        call_price = black_scholes_call(S, K, T_option, r, sigma)
        put_price = black_scholes_put(S, K, T_option, r, sigma)
        
        with col2:
            st.subheader("Option Prices")
            st.metric("Call Option Price", f"${call_price:.2f}")
            st.metric("Put Option Price", f"${put_price:.2f}")
            st.metric("Call-Put Parity Check", f"${call_price - put_price - (S - K * np.exp(-r * T_option)):.6f}")
        
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

if __name__ == "__main__":
    main()
