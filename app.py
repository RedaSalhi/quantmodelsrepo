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
    st.error(f"Erreur d'importation des modèles: {e}")
    st.info("Assurez-vous que tous les fichiers __init__.py sont créés dans les dossiers de modèles")

# Black-Scholes Option Pricing Functions
def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option pricing formula"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option pricing formula"""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calculate option Greeks"""
    if T <= 0:
        return {
            'call_delta': 1 if S > K else 0, 'put_delta': -1 if S < K else 0,
            'gamma': 0, 'call_theta': 0, 'put_theta': 0,
            'vega': 0, 'call_rho': 0, 'put_rho': 0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'call_theta': call_theta, 'put_theta': put_theta,
        'vega': vega, 'call_rho': call_rho, 'put_rho': put_rho
    }

# Utility function for plotting paths
def plot_paths(time_steps, paths, title, ylabel="Price", mean_path=True, max_paths=50):
    """Create a plotly figure for path visualization"""
    fig = go.Figure()
    
    # Add sample paths
    n_paths = min(max_paths, paths.shape[1])
    for i in range(n_paths):
        fig.add_trace(go.Scatter(
            x=time_steps, y=paths[:, i],
            mode='lines', line=dict(width=0.5),
            showlegend=False, opacity=0.6,
            hovertemplate=f'Path {i+1}<br>Time: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>'
        ))
    
    # Add mean path if requested
    if mean_path and paths.shape[1] > 1:
        mean_values = np.mean(paths, axis=1)
        fig.add_trace(go.Scatter(
            x=time_steps, y=mean_values,
            mode='lines', line=dict(color='red', width=3),
            name='Moyenne',
            hovertemplate=f'Moyenne<br>Time: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Temps (années)",
        yaxis_title=ylabel,
        height=500,
        hovermode='closest'
    )
    
    return fig

# Streamlit App
def main():
    st.set_page_config(
        page_title="Modèles Quantitatifs Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ Dashboard des Modèles de Finance Quantitative")
    st.markdown("Dashboard interactif utilisant vos modèles de finance quantitative existants")
    
    # Sidebar for model selection
    st.sidebar.title("🔧 Sélection du Modèle")
    model_type = st.sidebar.selectbox(
        "Choisissez un modèle:",
        [
            "🏠 Accueil",
            "📈 Black-Scholes SDE", 
            "🧮 Black-Scholes PDE", 
            "📊 Modèle de Heston", 
            "📋 Modèle SABR",
            "⚡ Jump Diffusion", 
            "🌊 Volatilité Locale", 
            "🏦 Hull-White", 
            "💰 Pricing d'Options & Greeks"
        ]
    )
    
    if model_type == "🏠 Accueil":
        st.header("Bienvenue dans le Dashboard des Modèles Quantitatifs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Modèles Disponibles")
            st.markdown("""
            - **Black-Scholes SDE** : Mouvement brownien géométrique
            - **Black-Scholes PDE** : Résolution par différences finies
            - **Heston** : Modèle de volatilité stochastique
            - **SABR** : Modèle SABR pour les taux d'intérêt
            - **Jump Diffusion** : Modèle de Merton avec sauts
            - **Volatilité Locale** : Modèle de Dupire
            - **Hull-White** : Modèle de taux d'intérêt
            - **Options & Greeks** : Pricing et sensibilités
            """)
        
        with col2:
            st.subheader("📊 Fonctionnalités")
            st.markdown("""
            - **Simulation Monte Carlo** avec paramètres ajustables
            - **Visualisations interactives** avec Plotly
            - **Analyse de sensibilité** pour les options
            - **Comparaisons** entre méthodes analytiques et numériques
            - **Calcul des Greeks** en temps réel
            - **Export des résultats** (à venir)
            """)
        
        st.subheader("🚀 Comment utiliser")
        st.markdown("""
        1. **Sélectionnez un modèle** dans la barre latérale
        2. **Ajustez les paramètres** selon vos besoins
        3. **Lancez la simulation** avec le bouton correspondant
        4. **Explorez les résultats** avec les graphiques interactifs
        """)
        
        # Model structure display
        st.subheader("📁 Structure des Modèles")
        st.code("""
        models/
        ├── black_scholes/
        │   ├── sde.py          # BlackScholesSDE
        │   └── pde.py          # BlackScholesPDE
        ├── stochastic_volatility/
        │   ├── heston.py       # HestonModel
        │   └── sabr.py         # SABRModel
        ├── jump_models/
        │   └── jump_diffusion.py  # JumpDiffusionModel
        ├── local_volatility/
        │   └── local_vol.py    # LocalVolatilityModel
        └── hull_white/
            ├── hull_white.py   # HullWhiteModel
            └── extensions.py   # HullWhiteExtensions
        """)
        
        return
    
    # Common parameters for all models
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Paramètres de Simulation")
    n_paths = st.sidebar.slider("Nombre de chemins", 10, 1000, 100, 10)
    T = st.sidebar.slider("Horizon temporel (années)", 0.1, 5.0, 1.0, 0.1)
    
    if model_type == "📈 Black-Scholes SDE":
        st.header("📈 Modèle Black-Scholes SDE")
        st.markdown("**Équation:** dS = μS dt + σS dW")
        st.code("from models.black_scholes.sde import BlackScholesSDE")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres")
            spot = st.number_input("Prix initial (S₀)", value=100.0, min_value=1.0, step=1.0)
            mu = st.slider("Dérive (μ)", -0.5, 0.5, 0.05, 0.01)
            sigma = st.slider("Volatilité (σ)", 0.01, 1.0, 0.2, 0.01)
            dt_option = st.selectbox("Pas de temps", [1/252, 1/52, 1/12], index=0, 
                                   format_func=lambda x: f"1/{int(1/x)} jours" if x == 1/252 else f"1/{int(1/x)}")
            
            simulate_button = st.button("🚀 Simuler les chemins SDE", type="primary")
        
        if simulate_button:
            try:
                with st.spinner("Simulation en cours..."):
                    model = BlackScholesSDE(spot, mu, sigma, dt_option)
                    paths = model.simulate_paths(T, n_paths)
                    
                    time_steps = np.linspace(0, T, paths.shape[0])
                    
                    with col2:
                        fig = plot_paths(time_steps, paths, 
                                       f"Simulation Black-Scholes SDE ({n_paths} chemins)", 
                                       "Prix de l'action")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                st.subheader("📊 Statistiques des chemins")
                final_prices = paths[-1, :]
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Prix final moyen", f"{np.mean(final_prices):.2f} €")
                with stats_col2:
                    st.metric("Écart-type", f"{np.std(final_prices):.2f} €")
                with stats_col3:
                    st.metric("Prix minimum", f"{np.min(final_prices):.2f} €")
                with stats_col4:
                    st.metric("Prix maximum", f"{np.max(final_prices):.2f} €")
                
                # Distribution finale
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=final_prices, nbinsx=50, name="Distribution finale"))
                fig_hist.update_layout(title="Distribution des prix finaux", 
                                     xaxis_title="Prix final", yaxis_title="Fréquence")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de la simulation: {e}")

    elif model_type == "🧮 Black-Scholes PDE":
        st.header("🧮 Solveur PDE Black-Scholes")
        st.markdown("**Méthode:** Résolution par différences finies de l'EDP de Black-Scholes")
        st.code("from models.black_scholes.pde import BlackScholesPDE")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres PDE")
            S_max = st.number_input("Prix maximum de l'actif", value=200.0, min_value=50.0, step=10.0)
            K = st.number_input("Prix d'exercice (Strike)", value=100.0, min_value=1.0, step=1.0)
            T_pde = st.slider("Temps jusqu'à maturité", 0.1, 2.0, 0.25, 0.05)
            r = st.slider("Taux sans risque", 0.0, 0.2, 0.05, 0.001)
            sigma = st.slider("Volatilité", 0.01, 1.0, 0.2, 0.01)
            M = st.slider("Étapes de prix", 50, 200, 100, 10)
            N = st.slider("Étapes de temps", 100, 2000, 1000, 100)
            
            solve_button = st.button("🧮 Résoudre l'EDP", type="primary")
        
        if solve_button:
            try:
                with st.spinner("Résolution de l'EDP en cours..."):
                    pde_solver = BlackScholesPDE(S_max, K, T_pde, r, sigma, M, N)
                    S, V = pde_solver.solve_call()
                    
                    with col2:
                        # Plot option price vs spot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=S, y=V, mode='lines', name='Prix de l\'option call (PDE)',
                                               line=dict(color='blue', width=2)))
                        fig.add_vline(x=K, line_dash="dash", annotation_text="Strike", 
                                    line_color="red")
                        fig.update_layout(
                            title="Prix de l'option call vs Prix spot (Solution PDE)",
                            xaxis_title="Prix spot", yaxis_title="Prix de l'option", height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Comparaison PDE vs Analytique
                st.subheader("📊 Comparaison PDE vs Solution Analytique")
                analytical_prices = [black_scholes_call(s, K, T_pde, r, sigma) for s in S]
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=S, y=V, name='Solution PDE', 
                                            line=dict(color='blue', width=2)))
                fig_comp.add_trace(go.Scatter(x=S, y=analytical_prices, name='Solution Analytique', 
                                            line=dict(color='red', dash='dash', width=2)))
                fig_comp.update_layout(
                    title="Comparaison PDE vs Analytique", 
                    xaxis_title="Prix spot", yaxis_title="Prix de l'option"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Erreur
                error = np.abs(np.array(V) - np.array(analytical_prices))
                max_error = np.max(error)
                mean_error = np.mean(error)
                
                error_col1, error_col2 = st.columns(2)
                with error_col1:
                    st.metric("Erreur maximale", f"{max_error:.6f}")
                with error_col2:
                    st.metric("Erreur moyenne", f"{mean_error:.6f}")
                    
            except Exception as e:
                st.error(f"Erreur lors de la résolution PDE: {e}")

    elif model_type == "📊 Modèle de Heston":
        st.header("📊 Modèle de Heston - Volatilité Stochastique")
        st.markdown("**Équations:** dS = μS dt + √v S dW₁, dv = κ(θ-v) dt + ξ√v dW₂")
        st.code("from models.stochastic_volatility.heston import HestonModel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres")
            spot = st.number_input("Prix initial (S₀)", value=100.0, min_value=1.0, step=1.0)
            mu = st.slider("Dérive (μ)", -0.5, 0.5, 0.05, 0.01)
            v0 = st.slider("Variance initiale (v₀)", 0.01, 1.0, 0.04, 0.01)
            kappa = st.slider("Vitesse de retour à la moyenne (κ)", 0.1, 10.0, 2.0, 0.1)
            theta = st.slider("Variance long terme (θ)", 0.01, 1.0, 0.04, 0.01)
            xi = st.slider("Vol de vol (ξ)", 0.01, 2.0, 0.3, 0.01)
            rho = st.slider("Corrélation (ρ)", -1.0, 1.0, -0.7, 0.1)
            dt_option = st.selectbox("Pas de temps", [1/252, 1/52], index=0, 
                                   format_func=lambda x: f"1/{int(1/x)} jours" if x == 1/252 else f"1/{int(1/x)}")
            
            simulate_button = st.button("📊 Simuler Heston", type="primary")
        
        if simulate_button:
            try:
                with st.spinner("Simulation du modèle de Heston..."):
                    model = HestonModel(spot, mu, v0, kappa, theta, xi, rho, dt_option)
                    S_paths, v_paths = model.simulate_paths(T, n_paths)
                    
                    time_steps = np.linspace(0, T, S_paths.shape[0])
                    
                    with col2:
                        # Create subplots
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Prix de l\'actif', 'Variance'),
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
                        
                        # Mean paths
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=np.mean(S_paths, axis=1),
                            mode='lines', line=dict(color='red', width=3),
                            name='Prix moyen'
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=np.mean(v_paths, axis=1),
                            mode='lines', line=dict(color='red', width=3),
                            name='Variance moyenne'
                        ), row=2, col=1)
                        
                        fig.update_layout(height=600, title="Simulation du Modèle de Heston")
                        fig.update_xaxes(title_text="Temps (années)", row=2, col=1)
                        fig.update_yaxes(title_text="Prix", row=1, col=1)
                        fig.update_yaxes(title_text="Variance", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                st.subheader("📊 Statistiques")
                final_prices = S_paths[-1, :]
                final_variances = v_paths[-1, :]
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Prix final moyen", f"{np.mean(final_prices):.2f}")
                with stats_col2:
                    st.metric("Variance finale moyenne", f"{np.mean(final_variances):.4f}")
                with stats_col3:
                    st.metric("Vol finale moyenne", f"{np.sqrt(np.mean(final_variances)):.3f}")
                with stats_col4:
                    st.metric("Corrélation réalisée", f"{np.corrcoef(S_paths[-1,:], v_paths[-1,:])[0,1]:.3f}")
                    
            except Exception as e:
                st.error(f"Erreur lors de la simulation Heston: {e}")

    elif model_type == "📋 Modèle SABR":
        st.header("📋 Modèle SABR")
        st.markdown("**Équations:** dF = σ F^β dW₁, dσ = νσ dW₂")
        st.code("from models.stochastic_volatility.sabr import SABRModel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres")
            F0 = st.number_input("Forward initial (F₀)", value=100.0, min_value=1.0, step=1.0)
            alpha = st.slider("Volatilité initiale (α)", 0.01, 1.0, 0.2, 0.01)
            beta = st.slider("Élasticité (β)", 0.0, 1.0, 0.5, 0.1)
            rho = st.slider("Corrélation (ρ)", -1.0, 1.0, -0.3, 0.1)
            nu = st.slider("Vol de vol (ν)", 0.01, 2.0, 0.3, 0.01)
            dt_option = st.selectbox("Pas de temps", [1/252, 1/52], index=0, 
                                   format_func=lambda x: f"1/{int(1/x)} jours")
            
            simulate_button = st.button("📋 Simuler SABR", type="primary")
        
        if simulate_button:
            try:
                with st.spinner("Simulation du modèle SABR..."):
                    model = SABRModel(F0, alpha, beta, rho, nu, dt_option)
                    F_paths, sigma_paths = model.simulate_paths(T, n_paths)
                    
                    time_steps = np.linspace(0, T, F_paths.shape[0])
                    
                    with col2:
                        # Create subplots
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Prix Forward', 'Volatilité'),
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
                        
                        # Mean paths
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=np.mean(F_paths, axis=1),
                            mode='lines', line=dict(color='red', width=3),
                            name='Forward moyen'
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=time_steps, y=np.mean(sigma_paths, axis=1),
                            mode='lines', line=dict(color='red', width=3),
                            name='Volatilité moyenne'
                        ), row=2, col=1)
                        
                        fig.update_layout(height=600, title="Simulation du Modèle SABR")
                        fig.update_xaxes(title_text="Temps (années)", row=2, col=1)
                        fig.update_yaxes(title_text="Prix Forward", row=1, col=1)
                        fig.update_yaxes(title_text="Volatilité", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erreur lors de la simulation SABR: {e}")

    elif model_type == "⚡ Jump Diffusion":
        st.header("⚡ Modèle de Jump Diffusion de Merton")
        st.markdown("**Équation:** dS/S = (μ - λk) dt + σ dW + J dq")
        st.code("from models.jump_models.jump_diffusion import JumpDiffusionModel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres")
            spot = st.number_input("Prix initial (S₀)", value=100.0, min_value=1.0, step=1.0)
            mu = st.slider("Dérive (μ)", -0.5, 0.5, 0.05, 0.01)
            sigma = st.slider("Vol de diffusion (σ)", 0.01, 1.0, 0.2, 0.01)
            lamb = st.slider("Intensité de saut (λ)", 0.0, 10.0, 1.0, 0.1)
            mu_j = st.slider("Taille moyenne de saut (μⱼ)", -0.5, 0.5, -0.1, 0.01)
            sigma_j = st.slider("Vol de saut (σⱼ)", 0.01, 1.0, 0.3, 0.01)
            dt_option = st.selectbox("Pas de temps", [1/252, 1/52], index=0, 
                                   format_func=lambda x: f"1/{int(1/x)} jours")
            
            simulate_button = st.button("⚡ Simuler Jump Diffusion", type="primary")
        
        if simulate_button:
            try:
                with st.spinner("Simulation du modèle Jump Diffusion..."):
                    model = JumpDiffusionModel(spot, mu, sigma, lamb, mu_j, sigma_j, dt_option)
                    paths = model.simulate_paths(T, n_paths)
                    
                    time_steps = np.linspace(0, T, paths.shape[0])
                    
                    with col2:
                        fig = plot_paths(time_steps, paths, 
                                       f"Simulation Jump Diffusion ({n_paths} chemins)", 
                                       "Prix de l'action")
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erreur lors de la simulation Jump Diffusion: {e}")

    elif model_type == "🌊 Volatilité Locale":
        st.header("🌊 Modèle de Volatilité Locale")
        st.markdown("**Équation:** dS = σ(S,t) S dW")
        st.code("from models.local_volatility.local_vol import LocalVolatilityModel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres")
            spot = st.number_input("Prix initial (S₀)", value=100.0, min_value=1.0, step=1.0)
            vol_type = st.selectbox("Type de surface de volatilité", 
                                  ["Constante", "Dépendante du temps", "Dépendante du niveau"])
            
            if vol_type == "Constante":
                base_vol = st.slider("Volatilité constante", 0.1, 1.0, 0.2, 0.01)
                local_vol_surface = lambda S, t: base_vol
            elif vol_type == "Dépendante du temps":
                base_vol = st.slider("Volatilité de base", 0.1, 1.0, 0.2, 0.01)
                time_decay = st.slider("Facteur de décroissance temporelle", 0.0, 2.0, 0.5, 0.1)
                local_vol_surface = lambda S, t: base_vol * np.exp(-time_decay * t)
            else:  # Dépendante du niveau
                base_vol = st.slider("Volatilité de base", 0.1, 1.0, 0.2, 0.01)
                level_factor = st.slider("Facteur de niveau", 0.0, 2.0, 0.5, 0.1)
                local_vol_surface = lambda S, t: base_vol * (1 + level_factor * (S / spot - 1))
            
            dt_option = st.selectbox("Pas de temps", [1/252, 1/52], index=0, 
                                   format_func=lambda x: f"1/{int(1/x)} jours")
            
            simulate_button = st.button("🌊 Simuler Volatilité Locale", type="primary")
        
        if simulate_button:
            try:
                with st.spinner("Simulation du modèle de volatilité locale..."):
                    model = LocalVolatilityModel(spot, local_vol_surface, dt_option)
                    paths = model.simulate_paths(T, n_paths)
                    
                    time_steps = np.linspace(0, T, paths.shape[0])
                    
                    with col2:
                        fig = plot_paths(time_steps, paths, 
                                       f"Simulation Volatilité Locale ({n_paths} chemins)", 
                                       "Prix de l'action")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Visualisation de la surface de volatilité
                st.subheader("🎯 Surface de Volatilité Locale")
                S_range = np.linspace(spot * 0.5, spot * 1.5, 50)
                t_range = np.linspace(0, T, 50)
                S_grid, t_grid = np.meshgrid(S_range, t_range)
                vol_surface = np.array([[local_vol_surface(S, t) for S in S_range] for t in t_range])
                
                fig_surface = go.Figure(data=[go.Surface(z=vol_surface, x=S_grid, y=t_grid)])
                fig_surface.update_layout(
                    title='Surface de Volatilité Locale σ(S,t)',
                    scene=dict(
                        xaxis_title='Prix (S)',
                        yaxis_title='Temps (t)',
                        zaxis_title='Volatilité σ(S,t)'
                    ),
                    height=500
                )
                st.plotly_chart(fig_surface, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erreur lors de la simulation Volatilité Locale: {e}")

    elif model_type == "🏦 Hull-White":
        st.header("🏦 Modèle de Taux d'Intérêt Hull-White")
        st.markdown("**Équation:** dr = [θ(t) - ar] dt + σ dW")
        st.code("from models.hull_white.hull_white import HullWhiteModel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres")
            r0 = st.slider("Taux initial (r₀)", 0.0, 0.2, 0.05, 0.001)
            a = st.slider("Retour à la moyenne (a)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatilité (σ)", 0.001, 0.1, 0.02, 0.001)
            theta = st.slider("Taux long terme (θ)", 0.0, 0.2, 0.05, 0.001)
            dt_option = st.selectbox("Pas de temps", [1/252, 1/52], index=0, 
                                   format_func=lambda x: f"1/{int(1/x)} jours")
            
            simulate_button = st.button("🏦 Simuler Taux d'Intérêt", type="primary")
        
        if simulate_button:
            try:
                with st.spinner("Simulation du modèle Hull-White..."):
                    model = HullWhiteModel(r0, a, sigma, theta, dt_option)
                    rates = model.simulate_short_rate(T, n_paths)
                    
                    time_steps = np.linspace(0, T, rates.shape[0])
                    
                    with col2:
                        # Convert to percentage for display
                        rates_percent = rates * 100
                        fig = plot_paths(time_steps, rates_percent, 
                                       f"Simulation Hull-White ({n_paths} chemins)", 
                                       "Taux d'intérêt (%)")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Pricing d'obligations zéro-coupon
                st.subheader("💰 Pricing d'Obligations Zéro-Coupon")
                bond_col1, bond_col2, bond_col3 = st.columns(3)
                
                with bond_col1:
                    current_rate = st.slider("Taux actuel", 0.0, 0.2, r0, 0.001)
                    bond_maturity = st.slider("Maturité de l'obligation", 0.1, 10.0, 1.0, 0.1)
                    current_time = st.slider("Temps actuel", 0.0, T, 0.0, 0.1)
                
                bond_price = model.zero_coupon_bond_price(current_rate, bond_maturity, current_time)
                implied_yield = -np.log(bond_price) / bond_maturity
                
                with bond_col2:
                    st.metric("Prix de l'obligation ZC", f"{bond_price:.4f}")
                    st.metric("Rendement implicite", f"{implied_yield * 100:.3f}%")
                
                with bond_col3:
                    # Courbe des taux
                    maturities = np.linspace(0.1, 10, 50)
                    bond_prices = [model.zero_coupon_bond_price(current_rate, mat, current_time) 
                                 for mat in maturities]
                    yields = [-np.log(price) / mat for price, mat in zip(bond_prices, maturities)]
                    
                    fig_yield = go.Figure()
                    fig_yield.add_trace(go.Scatter(x=maturities, y=np.array(yields) * 100, 
                                                 mode='lines', name='Courbe des taux'))
                    fig_yield.update_layout(title="Courbe des Taux Zéro-Coupon", 
                                          xaxis_title="Maturité (années)", 
                                          yaxis_title="Rendement (%)")
                    st.plotly_chart(fig_yield, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erreur lors de la simulation Hull-White: {e}")

    elif model_type == "💰 Pricing d'Options & Greeks":
        st.header("💰 Pricing d'Options Black-Scholes & Greeks")
        st.markdown("**Analyse complète des options avec calcul des Greeks**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres de l'Option")
            S = st.number_input("Prix actuel (S)", value=100.0, min_value=1.0, step=1.0)
            K = st.number_input("Prix d'exercice (K)", value=100.0, min_value=1.0, step=1.0)
            T_option = st.slider("Temps jusqu'à expiration (années)", 0.01, 5.0, 0.25, 0.01)
            r = st.slider("Taux sans risque", 0.0, 0.2, 0.05, 0.001)
            sigma = st.slider("Volatilité", 0.01, 2.0, 0.2, 0.01)
            
            # Calcul des prix et Greeks
            call_price = black_scholes_call(S, K, T_option, r, sigma)
            put_price = black_scholes_put(S, K, T_option, r, sigma)
            greeks = calculate_greeks(S, K, T_option, r, sigma)
            
            st.subheader("Prix des Options")
            st.metric("Option Call", f"{call_price:.3f} €")
            st.metric("Option Put", f"{put_price:.3f} €")
            parity_check = call_price - put_price - (S - K * np.exp(-r * T_option))
            st.metric("Vérification Parité Call-Put", f"{parity_check:.6f}")
        
        with col2:
            # Affichage des Greeks
            st.subheader("📊 Les Greeks")
            
            greek_col1, greek_col2 = st.columns(2)
            
            with greek_col1:
                st.metric("Delta Call", f"{greeks['call_delta']:.4f}")
                st.metric("Delta Put", f"{greeks['put_delta']:.4f}")
                st.metric("Gamma", f"{greeks['gamma']:.4f}")
                st.metric("Vega", f"{greeks['vega']:.3f}")
            
            with greek_col2:
                st.metric("Theta Call", f"{greeks['call_theta']:.3f}")
                st.metric("Theta Put", f"{greeks['put_theta']:.3f}")
                st.metric("Rho Call", f"{greeks['call_rho']:.3f}")
                st.metric("Rho Put", f"{greeks['put_rho']:.3f}")
        
        # Analyse de sensibilité
        st.subheader("📈 Analyse de Sensibilité")
        
        # Prix vs Spot
        spot_range = np.linspace(S * 0.7, S * 1.3, 100)
        call_prices_spot = [black_scholes_call(s, K, T_option, r, sigma) for s in spot_range]
        put_prices_spot = [black_scholes_put(s, K, T_option, r, sigma) for s in spot_range]
        
        fig_spot = go.Figure()
        fig_spot.add_trace(go.Scatter(x=spot_range, y=call_prices_spot, name='Call', 
                                    line=dict(color='green', width=2)))
        fig_spot.add_trace(go.Scatter(x=spot_range, y=put_prices_spot, name='Put', 
                                    line=dict(color='red', width=2)))
        fig_spot.add_vline(x=S, line_dash="dash", annotation_text="Prix actuel", 
                         line_color="blue")
        fig_spot.add_vline(x=K, line_dash="dot", annotation_text="Strike", 
                         line_color="orange")
        fig_spot.update_layout(title="Prix des Options vs Prix Spot", 
                             xaxis_title="Prix Spot", yaxis_title="Prix de l'Option")
        st.plotly_chart(fig_spot, use_container_width=True)
        
        # Sensibilité à la volatilité
        vol_range = np.linspace(0.05, 1.0, 100)
        call_prices_vol = [black_scholes_call(S, K, T_option, r, v) for v in vol_range]
        put_prices_vol = [black_scholes_put(S, K, T_option, r, v) for v in vol_range]
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=vol_range, y=call_prices_vol, name='Call', 
                                   line=dict(color='green', width=2)))
        fig_vol.add_trace(go.Scatter(x=vol_range, y=put_prices_vol, name='Put', 
                                   line=dict(color='red', width=2)))
        fig_vol.add_vline(x=sigma, line_dash="dash", annotation_text="Vol actuelle", 
                        line_color="blue")
        fig_vol.update_layout(title="Prix des Options vs Volatilité", 
                            xaxis_title="Volatilité", yaxis_title="Prix de l'Option")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Décroissance temporelle
        if T_option > 0.01:
            time_range = np.linspace(0.01, T_option, 100)
            call_prices_time = [black_scholes_call(S, K, t, r, sigma) for t in time_range]
            put_prices_time = [black_scholes_put(S, K, t, r, sigma) for t in time_range]
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=time_range, y=call_prices_time, name='Call', 
                                        line=dict(color='green', width=2)))
            fig_time.add_trace(go.Scatter(x=time_range, y=put_prices_time, name='Put', 
                                        line=dict(color='red', width=2)))
            fig_time.add_vline(x=T_option, line_dash="dash", annotation_text="TTM actuel", 
                             line_color="blue")
            fig_time.update_layout(title="Prix des Options vs Temps jusqu'à Maturité", 
                                 xaxis_title="Temps jusqu'à Maturité", yaxis_title="Prix de l'Option")
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Surface de volatilité implicite (simulation)
        st.subheader("🎯 Surface de Volatilité Implicite (Exemple)")
        strikes = np.linspace(S * 0.8, S * 1.2, 20)
        times = np.linspace(0.1, 2.0, 20)
        
        # Simulation d'une surface de vol implicite
        K_grid, T_grid = np.meshgrid(strikes, times)
        vol_surface = 0.2 + 0.1 * np.sin(K_grid / S) + 0.05 * np.cos(T_grid)
        
        fig_surface = go.Figure(data=[go.Surface(z=vol_surface, x=K_grid, y=T_grid, 
                                               colorscale='Viridis')])
        fig_surface.update_layout(
            title='Surface de Volatilité Implicite (Exemple)',
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Temps jusqu\'à Maturité',
                zaxis_title='Volatilité Implicite'
            ),
            height=500
        )
        st.plotly_chart(fig_surface, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🏛️ Dashboard des Modèles de Finance Quantitative</p>
        <p>Développé avec Streamlit et vos modèles quantitatifs existants</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
