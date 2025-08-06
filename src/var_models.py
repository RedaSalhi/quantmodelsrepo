import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm, t
import streamlit as st
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class VaRCalculator:
    """Calculate Value at Risk using various methods"""
    
    def __init__(self, returns: pd.DataFrame, portfolio_weights: np.ndarray = None):
        """
        Initialize VaR calculator
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns data
        portfolio_weights : np.ndarray, optional
            Portfolio weights. If None, equal weights assumed
        """
        self.returns = returns.dropna()
        
        if portfolio_weights is None:
            self.portfolio_weights = np.array([1/len(returns.columns)] * len(returns.columns))
        else:
            self.portfolio_weights = portfolio_weights
            
        # Calculate portfolio returns
        self.portfolio_returns = (self.returns * self.portfolio_weights).sum(axis=1)
        self.portfolio_value = 1000000  # Default $1M portfolio
        
    def parametric_var(self, confidence_levels: List[float] = [0.01, 0.05, 0.10],
                      distribution: str = 'normal') -> Dict[str, float]:
        """
        Calculate parametric VaR using variance-covariance method
        
        Parameters:
        -----------
        confidence_levels : List[float]
            Confidence levels for VaR calculation
        distribution : str
            Distribution assumption ('normal' or 't-distribution')
            
        Returns:
        --------
        Dict[str, float]
            VaR values for different confidence levels
        """
        mean_return = self.portfolio_returns.mean()
        std_return = self.portfolio_returns.std()
        
        var_results = {}
        
        for alpha in confidence_levels:
            if distribution == 'normal':
                var_percentile = norm.ppf(alpha, mean_return, std_return)
            else:  # t-distribution
                # Estimate degrees of freedom
                df = len(self.portfolio_returns) - 1
                var_percentile = t.ppf(alpha, df, mean_return, std_return)
            
            var_dollar = var_percentile * self.portfolio_value
            var_results[f'VaR_{int(alpha*100)}%'] = {
                'percentage': var_percentile,
                'dollar': var_dollar,
                'method': f'Parametric ({distribution})'
            }
        
        return var_results
    
    def historical_var(self, confidence_levels: List[float] = [0.01, 0.05, 0.10]) -> Dict[str, float]:
        """
        Calculate historical VaR using empirical distribution
        
        Parameters:
        -----------
        confidence_levels : List[float]
            Confidence levels for VaR calculation
            
        Returns:
        --------
        Dict[str, float]
            VaR values for different confidence levels
        """
        var_results = {}
        
        for alpha in confidence_levels:
            var_percentile = np.percentile(self.portfolio_returns, alpha * 100)
            var_dollar = var_percentile * self.portfolio_value
            
            var_results[f'VaR_{int(alpha*100)}%'] = {
                'percentage': var_percentile,
                'dollar': var_dollar,
                'method': 'Historical Simulation'
            }
        
        return var_results
    
    def monte_carlo_var(self, confidence_levels: List[float] = [0.01, 0.05, 0.10],
                       n_simulations: int = 10000, time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Monte Carlo VaR using simulated returns
        
        Parameters:
        -----------
        confidence_levels : List[float]
            Confidence levels for VaR calculation
        n_simulations : int
            Number of Monte Carlo simulations
        time_horizon : int
            Time horizon in days
            
        Returns:
        --------
        Dict[str, float]
            VaR values for different confidence levels
        """
        # Calculate portfolio statistics
        mean_return = self.portfolio_returns.mean()
        std_return = self.portfolio_returns.std()
        
        # Adjust for time horizon
        mean_return_adjusted = mean_return * time_horizon
        std_return_adjusted = std_return * np.sqrt(time_horizon)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return_adjusted, std_return_adjusted, n_simulations
        )
        
        var_results = {}
        
        for alpha in confidence_levels:
            var_percentile = np.percentile(simulated_returns, alpha * 100)
            var_dollar = var_percentile * self.portfolio_value
            
            var_results[f'VaR_{int(alpha*100)}%'] = {
                'percentage': var_percentile,
                'dollar': var_dollar,
                'method': f'Monte Carlo ({n_simulations:,} sims)',
                'simulated_returns': simulated_returns
            }
        
        return var_results
    
    def calculate_expected_shortfall(self, confidence_levels: List[float] = [0.01, 0.05, 0.10],
                                   method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Parameters:
        -----------
        confidence_levels : List[float]
            Confidence levels for ES calculation
        method : str
            Method to use ('historical', 'parametric', 'monte_carlo')
            
        Returns:
        --------
        Dict[str, float]
            Expected Shortfall values
        """
        es_results = {}
        
        if method == 'historical':
            returns_data = self.portfolio_returns
        elif method == 'monte_carlo':
            mc_results = self.monte_carlo_var(confidence_levels)
            returns_data = mc_results[f'VaR_{int(confidence_levels[0]*100)}%']['simulated_returns']
        else:  # parametric
            mean_return = self.portfolio_returns.mean()
            std_return = self.portfolio_returns.std()
            np.random.seed(42)
            returns_data = np.random.normal(mean_return, std_return, 10000)
        
        for alpha in confidence_levels:
            var_threshold = np.percentile(returns_data, alpha * 100)
            tail_returns = returns_data[returns_data <= var_threshold]
            es_percentage = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
            es_dollar = es_percentage * self.portfolio_value
            
            es_results[f'ES_{int(alpha*100)}%'] = {
                'percentage': es_percentage,
                'dollar': es_dollar,
                'method': method.title()
            }
        
        return es_results
    
    def backtesting_kupiec(self, var_values: Dict, actual_returns: pd.Series) -> Dict:
        """
        Kupiec backtest for VaR model validation
        
        Parameters:
        -----------
        var_values : Dict
            VaR values from model
        actual_returns : pd.Series
            Actual historical returns for testing
            
        Returns:
        --------
        Dict
            Backtest results including test statistics
        """
        results = {}
        
        for var_level, var_data in var_values.items():
            alpha = float(var_level.split('_')[1].replace('%', '')) / 100
            var_threshold = var_data['percentage']
            
            # Count violations
            violations = (actual_returns <= var_threshold).sum()
            n_observations = len(actual_returns)
            expected_violations = n_observations * alpha
            
            # Kupiec likelihood ratio test
            if violations > 0 and violations < n_observations:
                lr_stat = 2 * (
                    violations * np.log(violations / expected_violations) +
                    (n_observations - violations) * np.log((n_observations - violations) / (n_observations - expected_violations))
                )
                p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
            else:
                lr_stat = np.inf
                p_value = 0.0
            
            results[var_level] = {
                'violations': violations,
                'expected_violations': expected_violations,
                'violation_rate': violations / n_observations,
                'expected_rate': alpha,
                'lr_statistic': lr_stat,
                'p_value': p_value,
                'model_adequate': p_value > 0.05
            }
        
        return results
    
    def stress_testing(self, stress_scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing on portfolio
        
        Parameters:
        -----------
        stress_scenarios : Dict[str, float]
            Stress scenarios with return shocks
            
        Returns:
        --------
        Dict[str, float]
            Portfolio losses under stress scenarios
        """
        stress_results = {}
        
        for scenario_name, return_shock in stress_scenarios.items():
            portfolio_loss = return_shock * self.portfolio_value
            stress_results[scenario_name] = {
                'return_shock': return_shock,
                'portfolio_loss': portfolio_loss,
                'loss_percentage': return_shock * 100
            }
        
        return stress_results

class VaRVisualizer:
    """Create visualizations for VaR analysis"""
    
    def __init__(self, var_calculator: VaRCalculator):
        self.var_calculator = var_calculator
    
    def plot_var_comparison(self, parametric_var: Dict, historical_var: Dict, 
                           monte_carlo_var: Dict) -> go.Figure:
        """
        Compare VaR estimates across different methods
        
        Parameters:
        -----------
        parametric_var : Dict
            Parametric VaR results
        historical_var : Dict
            Historical VaR results
        monte_carlo_var : Dict
            Monte Carlo VaR results
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        confidence_levels = ['VaR_1%', 'VaR_5%', 'VaR_10%']
        methods = ['Parametric', 'Historical', 'Monte Carlo']
        
        var_data = {
            'Parametric': [abs(parametric_var[level]['dollar']) for level in confidence_levels],
            'Historical': [abs(historical_var[level]['dollar']) for level in confidence_levels],
            'Monte Carlo': [abs(monte_carlo_var[level]['dollar']) for level in confidence_levels]
        }
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, method in enumerate(methods):
            fig.add_trace(go.Bar(
                name=method,
                x=[level.replace('VaR_', '').replace('%', '% VaR') for level in confidence_levels],
                y=var_data[method],
                marker_color=colors[i],
                text=[f'${val:,.0f}' for val in var_data[method]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='VaR Comparison Across Methods',
            xaxis_title='Confidence Level',
            yaxis_title='Value at Risk ($)',
            barmode='group',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_return_distribution(self, show_var: bool = True, 
                                confidence_level: float = 0.05) -> go.Figure:
        """
        Plot portfolio return distribution with VaR overlay
        
        Parameters:
        -----------
        show_var : bool
            Whether to show VaR lines
        confidence_level : float
            Confidence level for VaR
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        returns = self.var_calculator.portfolio_returns * 100  # Convert to percentage
        
        fig = go.Figure()
        
        # Histogram of returns
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Portfolio Returns',
            opacity=0.7,
            marker_color='lightblue',
            histnorm='probability density'
        ))
        
        # Normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        if show_var:
            # Calculate and show VaR
            historical_var = np.percentile(returns, confidence_level * 100)
            parametric_var = stats.norm.ppf(confidence_level, returns.mean(), returns.std())
            
            # Historical VaR line
            fig.add_vline(
                x=historical_var,
                line=dict(color='orange', width=3, dash='dash'),
                annotation_text=f'Historical VaR ({confidence_level*100:.0f}%): {historical_var:.2f}%',
                annotation_position='top'
            )
            
            # Parametric VaR line
            fig.add_vline(
                x=parametric_var,
                line=dict(color='green', width=3, dash='dot'),
                annotation_text=f'Parametric VaR ({confidence_level*100:.0f}%): {parametric_var:.2f}%',
                annotation_position='bottom'
            )
        
        fig.update_layout(
            title='Portfolio Return Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Density',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_var_time_series(self, window_size: int = 252) -> go.Figure:
        """
        Plot rolling VaR over time
        
        Parameters:
        -----------
        window_size : int
            Rolling window size in days
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        returns = self.var_calculator.portfolio_returns
        
        # Calculate rolling VaR
        rolling_var_5 = returns.rolling(window_size).quantile(0.05) * 100
        rolling_var_1 = returns.rolling(window_size).quantile(0.01) * 100
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window_size).std() * np.sqrt(252) * 100
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Portfolio Returns', 'Rolling VaR (5% & 1%)', 'Rolling Volatility'),
            vertical_spacing=0.08
        )
        
        # Portfolio returns
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns * 100,
            mode='lines',
            name='Daily Returns',
            line=dict(color='blue', width=1),
            opacity=0.7
        ), row=1, col=1)
        
        # Rolling VaR
        fig.add_trace(go.Scatter(
            x=rolling_var_5.index,
            y=rolling_var_5,
            mode='lines',
            name='5% VaR',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=rolling_var_1.index,
            y=rolling_var_1,
            mode='lines',
            name='1% VaR',
            line=dict(color='darkred', width=2)
        ), row=2, col=1)
        
        # Rolling volatility
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='Volatility',
            line=dict(color='green', width=2)
        ), row=3, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_white',
            title=f'Time-Varying Risk Metrics ({window_size}-day rolling window)',
            showlegend=True
        )
        
        fig.update_yaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='VaR (%)', row=2, col=1)
        fig.update_yaxes(title_text='Volatility (%)', row=3, col=1)
        fig.update_xaxes(title_text='Date', row=3, col=1)
        
        return fig
    
    def plot_monte_carlo_simulation(self, n_simulations: int = 1000, 
                                   time_horizon: int = 30) -> go.Figure:
        """
        Plot Monte Carlo simulation results
        
        Parameters:
        -----------
        n_simulations : int
            Number of simulation paths to display
        time_horizon : int
            Time horizon in days
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        # Portfolio statistics
        mean_return = self.var_calculator.portfolio_returns.mean()
        std_return = self.var_calculator.portfolio_returns.std()
        
        # Generate simulation paths
        np.random.seed(42)
        dt = 1
        price_paths = np.zeros((time_horizon + 1, n_simulations))
        price_paths[0] = 100  # Starting at $100
        
        for t in range(1, time_horizon + 1):
            random_shocks = np.random.normal(0, 1, n_simulations)
            price_paths[t] = price_paths[t-1] * (
                1 + mean_return * dt + std_return * np.sqrt(dt) * random_shocks
            )
        
        fig = go.Figure()
        
        # Plot simulation paths (show subset for clarity)
        time_axis = np.arange(time_horizon + 1)
        show_paths = min(50, n_simulations)
        
        for i in range(show_paths):
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=price_paths[:, i],
                mode='lines',
                line=dict(color='lightblue', width=0.5),
                opacity=0.3,
                showlegend=False,
                hovertemplate='Day %{x}<br>Value: $%{y:.2f}<extra></extra>'
            ))
        
        # Add mean path
        mean_path = price_paths.mean(axis=1)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color='red', width=3)
        ))
        
        # Add percentile bands
        percentiles = [5, 25, 75, 95]
        colors = ['red', 'orange', 'orange', 'red']
        names = ['5th Percentile', '25th Percentile', '75th Percentile', '95th Percentile']
        
        for i, (p, color, name) in enumerate(zip(percentiles, colors, names)):
            percentile_path = np.percentile(price_paths, p, axis=1)
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=percentile_path,
                mode='lines',
                name=name,
                line=dict(color=color, width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f'Monte Carlo Simulation ({n_simulations:,} paths, {time_horizon} days)',
            xaxis_title='Days',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_backtest_results(self, backtest_results: Dict) -> go.Figure:
        """
        Plot VaR backtest results
        
        Parameters:
        -----------
        backtest_results : Dict
            Results from backtesting
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        var_levels = list(backtest_results.keys())
        violation_rates = [backtest_results[level]['violation_rate'] * 100 
                          for level in var_levels]
        expected_rates = [backtest_results[level]['expected_rate'] * 100 
                         for level in var_levels]
        p_values = [backtest_results[level]['p_value'] for level in var_levels]
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Violation Rate Comparison', 'P-Values (Kupiec Test)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Violation rates
        x_labels = [level.replace('VaR_', '').replace('%', '% VaR') for level in var_levels]
        
        fig.add_trace(go.Bar(
            x=x_labels,
            y=expected_rates,
            name='Expected Rate',
            marker_color='lightblue',
            text=[f'{rate:.1f}%' for rate in expected_rates],
            textposition='auto'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=x_labels,
            y=violation_rates,
            name='Actual Rate',
            marker_color='red',
            text=[f'{rate:.1f}%' for rate in violation_rates],
            textposition='auto'
        ), row=1, col=1)
        
        # P-values
        colors = ['green' if p > 0.05 else 'red' for p in p_values]
        fig.add_trace(go.Bar(
            x=x_labels,
            y=p_values,
            name='P-Value',
            marker_color=colors,
            text=[f'{p:.3f}' for p in p_values],
            textposition='auto',
            showlegend=False
        ), row=1, col=2)
        
        # Add significance line
        fig.add_hline(
            y=0.05, line_dash="dash", line_color="red",
            annotation_text="5% Significance Level",
            row=1, col=2
        )
        
        fig.update_layout(
            title='VaR Model Backtesting Results',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        fig.update_yaxes(title_text='Violation Rate (%)', row=1, col=1)
        fig.update_yaxes(title_text='P-Value', row=1, col=2)
        
        return fig

class ComponentVaR:
    """Calculate Component VaR and Marginal VaR for portfolio"""
    
    def __init__(self, returns: pd.DataFrame, weights: np.ndarray):
        """
        Initialize Component VaR calculator
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        weights : np.ndarray
            Portfolio weights
        """
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = (returns * weights).sum(axis=1)
        self.cov_matrix = returns.cov() * 252  # Annualized
        
    def calculate_component_var(self, confidence_level: float = 0.05) -> pd.DataFrame:
        """
        Calculate Component VaR for each asset
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for VaR calculation
            
        Returns:
        --------
        pd.DataFrame
            Component VaR results
        """
        # Portfolio VaR
        portfolio_var = np.percentile(self.portfolio_returns, confidence_level * 100)
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        
        # Marginal VaR (derivative of portfolio VaR w.r.t. weights)
        marginal_var = np.dot(self.cov_matrix, self.weights) / portfolio_volatility
        
        # Component VaR
        component_var = self.weights * marginal_var * abs(portfolio_var) / portfolio_volatility
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Asset': self.returns.columns,
            'Weight': self.weights,
            'Marginal_VaR': marginal_var,
            'Component_VaR': component_var,
            'Percent_Contribution': component_var / abs(portfolio_var) * 100
        })
        
        return results.sort_values('Component_VaR', ascending=False)
    
    def calculate_incremental_var(self, confidence_level: float = 0.05, 
                                 delta_weight: float = 0.01) -> pd.DataFrame:
        """
        Calculate Incremental VaR for small position changes
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for VaR calculation
        delta_weight : float
            Small change in weight for calculation
            
        Returns:
        --------
        pd.DataFrame
            Incremental VaR results
        """
        base_var = np.percentile(self.portfolio_returns, confidence_level * 100)
        incremental_vars = []
        
        for i in range(len(self.weights)):
            # Create modified weights
            modified_weights = self.weights.copy()
            modified_weights[i] += delta_weight
            modified_weights = modified_weights / modified_weights.sum()  # Renormalize
            
            # Calculate new portfolio returns and VaR
            new_portfolio_returns = (self.returns * modified_weights).sum(axis=1)
            new_var = np.percentile(new_portfolio_returns, confidence_level * 100)
            
            incremental_var = new_var - base_var
            incremental_vars.append(incremental_var)
        
        results = pd.DataFrame({
            'Asset': self.returns.columns,
            'Current_Weight': self.weights,
            'Weight_Change': delta_weight,
            'Incremental_VaR': incremental_vars,
            'IVaR_per_1%': np.array(incremental_vars) / delta_weight
        })
        
        return results.sort_values('Incremental_VaR', ascending=True)

class StressTestScenarios:
    """Predefined stress testing scenarios"""
    
    @staticmethod
    def get_market_scenarios() -> Dict[str, Dict[str, float]]:
        """
        Get predefined market stress scenarios
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Stress scenarios with asset-specific shocks
        """
        scenarios = {
            "2008 Financial Crisis": {
                "description": "Global financial crisis scenario",
                "equity_shock": -0.40,  # 40% equity decline
                "bond_shock": 0.15,     # 15% bond rally
                "commodity_shock": -0.30, # 30% commodity decline
                "fx_shock": 0.20        # 20% USD strength
            },
            "COVID-19 Pandemic": {
                "description": "Pandemic-induced market shock",
                "equity_shock": -0.35,
                "bond_shock": 0.10,
                "commodity_shock": -0.25,
                "fx_shock": 0.15
            },
            "Interest Rate Shock": {
                "description": "Rapid interest rate increase",
                "equity_shock": -0.15,
                "bond_shock": -0.20,
                "commodity_shock": -0.10,
                "fx_shock": 0.10
            },
            "Inflation Surge": {
                "description": "High inflation environment",
                "equity_shock": -0.20,
                "bond_shock": -0.25,
                "commodity_shock": 0.30,
                "fx_shock": -0.15
            },
            "Geopolitical Crisis": {
                "description": "Major geopolitical event",
                "equity_shock": -0.25,
                "bond_shock": 0.08,
                "commodity_shock": 0.20,
                "fx_shock": 0.12
            },
            "Technology Bubble Burst": {
                "description": "Tech sector collapse",
                "equity_shock": -0.50,  # Higher impact on equities
                "bond_shock": 0.12,
                "commodity_shock": -0.15,
                "fx_shock": 0.08
            }
        }
        return scenarios
    
    @staticmethod
    def apply_scenario_to_portfolio(returns: pd.DataFrame, weights: np.ndarray,
                                   scenario: Dict[str, float], 
                                   asset_mapping: Dict[str, str] = None) -> float:
        """
        Apply stress scenario to portfolio
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns
        weights : np.ndarray
            Portfolio weights
        scenario : Dict[str, float]
            Stress scenario shocks
        asset_mapping : Dict[str, str], optional
            Mapping of assets to shock categories
            
        Returns:
        --------
        float
            Portfolio loss under stress scenario
        """
        if asset_mapping is None:
            # Default mapping - assume all assets are equities
            asset_mapping = {asset: 'equity' for asset in returns.columns}
        
        portfolio_shock = 0
        shock_mapping = {
            'equity': scenario.get('equity_shock', 0),
            'bond': scenario.get('bond_shock', 0),
            'commodity': scenario.get('commodity_shock', 0),
            'fx': scenario.get('fx_shock', 0)
        }
        
        for i, asset in enumerate(returns.columns):
            asset_type = asset_mapping.get(asset, 'equity')
            asset_shock = shock_mapping.get(asset_type, 0)
            portfolio_shock += weights[i] * asset_shock
        
        return portfolio_shock

class RiskBudgeting:
    """Risk budgeting and risk parity methods"""
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize risk budgeting calculator
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        """
        self.returns = returns
        self.cov_matrix = returns.cov() * 252  # Annualized
        
    def equal_risk_contribution(self, max_iterations: int = 1000, 
                               tolerance: float = 1e-8) -> np.ndarray:
        """
        Calculate equal risk contribution (risk parity) weights
        
        Parameters:
        -----------
        max_iterations : int
            Maximum iterations for optimization
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        np.ndarray
            Risk parity weights
        """
        n_assets = len(self.returns.columns)
        
        def risk_budget_objective(weights):
            """Objective function for equal risk contribution"""
            weights = weights / np.sum(weights)  # Normalize
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Objective: minimize sum of squared differences from equal contribution
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Optimization constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.001, 0.999) for _ in range(n_assets))  # Small bounds to avoid division by zero
        initial_guess = np.array([1/n_assets] * n_assets)
        
        from scipy.optimize import minimize
        result = minimize(
            risk_budget_objective, 
            initial_guess, 
            method='SLSQP',
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        if result.success:
            return result.x / np.sum(result.x)  # Normalize
        else:
            st.warning("Risk parity optimization did not converge. Using equal weights.")
            return np.array([1/n_assets] * n_assets)
    
    def risk_budgeting_weights(self, risk_budgets: np.ndarray) -> np.ndarray:
        """
        Calculate weights for given risk budgets
        
        Parameters:
        -----------
        risk_budgets : np.ndarray
            Target risk budgets (should sum to 1)
            
        Returns:
        --------
        np.ndarray
            Optimal weights for risk budgeting
        """
        n_assets = len(risk_budgets)
        
        def objective(weights):
            weights = weights / np.sum(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            contrib_pct = contrib / portfolio_vol
            
            # Minimize squared differences from target budgets
            return np.sum((contrib_pct - risk_budgets) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.001, 0.999) for _ in range(n_assets))
        initial_guess = risk_budgets  # Start with risk budgets as initial weights
        
        from scipy.optimize import minimize
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x / np.sum(result.x)
        else:
            st.warning("Risk budgeting optimization failed. Using proportional allocation.")
            return risk_budgets / np.sum(risk_budgets)
