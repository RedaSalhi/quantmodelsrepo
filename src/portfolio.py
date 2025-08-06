import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import streamlit as st
from typing import Tuple, Dict, List, Optional

class PortfolioOptimizer:
    """Modern Portfolio Theory implementation with optimization methods"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns data
        risk_free_rate : float
            Risk-free rate (annualized)
        """
        self.returns = returns.dropna()
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        
        # Calculate statistics
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        Tuple[float, float, float]
            Expected return, volatility, Sharpe ratio
        """
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def minimize_volatility(self) -> Dict:
        """
        Find minimum variance portfolio
        
        Returns:
        --------
        Dict
            Portfolio results including weights and performance
        """
        def objective(weights):
            return self.portfolio_performance(weights)[1]  # Minimize volatility
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))  # Long-only
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}
    
    def maximize_sharpe_ratio(self) -> Dict:
        """
        Find maximum Sharpe ratio portfolio (tangency portfolio)
        
        Returns:
        --------
        Dict
            Portfolio results including weights and performance
        """
        def objective(weights):
            return -self.portfolio_performance(weights)[2]  # Minimize negative Sharpe ratio
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))  # Long-only
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}
    
    def efficient_frontier(self, n_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios
        
        Parameters:
        -----------
        n_portfolios : int
            Number of portfolios to generate
            
        Returns:
        --------
        pd.DataFrame
            Efficient frontier data
        """
        # Find min and max return portfolios
        min_vol_port = self.minimize_volatility()
        max_sharpe_port = self.maximize_sharpe_ratio()
        
        if not (min_vol_port['success'] and max_sharpe_port['success']):
            st.error("Failed to compute efficient frontier")
            return pd.DataFrame()
        
        # Generate target returns
        min_ret = min_vol_port['expected_return']
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            portfolio = self._optimize_for_target_return(target_ret)
            if portfolio['success']:
                efficient_portfolios.append({
                    'Return': portfolio['expected_return'],
                    'Volatility': portfolio['volatility'],
                    'Sharpe_Ratio': portfolio['sharpe_ratio'],
                    'Weights': portfolio['weights']
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def _optimize_for_target_return(self, target_return: float) -> Dict:
        """Optimize portfolio for a target return"""
        def objective(weights):
            return self.portfolio_performance(weights)[1]  # Minimize volatility
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - target_return}  # Target return
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False}
    
    def calculate_cml(self, market_portfolio_weights: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Capital Market Line
        
        Parameters:
        -----------
        market_portfolio_weights : np.ndarray, optional
            Market portfolio weights. If None, uses max Sharpe ratio portfolio
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Volatility and return arrays for CML
        """
        if market_portfolio_weights is None:
            max_sharpe_port = self.maximize_sharpe_ratio()
            if not max_sharpe_port['success']:
                return np.array([]), np.array([])
            market_return = max_sharpe_port['expected_return']
            market_volatility = max_sharpe_port['volatility']
        else:
            market_return, market_volatility, _ = self.portfolio_performance(market_portfolio_weights)
        
        # CML equation: E(R) = Rf + (E(Rm) - Rf) / σm * σp
        volatility_range = np.linspace(0, market_volatility * 1.5, 100)
        cml_returns = self.risk_free_rate + (market_return - self.risk_free_rate) / market_volatility * volatility_range
        
        return volatility_range, cml_returns

class PortfolioVisualizer:
    """Create visualizations for portfolio analysis"""
    
    def __init__(self, optimizer: PortfolioOptimizer):
        self.optimizer = optimizer
        
    def plot_efficient_frontier(self, efficient_frontier: pd.DataFrame, 
                               min_vol_port: Dict = None, max_sharpe_port: Dict = None,
                               market_portfolio: Dict = None) -> go.Figure:
        """
        Plot efficient frontier with special portfolios
        
        Parameters:
        -----------
        efficient_frontier : pd.DataFrame
            Efficient frontier data
        min_vol_port : Dict, optional
            Minimum variance portfolio
        max_sharpe_port : Dict, optional
            Maximum Sharpe ratio portfolio
        market_portfolio : Dict, optional
            Market portfolio (e.g., S&P 500)
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = go.Figure()
        
        # Plot efficient frontier
        if not efficient_frontier.empty:
            fig.add_trace(go.Scatter(
                x=efficient_frontier['Volatility'],
                y=efficient_frontier['Return'],
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='blue', width=3),
                marker=dict(size=4),
                hovertemplate='<b>Efficient Frontier</b><br>' +
                             'Return: %{y:.2%}<br>' +
                             'Volatility: %{x:.2%}<br>' +
                             '<extra></extra>'
            ))
        
        # Plot minimum variance portfolio
        if min_vol_port and min_vol_port.get('success'):
            fig.add_trace(go.Scatter(
                x=[min_vol_port['volatility']],
                y=[min_vol_port['expected_return']],
                mode='markers',
                name='Min Variance Portfolio',
                marker=dict(color='green', size=12, symbol='star'),
                hovertemplate='<b>Min Variance Portfolio</b><br>' +
                             'Return: %{y:.2%}<br>' +
                             'Volatility: %{x:.2%}<br>' +
                             f"Sharpe Ratio: {min_vol_port['sharpe_ratio']:.3f}<br>" +
                             '<extra></extra>'
            ))
        
        # Plot maximum Sharpe ratio portfolio
        if max_sharpe_port and max_sharpe_port.get('success'):
            fig.add_trace(go.Scatter(
                x=[max_sharpe_port['volatility']],
                y=[max_sharpe_port['expected_return']],
                mode='markers',
                name='Max Sharpe Portfolio',
                marker=dict(color='red', size=12, symbol='diamond'),
                hovertemplate='<b>Max Sharpe Portfolio</b><br>' +
                             'Return: %{y:.2%}<br>' +
                             'Volatility: %{x:.2%}<br>' +
                             f"Sharpe Ratio: {max_sharpe_port['sharpe_ratio']:.3f}<br>" +
                             '<extra></extra>'
            ))
        
        # Plot market portfolio
        if market_portfolio and market_portfolio.get('success'):
            fig.add_trace(go.Scatter(
                x=[market_portfolio['volatility']],
                y=[market_portfolio['expected_return']],
                mode='markers',
                name='Market Portfolio',
                marker=dict(color='purple', size=12, symbol='square'),
                hovertemplate='<b>Market Portfolio</b><br>' +
                             'Return: %{y:.2%}<br>' +
                             'Volatility: %{x:.2%}<br>' +
                             f"Sharpe Ratio: {market_portfolio['sharpe_ratio']:.3f}<br>" +
                             '<extra></extra>'
            ))
        
        # Plot Capital Market Line
        if max_sharpe_port and max_sharpe_port.get('success'):
            vol_cml, ret_cml = self.optimizer.calculate_cml()
            if len(vol_cml) > 0:
                fig.add_trace(go.Scatter(
                    x=vol_cml,
                    y=ret_cml,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='orange', width=2, dash='dash'),
                    hovertemplate='<b>Capital Market Line</b><br>' +
                                 'Return: %{y:.2%}<br>' +
                                 'Volatility: %{x:.2%}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title='Efficient Frontier Analysis',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=600,
            showlegend=True
        )
        
        # Format axes as percentages
        fig.update_xaxis(tickformat='.1%')
        fig.update_yaxis(tickformat='.1%')
        
        return fig
    
    def plot_portfolio_weights(self, weights: np.ndarray, portfolio_name: str = "Portfolio") -> go.Figure:
        """
        Plot portfolio weights as pie chart
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        portfolio_name : str
            Name of the portfolio
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        # Filter out very small weights for cleaner visualization
        min_weight = 0.01
        filtered_weights = []
        filtered_assets = []
        other_weight = 0
        
        for i, weight in enumerate(weights):
            if weight >= min_weight:
                filtered_weights.append(weight)
                filtered_assets.append(self.optimizer.assets[i])
            else:
                other_weight += weight
        
        if other_weight > 0:
            filtered_weights.append(other_weight)
            filtered_assets.append('Other')
        
        fig = go.Figure(data=[go.Pie(
            labels=filtered_assets,
            values=filtered_weights,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br><extra></extra>'
        )])
        
        fig.update_layout(
            title=f'{portfolio_name} - Asset Allocation',
            template='plotly_white',
            width=600,
            height=400
        )
        
        return fig
    
    def plot_correlation_matrix(self) -> go.Figure:
        """
        Plot correlation matrix heatmap
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        corr_matrix = self.optimizer.returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            template='plotly_white',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_returns_distribution(self) -> go.Figure:
        """
        Plot returns distribution for each asset
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Distributions', 'Rolling Volatility', 
                          'Cumulative Returns', 'Risk-Return Scatter'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Returns distribution
        for asset in self.optimizer.assets[:4]:  # Limit to 4 assets for clarity
            returns_data = self.optimizer.returns[asset] * 100
            fig.add_trace(
                go.Histogram(x=returns_data, name=f'{asset} Returns', 
                           opacity=0.7, nbinsx=30),
                row=1, col=1
            )
        
        # Rolling volatility (30-day)
        rolling_vol = self.optimizer.returns.rolling(30).std() * np.sqrt(252) * 100
        for asset in self.optimizer.assets[:4]:
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol[asset],
                          name=f'{asset} Vol', mode='lines'),
                row=1, col=2
            )
        
        # Cumulative returns
        cumulative_returns = (1 + self.optimizer.returns).cumprod() - 1
        for asset in self.optimizer.assets[:4]:
            fig.add_trace(
                go.Scatter(x=cumulative_returns.index, y=cumulative_returns[asset] * 100,
                          name=f'{asset} Cumulative', mode='lines'),
                row=2, col=1
            )
        
        # Risk-return scatter
        annual_returns = self.optimizer.mean_returns * 100
        annual_volatility = np.sqrt(np.diag(self.optimizer.cov_matrix)) * 100
        
        fig.add_trace(
            go.Scatter(x=annual_volatility, y=annual_returns,
                      mode='markers+text', 
                      text=self.optimizer.assets,
                      textposition='top center',
                      marker=dict(size=10, color='red'),
                      name='Assets'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            template='plotly_white',
            title_text="Portfolio Analysis Dashboard"
        )
        
        return fig

class RiskMetrics:
    """Calculate various risk metrics for portfolios"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk using historical method"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
            else:
                current_drawdown_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    @staticmethod
    def calculate_tracking_error(portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        return (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        return (excess_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
