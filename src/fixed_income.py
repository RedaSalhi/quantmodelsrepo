import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

class BondPricer:
    """Bond pricing and analytics"""
    
    @staticmethod
    def bond_price(face_value: float, coupon_rate: float, yield_to_maturity: float,
                   years_to_maturity: float, frequency: int = 2) -> float:
        """
        Calculate bond price using present value formula
        
        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate
        yield_to_maturity : float
            Yield to maturity
        years_to_maturity : float
            Years until maturity
        frequency : int
            Coupon payment frequency per year
            
        Returns:
        --------
        float
            Bond price
        """
        periods = int(years_to_maturity * frequency)
        coupon_payment = (coupon_rate * face_value) / frequency
        discount_rate = yield_to_maturity / frequency
        
        if discount_rate == 0:
            return face_value + coupon_payment * periods
        
        # Present value of coupon payments
        pv_coupons = coupon_payment * (1 - (1 + discount_rate) ** (-periods)) / discount_rate
        
        # Present value of principal
        pv_principal = face_value / ((1 + discount_rate) ** periods)
        
        return pv_coupons + pv_principal
    
    @staticmethod
    def bond_duration(face_value: float, coupon_rate: float, yield_to_maturity: float,
                      years_to_maturity: float, frequency: int = 2) -> Tuple[float, float]:
        """
        Calculate Macaulay and Modified duration
        
        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate
        yield_to_maturity : float
            Yield to maturity
        years_to_maturity : float
            Years until maturity
        frequency : int
            Coupon payment frequency per year
            
        Returns:
        --------
        Tuple[float, float]
            Macaulay duration, Modified duration
        """
        periods = int(years_to_maturity * frequency)
        coupon_payment = (coupon_rate * face_value) / frequency
        discount_rate = yield_to_maturity / frequency
        
        bond_price = BondPricer.bond_price(face_value, coupon_rate, yield_to_maturity,
                                          years_to_maturity, frequency)
        
        # Calculate weighted time to cash flows
        weighted_time = 0
        for t in range(1, periods + 1):
            if t < periods:
                cash_flow = coupon_payment
            else:
                cash_flow = coupon_payment + face_value
            
            present_value = cash_flow / ((1 + discount_rate) ** t)
            weight = present_value / bond_price
            weighted_time += weight * (t / frequency)  # Convert to years
        
        macaulay_duration = weighted_time
        modified_duration = macaulay_duration / (1 + yield_to_maturity / frequency)
        
        return macaulay_duration, modified_duration
    
    @staticmethod
    def bond_convexity(face_value: float, coupon_rate: float, yield_to_maturity: float,
                       years_to_maturity: float, frequency: int = 2) -> float:
        """
        Calculate bond convexity
        
        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate
        yield_to_maturity : float
            Yield to maturity
        years_to_maturity : float
            Years until maturity
        frequency : int
            Coupon payment frequency per year
            
        Returns:
        --------
        float
            Bond convexity
        """
        periods = int(years_to_maturity * frequency)
        coupon_payment = (coupon_rate * face_value) / frequency
        discount_rate = yield_to_maturity / frequency
        
        bond_price = BondPricer.bond_price(face_value, coupon_rate, yield_to_maturity,
                                          years_to_maturity, frequency)
        
        convexity = 0
        for t in range(1, periods + 1):
            if t < periods:
                cash_flow = coupon_payment
            else:
                cash_flow = coupon_payment + face_value
            
            present_value = cash_flow / ((1 + discount_rate) ** t)
            convexity += (present_value / bond_price) * (t * (t + 1)) / ((1 + discount_rate) ** 2)
        
        return convexity / (frequency ** 2)

class YieldCurveAnalyzer:
    """Yield curve construction and analysis"""
    
    def __init__(self, yield_data: pd.DataFrame):
        """
        Initialize yield curve analyzer
        
        Parameters:
        -----------
        yield_data : pd.DataFrame
            Yield data with maturity as columns
        """
        self.yield_data = yield_data.dropna()
        self.maturities = self._parse_maturities(yield_data.columns)
        
    def _parse_maturities(self, maturity_labels: List[str]) -> np.ndarray:
        """Parse maturity labels to numeric values (in years)"""
        maturities = []
        for label in maturity_labels:
            if 'M' in label:  # Months
                months = float(label.replace('M', ''))
                maturities.append(months / 12)
            elif 'Y' in label:  # Years
                years = float(label.replace('Y', ''))
                maturities.append(years)
            else:
                # Try to parse as float (assume years)
                try:
                    maturities.append(float(label))
                except:
                    maturities.append(1.0)  # Default to 1 year
        
        return np.array(maturities)
    
    def interpolate_yield_curve(self, target_date: str = None, 
                               method: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate yield curve for smoother visualization
        
        Parameters:
        -----------
        target_date : str, optional
            Specific date for yield curve. If None, uses latest
        method : str
            Interpolation method ('linear', 'cubic')
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Interpolated maturities and yields
        """
        if target_date is None:
            yields = self.yield_data.iloc[-1].values
        else:
            yields = self.yield_data.loc[target_date].values
        
        # Remove NaN values
        valid_idx = ~np.isnan(yields)
        valid_maturities = self.maturities[valid_idx]
        valid_yields = yields[valid_idx]
        
        if len(valid_maturities) < 2:
            return valid_maturities, valid_yields
        
        # Create interpolated points
        interp_maturities = np.linspace(valid_maturities.min(), valid_maturities.max(), 100)
        
        if method == 'cubic':
            interp_func = interp1d(valid_maturities, valid_yields, kind='cubic', 
                                 bounds_error=False, fill_value='extrapolate')
        else:
            interp_func = interp1d(valid_maturities, valid_yields, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
        
        interp_yields = interp_func(interp_maturities)
        
        return interp_maturities, interp_yields
    
    def calculate_yield_curve_metrics(self, date: str = None) -> Dict[str, float]:
        """
        Calculate yield curve shape metrics
        
        Parameters:
        -----------
        date : str, optional
            Date for analysis. If None, uses latest
            
        Returns:
        --------
        Dict[str, float]
            Yield curve metrics
        """
        if date is None:
            yields = self.yield_data.iloc[-1]
        else:
            yields = self.yield_data.loc[date]
        
        # Remove NaN values
        valid_data = yields.dropna()
        valid_maturities = self.maturities[yields.index.isin(valid_data.index)]
        
        if len(valid_data) < 3:
            return {}
        
        # Find approximate indices for common metrics
        short_idx = np.argmin(np.abs(valid_maturities - 2))    # 2-year
        medium_idx = np.argmin(np.abs(valid_maturities - 10))   # 10-year
        long_idx = np.argmin(np.abs(valid_maturities - 30))     # 30-year
        
        metrics = {
            'level': valid_data.mean(),
            'slope': valid_data.iloc[long_idx] - valid_data.iloc[short_idx] if long_idx < len(valid_data) else 0,
            'curvature': 2 * valid_data.iloc[medium_idx] - valid_data.iloc[short_idx] - valid_data.iloc[long_idx] if medium_idx < len(valid_data) and long_idx < len(valid_data) else 0,
            'steepness_2_10': valid_data.iloc[medium_idx] - valid_data.iloc[short_idx] if medium_idx < len(valid_data) else 0
        }
        
        return metrics
    
    def term_structure_analysis(self) -> pd.DataFrame:
        """
        Analyze term structure evolution over time
        
        Returns:
        --------
        pd.DataFrame
            Term structure metrics over time
        """
        metrics_list = []
        
        for date in self.yield_data.index:
            date_metrics = self.calculate_yield_curve_metrics(date)
            if date_metrics:
                date_metrics['date'] = date
                metrics_list.append(date_metrics)
        
        return pd.DataFrame(metrics_list).set_index('date')

class BondPortfolioAnalyzer:
    """Analyze bond portfolio characteristics"""
    
    def __init__(self, bonds_data: List[Dict], weights: np.ndarray = None):
        """
        Initialize bond portfolio analyzer
        
        Parameters:
        -----------
        bonds_data : List[Dict]
            List of bond characteristics
        weights : np.ndarray, optional
            Portfolio weights
        """
        self.bonds_data = bonds_data
        if weights is None:
            self.weights = np.array([1/len(bonds_data)] * len(bonds_data))
        else:
            self.weights = weights
    
    def portfolio_duration(self) -> float:
        """Calculate portfolio duration"""
        total_duration = 0
        
        for i, bond in enumerate(self.bonds_data):
            _, modified_duration = BondPricer.bond_duration(
                bond['face_value'], bond['coupon_rate'], 
                bond['yield'], bond['maturity']
            )
            total_duration += self.weights[i] * modified_duration
        
        return total_duration
    
    def portfolio_convexity(self) -> float:
        """Calculate portfolio convexity"""
        total_convexity = 0
        
        for i, bond in enumerate(self.bonds_data):
            convexity = BondPricer.bond_convexity(
                bond['face_value'], bond['coupon_rate'],
                bond['yield'], bond['maturity']
            )
            total_convexity += self.weights[i] * convexity
        
        return total_convexity
    
    def portfolio_yield(self) -> float:
        """Calculate portfolio yield to maturity"""
        total_yield = 0
        
        for i, bond in enumerate(self.bonds_data):
            total_yield += self.weights[i] * bond['yield']
        
        return total_yield
    
    def interest_rate_sensitivity(self, rate_changes: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio value changes for different interest rate scenarios
        
        Parameters:
        -----------
        rate_changes : np.ndarray
            Interest rate changes in basis points
            
        Returns:
        --------
        np.ndarray
            Portfolio value changes
        """
        duration = self.portfolio_duration()
        convexity = self.portfolio_convexity()
        
        value_changes = []
        
        for delta_r in rate_changes:
            delta_r_decimal = delta_r / 10000  # Convert bp to decimal
            
            # Duration and convexity adjustment
            price_change = -duration * delta_r_decimal + 0.5 * convexity * (delta_r_decimal ** 2)
            value_changes.append(price_change)
        
        return np.array(value_changes)

class FixedIncomeVaR:
    """Value at Risk calculations for fixed income portfolios"""
    
    def __init__(self, yield_changes: pd.DataFrame, portfolio_duration: float,
                 portfolio_convexity: float = 0):
        """
        Initialize fixed income VaR calculator
        
        Parameters:
        -----------
        yield_changes : pd.DataFrame
            Historical yield changes
        portfolio_duration : float
            Portfolio duration
        portfolio_convexity : float
            Portfolio convexity
        """
        self.yield_changes = yield_changes
        self.duration = portfolio_duration
        self.convexity = portfolio_convexity
        
    def duration_based_var(self, confidence_levels: List[float] = [0.01, 0.05, 0.10],
                          portfolio_value: float = 1000000) -> Dict[str, float]:
        """
        Calculate VaR using duration approximation
        
        Parameters:
        -----------
        confidence_levels : List[float]
            Confidence levels for VaR
        portfolio_value : float
            Portfolio value
            
        Returns:
        --------
        Dict[str, float]
            VaR estimates
        """
        # Aggregate yield changes (could be weighted by key rate durations)
        if isinstance(self.yield_changes, pd.DataFrame):
            aggregate_yield_changes = self.yield_changes.mean(axis=1)
        else:
            aggregate_yield_changes = self.yield_changes
        
        var_results = {}
        
        for alpha in confidence_levels:
            yield_var = np.percentile(aggregate_yield_changes, alpha * 100)
            
            # Convert yield change to price change using duration (and convexity if available)
            price_change = -self.duration * yield_var
            if self.convexity > 0:
                price_change += 0.5 * self.convexity * (yield_var ** 2)
            
            portfolio_var = price_change * portfolio_value
            
            var_results[f'VaR_{int(alpha*100)}%'] = {
                'yield_change': yield_var,
                'price_change': price_change,
                'portfolio_var': abs(portfolio_var)
            }
        
        return var_results
    
    def key_rate_duration_var(self, key_rate_durations: Dict[str, float],
                             confidence_levels: List[float] = [0.01, 0.05, 0.10],
                             portfolio_value: float = 1000000) -> Dict[str, float]:
        """
        Calculate VaR using key rate durations
        
        Parameters:
        -----------
        key_rate_durations : Dict[str, float]
            Key rate durations for different maturity buckets
        confidence_levels : List[float]
            Confidence levels for VaR
        portfolio_value : float
            Portfolio value
            
        Returns:
        --------
        Dict[str, float]
            VaR estimates using key rate approach
        """
        var_results = {}
        
        # Calculate portfolio returns based on key rate duration exposure
        portfolio_returns = np.zeros(len(self.yield_changes))
        
        for maturity, duration in key_rate_durations.items():
            if maturity in self.yield_changes.columns:
                maturity_yield_changes = self.yield_changes[maturity]
                portfolio_returns += -duration * maturity_yield_changes
        
        for alpha in confidence_levels:
            portfolio_var_return = np.percentile(portfolio_returns, alpha * 100)
            portfolio_var_dollar = portfolio_var_return * portfolio_value
            
            var_results[f'VaR_{int(alpha*100)}%'] = {
                'return_var': portfolio_var_return,
                'dollar_var': abs(portfolio_var_dollar)
            }
        
        return var_results

class FixedIncomeVisualizer:
    """Create visualizations for fixed income analysis"""
    
    def __init__(self, yield_curve_analyzer: YieldCurveAnalyzer = None):
        self.yield_analyzer = yield_curve_analyzer
    
    def plot_yield_curve(self, dates: List[str] = None, 
                        show_interpolated: bool = True) -> go.Figure:
        """
        Plot yield curve evolution
        
        Parameters:
        -----------
        dates : List[str], optional
            Specific dates to plot. If None, shows recent curves
        show_interpolated : bool
            Whether to show interpolated smooth curves
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        if self.yield_analyzer is None:
            return go.Figure()
        
        fig = go.Figure()
        
        if dates is None:
            # Show recent yield curves
            dates = self.yield_analyzer.yield_data.index[-5:].tolist()  # Last 5 curves
        
        colors = px.colors.qualitative.Set1
        
        for i, date in enumerate(dates):
            if date in self.yield_analyzer.yield_data.index:
                yields = self.yield_analyzer.yield_data.loc[date]
                valid_data = yields.dropna()
                
                if len(valid_data) > 0:
                    maturities = self.yield_analyzer.maturities[yields.index.isin(valid_data.index)]
                    
                    # Plot actual points
                    fig.add_trace(go.Scatter(
                        x=maturities,
                        y=valid_data.values,
                        mode='markers',
                        name=f'{date} (actual)',
                        marker=dict(color=colors[i % len(colors)], size=8),
                        showlegend=True
                    ))
                    
                    # Plot interpolated curve if requested
                    if show_interpolated:
                        try:
                            interp_mat, interp_yields = self.yield_analyzer.interpolate_yield_curve(date)
                            fig.add_trace(go.Scatter(
                                x=interp_mat,
                                y=interp_yields,
                                mode='lines',
                                name=f'{date} (interpolated)',
                                line=dict(color=colors[i % len(colors)], width=2),
                                showlegend=False
                            ))
                        except:
                            pass  # Skip if interpolation fails
        
        fig.update_layout(
            title='Yield Curve Evolution',
            xaxis_title='Maturity (Years)',
            yaxis_title='Yield (%)',
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_term_structure_metrics(self) -> go.Figure:
        """
        Plot term structure metrics over time
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        if self.yield_analyzer is None:
            return go.Figure()
        
        metrics_df = self.yield_analyzer.term_structure_analysis()
        
        if metrics_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Yield Curve Level', 'Slope (30Y-2Y)', 
                          'Curvature', '10Y-2Y Steepness'),
            vertical_spacing=0.1
        )
        
        # Level
        fig.add_trace(go.Scatter(
            x=metrics_df.index,
            y=metrics_df['level'],
            mode='lines',
            name='Level',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        # Slope
        fig.add_trace(go.Scatter(
            x=metrics_df.index,
            y=metrics_df['slope'],
            mode='lines',
            name='Slope',
            line=dict(color='red', width=2)
        ), row=1, col=2)
        
        # Curvature
        fig.add_trace(go.Scatter(
            x=metrics_df.index,
            y=metrics_df['curvature'],
            mode='lines',
            name='Curvature',
            line=dict(color='green', width=2)
        ), row=2, col=1)
        
        # Steepness
        fig.add_trace(go.Scatter(
            x=metrics_df.index,
            y=metrics_df['steepness_2_10'],
            mode='lines',
            name='2Y-10Y Steepness',
            line=dict(color='purple', width=2)
        ), row=2, col=2)
        
        fig.update_layout(
            title='Term Structure Metrics Over Time',
            template='plotly_white',
            height=800,
            showlegend=False
        )
        
        fig.update_yaxes(title_text='Yield (%)', row=1, col=1)
        fig.update_yaxes(title_text='Yield Spread (%)', row=1, col=2)
        fig.update_yaxes(title_text='Curvature', row=2, col=1)
        fig.update_yaxes(title_text='Yield Spread (%)', row=2, col=2)
        
        return fig
    
    def plot_duration_convexity_analysis(self, bonds_data: List[Dict], 
                                       yield_range: Tuple[float, float] = (0.01, 0.08)) -> go.Figure:
        """
        Plot duration and convexity analysis for bonds
        
        Parameters:
        -----------
        bonds_data : List[Dict]
            Bond characteristics
        yield_range : Tuple[float, float]
            Range of yields to analyze
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bond Prices vs Yield', 'Duration vs Yield',
                          'Convexity vs Yield', 'Price Sensitivity Comparison'),
            vertical_spacing=0.1
        )
        
        yields = np.linspace(yield_range[0], yield_range[1], 50)
        colors = px.colors.qualitative.Set1
        
        for i, bond in enumerate(bonds_data[:4]):  # Limit to 4 bonds for clarity
            bond_name = f"Bond {i+1} ({bond['coupon_rate']:.1%}, {bond['maturity']:.1f}Y)"
            
            prices = []
            durations = []
            convexities = []
            
            for y in yields:
                price = BondPricer.bond_price(
                    bond['face_value'], bond['coupon_rate'], y, bond['maturity']
                )
                _, duration = BondPricer.bond_duration(
                    bond['face_value'], bond['coupon_rate'], y, bond['maturity']
                )
                convexity = BondPricer.bond_convexity(
                    bond['face_value'], bond['coupon_rate'], y, bond['maturity']
                )
                
                prices.append(price)
                durations.append(duration)
                convexities.append(convexity)
            
            color = colors[i % len(colors)]
            
            # Bond prices
            fig.add_trace(go.Scatter(
                x=yields * 100,
                y=prices,
                mode='lines',
                name=bond_name,
                line=dict(color=color, width=2),
                legendgroup=f'bond{i}',
                showlegend=True
            ), row=1, col=1)
            
            # Duration
            fig.add_trace(go.Scatter(
                x=yields * 100,
                y=durations,
                mode='lines',
                name=bond_name,
                line=dict(color=color, width=2),
                legendgroup=f'bond{i}',
                showlegend=False
            ), row=1, col=2)
            
            # Convexity
            fig.add_trace(go.Scatter(
                x=yields * 100,
                y=convexities,
                mode='lines',
                name=bond_name,
                line=dict(color=color, width=2),
                legendgroup=f'bond{i}',
                showlegend=False
            ), row=2, col=1)
            
            # Price sensitivity (duration vs convexity contribution)
            base_yield = 0.05  # 5% base yield
            yield_changes = np.linspace(-0.02, 0.02, 21)  # -2% to +2%
            
            base_price = BondPricer.bond_price(
                bond['face_value'], bond['coupon_rate'], base_yield, bond['maturity']
            )
            _, base_duration = BondPricer.bond_duration(
                bond['face_value'], bond['coupon_rate'], base_yield, bond['maturity']
            )
            base_convexity = BondPricer.bond_convexity(
                bond['face_value'], bond['coupon_rate'], base_yield, bond['maturity']
            )
            
            duration_approx = []
            actual_prices = []
            
            for dy in yield_changes:
                new_yield = base_yield + dy
                actual_price = BondPricer.bond_price(
                    bond['face_value'], bond['coupon_rate'], new_yield, bond['maturity']
                )
                
                # Duration + convexity approximation
                approx_change = (-base_duration * dy + 0.5 * base_convexity * dy**2)
                approx_price = base_price * (1 + approx_change)
                
                actual_prices.append(actual_price)
                duration_approx.append(approx_price)
            
            # Show actual vs approximated prices
            if i == 0:  # Only show for first bond to avoid clutter
                fig.add_trace(go.Scatter(
                    x=yield_changes * 100,
                    y=actual_prices,
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ), row=2, col=2)
                
                fig.add_trace(go.Scatter(
                    x=yield_changes * 100,
                    y=duration_approx,
                    mode='lines+markers',
                    name='Duration+Convexity Approx',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ), row=2, col=2)
        
        fig.update_layout(
            title='Bond Analytics Dashboard',
            template='plotly_white',
            height=800
        )
        
        fig.update_xaxes(title_text='Yield (%)', row=1, col=1)
        fig.update_xaxes(title_text='Yield (%)', row=1, col=2)
        fig.update_xaxes(title_text='Yield (%)', row=2, col=1)
        fig.update_xaxes(title_text='Yield Change (bp)', row=2, col=2)
        
        fig.update_yaxes(title_text='Bond Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Modified Duration', row=1, col=2)
        fig.update_yaxes(title_text='Convexity', row=2, col=1)
        fig.update_yaxes(title_text='Bond Price ($)', row=2, col=2)
        
        return fig
    
    def plot_interest_rate_scenarios(self, portfolio_analyzer: BondPortfolioAnalyzer,
                                   scenario_range: Tuple[int, int] = (-500, 500)) -> go.Figure:
        """
        Plot portfolio value under different interest rate scenarios
        
        Parameters:
        -----------
        portfolio_analyzer : BondPortfolioAnalyzer
            Portfolio analyzer instance
        scenario_range : Tuple[int, int]
            Range of rate changes in basis points
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        rate_changes = np.linspace(scenario_range[0], scenario_range[1], 100)
        value_changes = portfolio_analyzer.interest_rate_sensitivity(rate_changes)
        
        fig = go.Figure()
        
        # Portfolio value changes
        fig.add_trace(go.Scatter(
            x=rate_changes,
            y=value_changes * 100,
            mode='lines',
            name='Portfolio Value Change',
            line=dict(color='blue', width=3),
            hovertemplate='Rate Change: %{x} bp<br>Value Change: %{y:.2f}%<extra></extra>'
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="No Change", annotation_position="right")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # Highlight common scenarios
        common_scenarios = [-200, -100, 100, 200]  # Common stress scenarios
        for scenario in common_scenarios:
            if scenario_range[0] <= scenario <= scenario_range[1]:
                scenario_idx = np.argmin(np.abs(rate_changes - scenario))
                scenario_value = value_changes[scenario_idx] * 100
                
                fig.add_trace(go.Scatter(
                    x=[scenario],
                    y=[scenario_value],
                    mode='markers',
                    name=f'{scenario} bp scenario',
                    marker=dict(size=10, color='red'),
                    hovertemplate=f'Scenario: {scenario} bp<br>Value Change: {scenario_value:.2f}%<extra></extra>'
                ))
        
        fig.update_layout(
            title='Interest Rate Sensitivity Analysis',
            xaxis_title='Interest Rate Change (basis points)',
            yaxis_title='Portfolio Value Change (%)',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_bond_ladder(ladder_bonds: List[Dict]) -> go.Figure:
        """
        Visualize bond ladder structure
        
        Parameters:
        -----------
        ladder_bonds : List[Dict]
            Bond ladder components with maturity and face value
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        maturities = [bond['maturity'] for bond in ladder_bonds]
        face_values = [bond['face_value'] for bond in ladder_bonds]
        coupon_rates = [bond.get('coupon_rate', 0) for bond in ladder_bonds]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Bond Ladder - Principal Amounts', 'Annual Cash Flows'),
            vertical_spacing=0.15
        )
        
        # Principal amounts by maturity
        fig.add_trace(go.Bar(
            x=maturities,
            y=face_values,
            name='Principal Amount',
            marker_color='lightblue',
            text=[f'${val:,.0f}' for val in face_values],
            textposition='auto'
        ), row=1, col=1)
        
        # Calculate annual cash flows
        years = np.arange(1, int(max(maturities)) + 1)
        annual_cashflows = np.zeros(len(years))
        
        for bond in ladder_bonds:
            maturity = int(bond['maturity'])
            face_value = bond['face_value']
            coupon_rate = bond.get('coupon_rate', 0)
            
            # Add coupon payments for each year until maturity
            for year in range(1, maturity + 1):
                if year <= len(annual_cashflows):
                    annual_cashflows[year - 1] += face_value * coupon_rate
            
            # Add principal repayment at maturity
            if maturity <= len(annual_cashflows):
                annual_cashflows[maturity - 1] += face_value
        
        fig.add_trace(go.Bar(
            x=years,
            y=annual_cashflows,
            name='Total Cash Flow',
            marker_color='green',
            text=[f'${val:,.0f}' for val in annual_cashflows],
            textposition='auto'
        ), row=2, col=1)
        
        fig.update_layout(
            title='Bond Ladder Analysis',
            template='plotly_white',
            height=700,
            showlegend=False
        )
        
        fig.update_xaxes(title_text='Years to Maturity', row=1, col=1)
        fig.update_xaxes(title_text='Year', row=2, col=1)
        fig.update_yaxes(title_text='Principal Amount ($)', row=1, col=1)
        fig.update_yaxes(title_text='Cash Flow ($)', row=2, col=1)
        
        return fig
