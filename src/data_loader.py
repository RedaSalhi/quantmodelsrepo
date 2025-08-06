import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import List, Dict, Optional, Tuple

class DataLoader:
    """Handles data loading from Yahoo Finance and FRED"""
    
    def __init__(self):
        self.cache_duration = 3600  # 1 hour cache
    
    @st.cache_data(ttl=3600)
    def get_stock_data(_self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
        period : str
            Time period ('1y', '2y', '5y', 'max')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with adjusted close prices
        """
        try:
            data = yf.download(tickers, period=period, progress=False)
            
            if len(tickers) == 1:
                return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
            else:
                return data['Adj Close']
                
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def get_fred_data(_self, series_id: str, start_date: str = None) -> pd.Series:
        """
        Fetch data from FRED
        
        Parameters:
        -----------
        series_id : str
            FRED series ID (e.g., 'DGS10' for 10-year Treasury)
        start_date : str
            Start date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.Series
            Time series data
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            data = pdr.get_data_fred(series_id, start=start_date)
            return data[series_id]
            
        except Exception as e:
            st.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return pd.Series()
    
    @st.cache_data(ttl=3600)
    def get_treasury_rates(_self) -> pd.DataFrame:
        """
        Fetch various Treasury rates from FRED
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with different maturity Treasury rates
        """
        treasury_series = {
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        treasury_data = {}
        
        for name, series_id in treasury_series.items():
            try:
                data = self.get_fred_data(series_id)
                if not data.empty:
                    treasury_data[name] = data
            except:
                continue
        
        if treasury_data:
            return pd.DataFrame(treasury_data)
        else:
            st.warning("Could not fetch Treasury rate data")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def get_commodity_data(_self, period: str = "2y") -> pd.DataFrame:
        """
        Fetch commodity data from Yahoo Finance
        
        Parameters:
        -----------
        period : str
            Time period for data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with commodity prices
        """
        commodity_tickers = {
            'Gold': 'GC=F',
            'Crude Oil': 'CL=F',
            'Natural Gas': 'NG=F',
            'Silver': 'SI=F',
            'Copper': 'HG=F'
        }
        
        commodity_data = {}
        
        for name, ticker in commodity_tickers.items():
            try:
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty:
                    commodity_data[name] = data['Adj Close']
            except:
                continue
        
        if commodity_data:
            return pd.DataFrame(commodity_data)
        else:
            st.warning("Could not fetch commodity data")
            return pd.DataFrame()
    
    def get_market_data(self, asset_type: str, tickers: List[str] = None, 
                       period: str = "2y") -> pd.DataFrame:
        """
        Unified method to get market data based on asset type
        
        Parameters:
        -----------
        asset_type : str
            Type of asset ('stocks', 'bonds', 'commodities')
        tickers : List[str], optional
            List of tickers (for stocks)
        period : str
            Time period for data
            
        Returns:
        --------
        pd.DataFrame
            Market data
        """
        if asset_type.lower() == 'stocks' and tickers:
            return self.get_stock_data(tickers, period)
        elif asset_type.lower() == 'bonds':
            return self.get_treasury_rates()
        elif asset_type.lower() == 'commodities':
            return self.get_commodity_data(period)
        else:
            st.error(f"Unknown asset type: {asset_type}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.DataFrame, 
                         method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        method : str
            'simple' or 'log' returns
            
        Returns:
        --------
        pd.DataFrame
            Returns data
        """
        if method.lower() == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def get_risk_free_rate(self, rate_type: str = 'DGS3MO') -> float:
        """
        Get current risk-free rate
        
        Parameters:
        -----------
        rate_type : str
            FRED series ID for risk-free rate
            
        Returns:
        --------
        float
            Current risk-free rate (annualized)
        """
        try:
            rate_data = self.get_fred_data(rate_type)
            if not rate_data.empty:
                latest_rate = rate_data.dropna().iloc[-1] / 100  # Convert to decimal
                return latest_rate
            else:
                return 0.02  # Default 2% if data unavailable
        except:
            return 0.02
    
    @staticmethod
    def get_sample_portfolios() -> Dict[str, List[str]]:
        """
        Get predefined sample portfolios
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary of portfolio names and tickers
        """
        return {
            "Tech Portfolio": ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"],
            "Dividend Portfolio": ["JNJ", "PG", "KO", "PFE", "VZ"],
            "Growth Portfolio": ["TSLA", "NFLX", "AMD", "SHOP", "SQ"],
            "Value Portfolio": ["BRK-B", "JPM", "WMT", "HD", "UNH"],
            "Balanced Portfolio": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
            "Sector ETFs": ["XLF", "XLK", "XLE", "XLV", "XLI"]
        }

class MarketDataProcessor:
    """Process and analyze market data"""
    
    @staticmethod
    def clean_data(data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Clean market data by handling missing values"""
        if method == 'forward_fill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate()
        else:
            return data.dropna()
    
    @staticmethod
    def calculate_statistics(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic statistics for returns"""
        stats = pd.DataFrame({
            'Mean': returns.mean() * 252,  # Annualized
            'Volatility': returns.std() * np.sqrt(252),  # Annualized
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'Min': returns.min(),
            'Max': returns.max()
        })
        return stats
    
    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        return returns.corr()
    
    @staticmethod
    def detect_outliers(returns: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect outliers in returns data"""
        if method == 'iqr':
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (returns < lower_bound) | (returns > upper_bound)
        else:
            # Z-score method
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            return z_scores > 3
