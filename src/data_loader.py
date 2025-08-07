import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import List, Dict, Optional, Tuple

class DataLoader:
    """Simplified data loader using only Yahoo Finance"""
    
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
            # Clean tickers
            clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
            
            if not clean_tickers:
                return pd.DataFrame()
            
            data = yf.download(clean_tickers, period=period, progress=False)
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle single vs multiple tickers
            if len(clean_tickers) == 1:
                if 'Adj Close' in data.columns:
                    return data[['Adj Close']].rename(columns={'Adj Close': clean_tickers[0]})
                else:
                    return pd.DataFrame()
            else:
                if 'Adj Close' in data.columns:
                    return data['Adj Close']
                else:
                    return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    def get_treasury_rates(self) -> pd.DataFrame:
        """
        Get Treasury rates using ETF proxies
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with Treasury ETF prices as proxy for rates
        """
        try:
            # Use Treasury ETFs as proxies
            treasury_etfs = {
                '3M': 'BIL',   # SPDR Bloomberg 1-3 Month T-Bill ETF
                '1-3Y': 'SHY', # iShares 1-3 Year Treasury Bond ETF
                '3-7Y': 'IEI', # iShares 3-7 Year Treasury Bond ETF
                '7-10Y': 'IEF', # iShares 7-10 Year Treasury Bond ETF
                '10-20Y': 'TLH', # iShares 10-20 Year Treasury Bond ETF
                '20+Y': 'TLT'  # iShares 20+ Year Treasury Bond ETF
            }
            
            etf_data = self.get_stock_data(list(treasury_etfs.values()), "2y")
            
            if not etf_data.empty:
                # Rename columns to maturity labels
                etf_data.columns = [list(treasury_etfs.keys())[list(treasury_etfs.values()).index(col)] 
                                   for col in etf_data.columns]
                return etf_data
            else:
                return self._generate_sample_treasury_data()
                
        except Exception as e:
            st.warning(f"Could not fetch Treasury ETF data: {str(e)}")
            return self._generate_sample_treasury_data()
    
    def _generate_sample_treasury_data(self) -> pd.DataFrame:
        """Generate sample Treasury rate data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Realistic Treasury rate levels (as of 2024/2025)
        base_rates = {
            '3M': 4.8,
            '1-3Y': 4.5,
            '3-7Y': 4.2,
            '7-10Y': 4.1,
            '10-20Y': 4.3,
            '20+Y': 4.4
        }
        
        np.random.seed(42)  # For reproducibility
        
        data = {}
        for maturity, base_rate in base_rates.items():
            rates = []
            current_rate = base_rate
            
            for _
