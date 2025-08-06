# Portfolio Analytics Hub 📊

A comprehensive Streamlit application for modern portfolio theory analysis and risk management. This application provides tools for portfolio optimization, Value-at-Risk (VaR) analysis, multi-asset analysis, and educational resources on quantitative finance.

## 🚀 Features

### 📈 Portfolio Theory Module
- **Modern Portfolio Theory Implementation**: Complete Markowitz optimization with efficient frontier analysis
- **Risk-Return Optimization**: Minimum variance and maximum Sharpe ratio portfolios
- **Capital Market Line**: CML visualization with user-selectable market portfolio
- **Interactive Visualizations**: Dynamic charts with Plotly for portfolio analysis
- **Risk Metrics**: Comprehensive risk assessment including VaR, CVaR, and drawdown analysis

### ⚠️ Value-at-Risk Analytics
- **Multiple VaR Methods**: 
  - Parametric VaR (variance-covariance method)
  - Historical VaR (simulation)
  - Monte Carlo VaR with customizable parameters
- **Multi-Asset Support**: Equities, fixed income, and commodities
- **Stress Testing**: Predefined market scenarios and custom scenario builder
- **Component VaR**: Risk attribution and incremental VaR analysis
- **Model Validation**: Kupiec backtesting for VaR model validation

### 🌐 Multi-Asset Analysis
- **Cross-Asset Correlation**: Comprehensive correlation analysis across asset classes
- **Asset Class Comparison**: Performance and risk metrics comparison
- **Strategic Asset Allocation**: Interactive portfolio construction tools
- **Fixed Income Analytics**: Yield curve analysis and duration/convexity calculations

### 🎓 Educational Hub
- **Theory Explanations**: In-depth coverage of MPT, VaR, and fixed income theory
- **Mathematical Foundations**: Formulas, assumptions, and statistical concepts
- **Interactive Examples**: Live calculations and visualizations
- **References and Sources**: Academic papers and implementation details

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or Download** the application files to your local directory

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
portfolio-analytics-hub/
├── streamlit_app.py              # Main application entry point
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── src/                          # Source modules
│   ├── data_loader.py           # Data fetching and processing
│   ├── portfolio.py             # Portfolio optimization and analysis
│   ├── var_models.py            # VaR calculation methods
│   └── fixed_income.py          # Bond analytics and yield curve analysis
└── pages/                       # Streamlit pages
    ├── 1_Portfolio_Theory.py    # Portfolio optimization interface
    ├── 2_VaR_Analytics.py       # VaR analysis interface
    ├── 3_Multi_Asset_Analysis.py # Multi-asset analysis interface
    └── 4_Educational.py         # Educational content and theory
```

## 📊 Data Sources

### Automated Data Fetching
- **Yahoo Finance**: Stock prices, ETF data, commodity futures (via `yfinance`)
- **FRED**: Treasury yields, economic indicators (via `pandas-datareader`)
- **No API Keys Required**: All data sources are freely accessible

### Sample Portfolios
- Technology Portfolio (AAPL, GOOGL, MSFT, NVDA, AMZN)
- Dividend Portfolio (JNJ, PG, KO, PFE, VZ)
- Growth Portfolio (TSLA, NFLX, AMD, SHOP, SQ)
- Value Portfolio (BRK-B, JPM, WMT, HD, UNH)
- Balanced Portfolio (SPY, QQQ, IWM, EFA, EEM)
- Sector ETFs (XLF, XLK, XLE, XLV, XLI)

## 🔧 Key Features

### Portfolio Optimization
- **Efficient Frontier Generation**: 50+ optimal portfolios across risk-return spectrum
- **Optimal Portfolio Identification**: Minimum variance and maximum Sharpe ratio portfolios
- **Risk-Free Rate Configuration**: User-selectable or auto-fetched from FRED
- **Market Portfolio Options**: Maximum Sharpe portfolio or S&P 500 as market proxy
- **Interactive Analysis**: Real-time portfolio weight adjustments and analysis

### Risk Management
- **Comprehensive VaR Analysis**: Multiple methodologies with confidence intervals
- **Stress Testing**: Historical market scenarios (2008 crisis, COVID-19, etc.)
- **Component Risk Analysis**: Individual asset risk contributions
- **Model Backtesting**: Statistical validation of VaR models
- **Multi-Asset Risk Assessment**: Cross-asset class risk analysis

### Educational Content
- **Theory and Practice**: Detailed explanations of MPT, VaR, and fixed income analytics
- **Mathematical Foundations**: Formulas, derivations, and statistical concepts
- **Interactive Examples**: Live calculations and parameter adjustments
- **Implementation Details**: Code explanations and methodology assumptions

## 🎯 Usage Examples

### 1. Basic Portfolio Optimization
```python
# Using the Portfolio Theory page:
# 1. Select "Tech Portfolio" from sample portfolios
# 2. Choose 2-year analysis period
# 3. Set risk-free rate (auto or manual)
# 4. Click "Run Optimization"
# 5. Analyze efficient frontier and optimal portfolios
```

### 2. VaR Analysis
```python
# Using the VaR Analytics page:
# 1. Select asset class (Equities/Fixed Income/Commodities)
# 2. Configure portfolio ($1M default)
# 3. Set confidence levels (1%, 5%, 10%)
# 4. Click "Calculate VaR"
# 5. Compare parametric, historical, and Monte Carlo methods
```

### 3. Multi-Asset Portfolio
```python
# Using the Multi-Asset Analysis page:
# 1. Select multiple asset classes
# 2. Configure allocations (50% equities, 30% bonds, 20% commodities)
# 3. Analyze correlation matrix
# 4. Review portfolio performance and risk metrics
```

## ⚠️ Important Disclaimers

### Model Limitations
- **Historical Data Dependency**: All models rely on past data which may not predict future performance
- **Normal Distribution Assumptions**: Many calculations assume normal return distributions
- **Constant Parameters**: Models assume correlations and volatilities remain stable
- **No Transaction Costs**: Real trading involves costs not reflected in calculations

### Investment Warnings
- **Educational Purpose**: This tool is for educational and research purposes only
- **Not Investment Advice**: Do not use as the sole basis for investment decisions
- **Professional Consultation**: Consult qualified financial advisors for investment planning
- **Risk Acknowledgment**: All investments carry risk of loss

## 🔬 Technical Details

### Optimization Methods
- **Solver**: SciPy's Sequential Least Squares Programming (SLSQP)
- **Constraints**: Long-only portfolios with weights summing to 100%
- **Convergence**: Maximum 1000 iterations with tolerance of 1e-8

### Statistical Methods
- **Return Calculations**: Simple returns (price percentage changes)
- **Annualization**: 252 trading days per year
- **Risk-Free Rate**: 3-month Treasury rate as default
- **Correlation**: Pearson correlation coefficient

### VaR Implementation
- **Parametric**: Normal distribution assumption with mean and standard deviation
- **Historical**: Empirical quantiles from historical return distribution
- **Monte Carlo**: 10,000 simulations with normal distribution assumption
- **Backtesting**: Kupiec likelihood ratio test for model validation

## 🤝 Contributing

We welcome contributions to improve the application:

1. **Bug Reports**: Submit issues via GitHub or email
2. **Feature Requests**: Suggest new functionality or improvements
3. **Code Contributions**: Follow coding standards and include documentation
4. **Educational Content**: Help improve explanations and examples

## 📚 References

### Academic Sources
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
- Sharpe, W. F. (1964). Capital Asset Prices. *The Journal of Finance*, 19(3), 425-442.
- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.

### Data Sources
- Yahoo Finance: https://finance.yahoo.com
- FRED Economic Data: https://fred.stlouisfed.org
- CFA Institute: https://www.cfainstitute.org

## 📞 Support

For questions, issues, or feedback:

- **Documentation**: Refer to the Educational Hub within the application
- **Issues**: Check the troubleshooting section below
- **Updates**: Monitor the GitHub repository for updates

## 🔧 Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Check internet connection for Yahoo Finance/FRED access
   - Verify ticker symbols are valid and actively traded
   - Some international tickers may require exchange suffixes

2. **Optimization Failures**:
   - Ensure at least 2 assets are selected
   - Check for sufficient historical data (minimum 100 observations)
   - Verify assets have overlapping date ranges

3. **Performance Issues**:
   - Large portfolios (>10 assets) may take longer to optimize
   - Monte Carlo simulations with high iteration counts may be slow
   - Consider reducing the analysis period for faster processing

4. **Visualization Problems**:
   - Ensure Plotly is properly installed
   - Clear browser cache if charts don't display
   - Try refreshing the page or restarting the application

### System Requirements
- **Memory**: Minimum 4GB RAM (8GB recommended for large portfolios)
- **Browser**: Modern browser with JavaScript enabled
- **Python**: Version 3.8+ with all required packages installed
- **Internet**: Required for real-time data fetching

## 📈 Future Enhancements

Planned features for future versions:
- **Factor Models**: Fama-French multi-factor risk models
- **Options Analytics**: Greeks calculation and options portfolio analysis
- **ESG Integration**: Environmental, social, and governance metrics
- **Alternative Assets**: Real estate and cryptocurrency analysis
- **Advanced Optimization**: Black-Litterman and robust optimization methods

---

**Disclaimer**: This application is for educational and research purposes only. It is not intended as investment advice. All investments carry risk, and past performance does not guarantee future results. Please consult with qualified financial advisors before making investment decisions.

**License**: MIT License - feel free to use and modify for educational purposes.

**Version**: 1.0.0 | **Last Updated**: 2025

---

*Built with ❤️ using Streamlit, NumPy, Pandas, SciPy, and Plotly*
