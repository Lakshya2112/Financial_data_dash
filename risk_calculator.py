import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

class RiskCalculator:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate (10-year Treasury)
    
    def calculate_volatility(self, price_data):
        """Calculate annualized volatility"""
        try:
            if price_data is None or len(price_data) < 2:
                return 0
            
            returns = price_data.pct_change().dropna()
            if returns is None or len(returns) == 0:
                return 0
            
            # Annualize volatility (252 trading days)
            volatility = returns.std() * np.sqrt(252) * 100
            return volatility
        
        except Exception as e:
            st.error(f"Error calculating volatility: {str(e)}")
            return 0
    
    def calculate_beta(self, stock_data, market_symbol="^GSPC"):
        """Calculate beta relative to market (S&P 500)"""
        try:
            if stock_data is None or len(stock_data) < 30:
                return 1.0  # Default beta
            
            # Get market data for the same period
            start_date = stock_data.index[0]
            end_date = stock_data.index[-1]
            
            market_data = yf.download(market_symbol, start=start_date, end=end_date, progress=False)
            
            if market_data is None or market_data.empty:
                return 1.0
            
            # Handle different stock_data formats
            if isinstance(stock_data, pd.DataFrame):
                if len(stock_data.columns) == 1:
                    # Single stock data
                    stock_prices = stock_data.iloc[:, 0]
                else:
                    # Multiple columns, try to find the right one
                    stock_prices = stock_data.iloc[:, 0]  # Use first column as fallback
            else:
                # Series
                stock_prices = stock_data
            
            # Calculate returns
            stock_returns = stock_prices.pct_change().dropna()
            
            # Handle market data structure
            if isinstance(market_data.columns, pd.MultiIndex):
                market_prices = market_data[('Close', market_symbol)] if ('Close', market_symbol) in market_data.columns else market_data.iloc[:, 0]
            else:
                market_prices = market_data['Close'] if 'Close' in market_data.columns else market_data.iloc[:, 0]
            
            market_returns = market_prices.pct_change().dropna()
            
            # Align dates
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) < 30:
                return 1.0
            
            stock_returns = stock_returns[common_dates]
            market_returns = market_returns[common_dates]
            
            # Calculate beta
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
        
        except Exception as e:
            st.error(f"Error calculating beta: {str(e)}")
            return 1.0
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        """Calculate Sharpe ratio"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if returns is None or len(returns) == 0:
                return 0
            
            # Annualize returns
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0
            
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            return sharpe_ratio
        
        except Exception as e:
            st.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
    
    def calculate_max_drawdown(self, price_data):
        """Calculate maximum drawdown"""
        try:
            if price_data is None or len(price_data) < 2:
                return 0
            
            # Calculate running maximum
            running_max = price_data.expanding().max()
            
            # Calculate drawdown
            drawdown = (price_data - running_max) / running_max
            
            # Return maximum drawdown as positive percentage
            max_drawdown = abs(drawdown.min()) * 100
            return max_drawdown
        
        except Exception as e:
            st.error(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def calculate_var(self, historical_data, portfolio, confidence_level=95, portfolio_value=None):
        """Calculate Value at Risk (VaR)"""
        try:
            if historical_data is None or historical_data.empty or portfolio is None or portfolio.empty:
                return 0
            
            # Calculate portfolio returns
            portfolio_returns = self.calculate_portfolio_returns(historical_data, portfolio)
            
            if portfolio_returns is None or len(portfolio_returns) == 0:
                return 0
            
            # Calculate VaR using historical simulation
            var_percentile = 100 - confidence_level
            var_return = np.percentile(portfolio_returns, var_percentile)
            
            # Convert to dollar amount
            if portfolio_value is None:
                portfolio_value = 100000  # Default value
            
            var_amount = abs(var_return * portfolio_value)
            return var_amount
        
        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
            return 0
    
    def calculate_portfolio_returns(self, historical_data, portfolio):
        """Calculate historical portfolio returns"""
        try:
            if historical_data is None or historical_data.empty or portfolio is None or portfolio.empty:
                return pd.Series()
            
            # Get portfolio weights
            from portfolio_manager import PortfolioManager
            from data_fetcher import DataFetcher
            
            portfolio_manager = PortfolioManager()
            data_fetcher = DataFetcher()
            
            current_data = data_fetcher.get_portfolio_data(portfolio['Symbol'].tolist())
            weights = portfolio_manager.get_portfolio_weights(portfolio, current_data)
            
            if not weights:
                return pd.Series()
            
            # Calculate returns for each stock
            stock_returns = historical_data.pct_change().dropna() if historical_data is not None else None
            
            # Calculate weighted portfolio returns
            portfolio_returns = pd.Series(0, index=stock_returns.index) if stock_returns is not None else pd.Series(0, index=[])
            
            if stock_returns is not None and hasattr(stock_returns, 'columns'):
                for symbol, weight in weights.items():
                    if symbol in stock_returns.columns:
                        portfolio_returns += stock_returns[symbol] * weight
                    elif len(stock_returns.columns) == 1:
                        # Handle single stock case where column name might be the stock symbol
                        portfolio_returns += stock_returns.iloc[:, 0] * weight
            
            return portfolio_returns
        
        except Exception as e:
            st.error(f"Error calculating portfolio returns: {str(e)}")
            return pd.Series()
    
    def calculate_portfolio_risk(self, historical_data, portfolio):
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if historical_data is None or historical_data.empty or portfolio is None or portfolio.empty:
                return {
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'beta': 1.0
                }
            
            # Calculate portfolio returns
            portfolio_returns = self.calculate_portfolio_returns(historical_data, portfolio)
            
            if portfolio_returns is None or len(portfolio_returns) == 0:
                return {
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'beta': 1.0
                }
            
            # Calculate portfolio value over time
            portfolio_value = (1 + portfolio_returns).cumprod()
            
            # Calculate metrics
            volatility = self.calculate_volatility(portfolio_value)
            sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
            max_drawdown = self.calculate_max_drawdown(portfolio_value)
            
            # Calculate portfolio beta
            market_data = yf.download("^GSPC", start=historical_data.index[0], end=historical_data.index[-1], progress=False)
            if market_data is not None and not market_data.empty and 'Close' in market_data.columns:
                market_returns = market_data['Close'].pct_change().dropna()
                common_dates = portfolio_returns.index.intersection(market_returns.index)
                if len(common_dates) > 30:
                    portfolio_returns_aligned = portfolio_returns.loc[common_dates]
                    market_returns_aligned = market_returns.loc[common_dates]
                    # Ensure both are 1D arrays of the same length
                    if len(portfolio_returns_aligned) == len(market_returns_aligned):
                        covariance = np.cov(portfolio_returns_aligned.values, market_returns_aligned.values)[0, 1]
                        market_variance = np.var(market_returns_aligned.values)
                        beta = covariance / market_variance if market_variance != 0 else 1.0
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'beta': beta
            }
        
        except Exception as e:
            st.error(f"Error calculating portfolio risk: {str(e)}")
            return {
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'beta': 1.0
            }
    
    def generate_risk_report(self, historical_data, portfolio):
        """Generate a comprehensive risk analysis report"""
        try:
            if historical_data is None or historical_data.empty or portfolio is None or portfolio.empty:
                return pd.DataFrame()
            
            # Calculate portfolio risk metrics
            portfolio_risk = self.calculate_portfolio_risk(historical_data, portfolio)
            
            # Calculate individual stock risk metrics
            stock_risk_data = []
            
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                
                if symbol in historical_data.columns:
                    stock_data = historical_data[symbol].dropna()
                    
                    if len(stock_data) > 30:
                        volatility = self.calculate_volatility(stock_data)
                        beta = self.calculate_beta(stock_data)
                        max_drawdown = self.calculate_max_drawdown(stock_data)
                        
                        returns = stock_data.pct_change().dropna()
                        sharpe_ratio = self.calculate_sharpe_ratio(returns)
                        
                        stock_risk_data.append({
                            'Symbol': symbol,
                            'Volatility (%)': round(volatility, 2),
                            'Beta': round(beta, 2),
                            'Sharpe Ratio': round(sharpe_ratio, 2),
                            'Max Drawdown (%)': round(max_drawdown, 2)
                        })
            
            # Add portfolio summary
            portfolio_summary = {
                'Symbol': 'PORTFOLIO',
                'Volatility (%)': round(portfolio_risk['volatility'], 2),
                'Beta': round(portfolio_risk['beta'], 2),
                'Sharpe Ratio': round(portfolio_risk['sharpe_ratio'], 2),
                'Max Drawdown (%)': round(portfolio_risk['max_drawdown'], 2)
            }
            
            stock_risk_data.append(portfolio_summary)
            
            return pd.DataFrame(stock_risk_data)
        
        except Exception as e:
            st.error(f"Error generating risk report: {str(e)}")
            return pd.DataFrame()
