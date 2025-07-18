import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def get_risk_level(volatility, beta):
    """Determine risk level based on volatility and beta"""
    try:
        # Define risk thresholds
        if volatility < 15 and abs(beta - 1) < 0.3:
            return "Low"
        elif volatility < 25 and abs(beta - 1) < 0.6:
            return "Medium"
        else:
            return "High"
    except:
        return "Unknown"

def format_currency(amount):
    """Format currency with appropriate symbols"""
    try:
        if abs(amount) >= 1e9:
            return f"${amount/1e9:.1f}B"
        elif abs(amount) >= 1e6:
            return f"${amount/1e6:.1f}M"
        elif abs(amount) >= 1e3:
            return f"${amount/1e3:.1f}K"
        else:
            return f"${amount:.2f}"
    except:
        return "$0.00"

def format_percentage(value):
    """Format percentage with appropriate precision"""
    try:
        return f"{value:.2f}%"
    except:
        return "0.00%"

def calculate_days_between(start_date, end_date):
    """Calculate number of days between two dates"""
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        return (end_date - start_date).days
    except:
        return 0

def validate_stock_symbol(symbol):
    """Basic validation for stock symbols"""
    try:
        if not symbol:
            return False
        
        # Basic checks
        symbol = symbol.upper().strip()
        
        # Check length (most symbols are 1-5 characters)
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # Check for valid characters (letters, numbers, some special characters)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        if not all(c in valid_chars for c in symbol):
            return False
        
        return True
    except:
        return False

def calculate_compound_annual_growth_rate(initial_value, final_value, years):
    """Calculate CAGR"""
    try:
        if initial_value <= 0 or final_value <= 0 or years <= 0:
            return 0
        
        cagr = ((final_value / initial_value) ** (1/years)) - 1
        return cagr * 100
    except:
        return 0

def get_color_for_change(change_percent):
    """Get color based on price change"""
    try:
        if change_percent > 0:
            return "green"
        elif change_percent < 0:
            return "red"
        else:
            return "gray"
    except:
        return "gray"

def calculate_correlation_matrix(price_data):
    """Calculate correlation matrix for multiple stocks"""
    try:
        if price_data.empty:
            return pd.DataFrame()
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    except Exception as e:
        st.error(f"Error calculating correlation matrix: {str(e)}")
        return pd.DataFrame()

def diversification_score(correlation_matrix):
    """Calculate a simple diversification score"""
    try:
        if correlation_matrix.empty:
            return 0
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Calculate mean absolute correlation
        mean_correlation = upper_triangle.abs().mean().mean()
        
        # Convert to diversification score (lower correlation = higher diversification)
        diversification_score = (1 - mean_correlation) * 100
        
        return diversification_score
    except Exception as e:
        st.error(f"Error calculating diversification score: {str(e)}")
        return 0

def get_sector_allocation(portfolio_data):
    """Get sector allocation from portfolio (simplified)"""
    try:
        # This is a simplified version - in a real app, you'd fetch sector data from APIs
        sector_mapping = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology',
            'MSFT': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'META': 'Technology',
            'NVDA': 'Technology',
            'NFLX': 'Communication Services',
            'AMD': 'Technology',
            'INTC': 'Technology',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'WMT': 'Consumer Staples',
            'PG': 'Consumer Staples',
            'JNJ': 'Healthcare',
            'UNH': 'Healthcare',
            'V': 'Financials',
            'MA': 'Financials'
        }
        
        sectors = {}
        for symbol, data in portfolio_data.items():
            sector = sector_mapping.get(symbol, 'Other')
            if sector not in sectors:
                sectors[sector] = 0
            sectors[sector] += data.get('current_value', 0)
        
        return sectors
    except Exception as e:
        st.error(f"Error calculating sector allocation: {str(e)}")
        return {}

def format_large_number(number):
    """Format large numbers with appropriate suffixes"""
    try:
        if abs(number) >= 1e12:
            return f"{number/1e12:.1f}T"
        elif abs(number) >= 1e9:
            return f"{number/1e9:.1f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.1f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.1f}K"
        else:
            return f"{number:.2f}"
    except:
        return "0"

def calculate_moving_average(price_data, window=20):
    """Calculate moving average"""
    try:
        if len(price_data) < window:
            return pd.Series()
        
        return price_data.rolling(window=window).mean()
    except Exception as e:
        st.error(f"Error calculating moving average: {str(e)}")
        return pd.Series()

def get_trading_signals(price_data, short_window=20, long_window=50):
    """Generate simple trading signals based on moving averages"""
    try:
        if len(price_data) < long_window:
            return pd.Series()
        
        short_ma = calculate_moving_average(price_data, short_window)
        long_ma = calculate_moving_average(price_data, long_window)
        
        signals = pd.Series(0, index=price_data.index)
        
        # Buy signal when short MA crosses above long MA
        signals[short_ma > long_ma] = 1
        
        # Sell signal when short MA crosses below long MA
        signals[short_ma < long_ma] = -1
        
        return signals
    except Exception as e:
        st.error(f"Error generating trading signals: {str(e)}")
        return pd.Series()

def validate_date_range(start_date, end_date):
    """Validate date range"""
    try:
        if start_date >= end_date:
            return False
        
        # Check if date range is reasonable (not too far in the past or future)
        today = datetime.now().date()
        
        if start_date > today or end_date > today:
            return False
        
        # Check if date range is not too long (e.g., more than 10 years)
        if (end_date - start_date).days > 3650:  # 10 years
            return False
        
        return True
    except:
        return False
