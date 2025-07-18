import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataFetcher:
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol, period="1y"):
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return None
            
            return data
        
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @st.cache_data(ttl=300)
    def get_portfolio_data(_self, symbols):
        """Get current data for multiple stocks"""
        try:
            portfolio_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        
                        change = current_price - prev_close
                        change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        portfolio_data[symbol] = {
                            'current_price': current_price,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0)
                        }
                    else:
                        st.warning(f"No recent data available for {symbol}")
                
                except Exception as e:
                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            return portfolio_data
        
        except Exception as e:
            st.error(f"Error fetching portfolio data: {str(e)}")
            return {}
    
    @st.cache_data(ttl=300)
    def get_portfolio_history(_self, symbols, period="1y"):
        """Get historical data for multiple stocks"""
        try:
            if not symbols:
                return pd.DataFrame()
            
            # Ensure symbols is a list
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Download data for all symbols
            data = yf.download(symbols, period=period, progress=False)
            
            if data.empty:
                st.error("No historical data found for the selected symbols")
                return pd.DataFrame()
            
            # Handle single symbol case
            if len(symbols) == 1:
                symbol = symbols[0]
                
                # Check if data has MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    # For single symbol with MultiIndex, columns are like ('Close', 'GOOGL')
                    close_cols = [col for col in data.columns if col[0] == 'Close']
                    if close_cols:
                        close_data = data[close_cols[0]]
                    else:
                        # Try Adj Close
                        adj_close_cols = [col for col in data.columns if col[0] == 'Adj Close']
                        if adj_close_cols:
                            close_data = data[adj_close_cols[0]]
                        else:
                            return pd.DataFrame()
                else:
                    # Single level columns
                    if 'Close' in data.columns:
                        close_data = data['Close']
                    elif 'Adj Close' in data.columns:
                        close_data = data['Adj Close']
                    else:
                        return pd.DataFrame()
                
                # Create DataFrame with proper column name and ensure proper index
                if isinstance(close_data, pd.Series):
                    result_df = pd.DataFrame({symbol: close_data})
                    # Ensure the index is properly named
                    if result_df.index.name is None:
                        result_df.index.name = 'Date'
                    return result_df
                else:
                    return pd.DataFrame()
            
            # Handle multiple symbols case
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    # Extract Close prices for all symbols
                    if 'Close' in data.columns.levels[0]:
                        close_data = data['Close']
                    elif 'Adj Close' in data.columns.levels[0]:
                        close_data = data['Adj Close']
                    else:
                        return pd.DataFrame()
                else:
                    # This shouldn't happen with multiple symbols, but handle it
                    close_data = data
                
                # Handle any missing data and ensure proper index
                if not close_data.empty:
                    close_data = close_data.fillna(method='ffill').fillna(method='bfill')
                    if close_data.index.name is None:
                        close_data.index.name = 'Date'
                    return close_data
                else:
                    return pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def get_market_indices(_self, indices):
        """Get data for market indices"""
        try:
            indices_data = {}
            
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        
                        change = current_price - prev_close
                        change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        indices_data[index] = {
                            'current_price': current_price,
                            'change': change,
                            'change_percent': change_percent
                        }
                
                except Exception as e:
                    st.warning(f"Error fetching data for index {index}: {str(e)}")
                    continue
            
            return indices_data
        
        except Exception as e:
            st.error(f"Error fetching market indices data: {str(e)}")
            return {}
    
    @st.cache_data(ttl=600)
    def get_stock_info(_self, symbol):
        """Get detailed stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'average_volume': info.get('averageVolume', 0),
                'description': info.get('longBusinessSummary', '')
            }
        
        except Exception as e:
            st.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {}
    
    def clear_cache(self):
        """Clear the data cache"""
        try:
            st.cache_data.clear()
            return True
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
            return False
    
    def validate_symbol(self, symbol):
        """Validate if a stock symbol exists"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            return not hist.empty
        except:
            return False
    
    def get_trending_stocks(self, count=10):
        """Get trending stocks (simplified - using popular stocks)"""
        try:
            # Popular stocks list (in a real app, this could come from a trending API)
            popular_symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                'META', 'NVDA', 'NFLX', 'AMD', 'INTC'
            ]
            
            trending_data = self.get_portfolio_data(popular_symbols[:count])
            
            # Sort by volume or price change
            trending_list = []
            for symbol, data in trending_data.items():
                trending_list.append({
                    'symbol': symbol,
                    'price': data['current_price'],
                    'change_percent': data['change_percent'],
                    'volume': data['volume']
                })
            
            # Sort by absolute price change
            trending_list.sort(key=lambda x: abs(x['change_percent']), reverse=True)
            
            return trending_list
        
        except Exception as e:
            st.error(f"Error fetching trending stocks: {str(e)}")
            return []
