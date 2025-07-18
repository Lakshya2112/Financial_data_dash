import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time
import json

class AlphaVantageFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or "G7XATNQRO5XGJJT0"
        self.base_url = "https://www.alphavantage.co/query"
        self.call_count = 0
        self.last_call_time = None
        self.rate_limit_delay = 12  # 12 seconds between calls for free tier
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        if self.last_call_time and (current_time - self.last_call_time) < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - (current_time - self.last_call_time)
            time.sleep(sleep_time)
        self.last_call_time = time.time()
        self.call_count += 1
    
    def _make_request(self, params):
        """Make API request with error handling"""
        try:
            self._rate_limit()
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                st.error(f"Alpha Vantage API Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                st.warning(f"Alpha Vantage API Note: {data['Note']}")
                return None
            
            return data
        
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_quote(_self, symbol):
        """Get real-time stock quote"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        quote_data = data.get('Global Quote', {})
        if not quote_data:
            return None
        
        return {
            'symbol': quote_data.get('01. symbol', symbol),
            'price': float(quote_data.get('05. price', 0)),
            'change': float(quote_data.get('09. change', 0)),
            'change_percent': quote_data.get('10. change percent', '0%').replace('%', ''),
            'volume': int(quote_data.get('06. volume', 0)),
            'latest_trading_day': quote_data.get('07. latest trading day', ''),
            'previous_close': float(quote_data.get('08. previous close', 0)),
            'open': float(quote_data.get('02. open', 0)),
            'high': float(quote_data.get('03. high', 0)),
            'low': float(quote_data.get('04. low', 0))
        }
    
    @st.cache_data(ttl=3600)
    def get_daily_prices(_self, symbol, outputsize='compact'):
        """Get daily stock prices"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return None
        
        # Convert to DataFrame
        df_data = []
        for date, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    @st.cache_data(ttl=3600)
    def get_company_overview(_self, symbol):
        """Get company fundamental data"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        # Convert string numbers to floats where applicable
        numeric_fields = [
            'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 
            'BookValue', 'DividendPerShare', 'DividendYield', 'EPS',
            'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM',
            'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM',
            'GrossProfitTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY',
            'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE',
            'ForwardPE', 'PriceToSalesRatioTTM', 'PriceToBookRatio',
            'EVToRevenue', 'EVToEBITDA', 'Beta', '52WeekHigh', '52WeekLow',
            '50DayMovingAverage', '200DayMovingAverage', 'SharesOutstanding'
        ]
        
        for field in numeric_fields:
            if field in data and data[field] != 'None' and data[field] != '-':
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError):
                    data[field] = 0
            else:
                data[field] = 0
        
        return data
    
    @st.cache_data(ttl=3600)
    def get_technical_indicators(_self, symbol, indicator='SMA', interval='daily', time_period=20):
        """Get technical indicators"""
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': 'close',
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        # The key varies by indicator
        possible_keys = [
            f'Technical Analysis: {indicator}',
            f'Technical Analysis: {indicator.upper()}',
            f'Technical Analysis: {indicator.lower()}'
        ]
        
        indicator_data = None
        for key in possible_keys:
            if key in data:
                indicator_data = data[key]
                break
        
        if not indicator_data:
            return None
        
        # Convert to DataFrame
        df_data = []
        for date, values in indicator_data.items():
            df_data.append({
                'Date': pd.to_datetime(date),
                indicator: float(list(values.values())[0])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    @st.cache_data(ttl=3600)
    def get_earnings_data(_self, symbol):
        """Get earnings data"""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        # Extract annual and quarterly earnings
        annual_earnings = data.get('annualEarnings', [])
        quarterly_earnings = data.get('quarterlyEarnings', [])
        
        result = {
            'annual': pd.DataFrame(annual_earnings),
            'quarterly': pd.DataFrame(quarterly_earnings)
        }
        
        # Convert fiscal dates to datetime
        if not result['annual'].empty:
            result['annual']['fiscalDateEnding'] = pd.to_datetime(result['annual']['fiscalDateEnding'])
        
        if not result['quarterly'].empty:
            result['quarterly']['fiscalDateEnding'] = pd.to_datetime(result['quarterly']['fiscalDateEnding'])
        
        return result
    
    @st.cache_data(ttl=3600)
    def get_sector_performance(_self):
        """Get sector performance data"""
        params = {
            'function': 'SECTOR',
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        # Extract different time periods
        sectors = {}
        time_periods = [
            'Rank A: Real-Time Performance',
            'Rank B: 1 Day Performance',
            'Rank C: 5 Day Performance',
            'Rank D: 1 Month Performance',
            'Rank E: 3 Month Performance',
            'Rank F: Year-to-Date (YTD) Performance',
            'Rank G: 1 Year Performance'
        ]
        
        for period in time_periods:
            if period in data:
                sectors[period] = data[period]
        
        return sectors
    
    @st.cache_data(ttl=3600)
    def get_forex_rates(_self, from_currency='USD', to_currency='EUR'):
        """Get foreign exchange rates"""
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        time_series = data.get(f'Time Series FX (Daily)', {})
        if not time_series:
            return None
        
        # Convert to DataFrame
        df_data = []
        for date, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    @st.cache_data(ttl=3600)
    def get_market_news(_self, topics='technology,earnings', sort='LATEST', limit=50):
        """Get market news and sentiment"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': topics,
            'sort': sort,
            'limit': limit,
            'apikey': _self.api_key
        }
        
        data = _self._make_request(params)
        if not data:
            return None
        
        news_feed = data.get('feed', [])
        
        # Process news data
        processed_news = []
        for article in news_feed:
            processed_news.append({
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'published': article.get('time_published', ''),
                'sentiment_score': float(article.get('overall_sentiment_score', 0)),
                'sentiment_label': article.get('overall_sentiment_label', 'Neutral'),
                'topics': [topic.get('topic', '') for topic in article.get('topics', [])]
            })
        
        return processed_news
    
    def get_multiple_quotes(self, symbols):
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quote = self.get_stock_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes
    
    def get_portfolio_fundamentals(self, symbols):
        """Get fundamental data for portfolio stocks"""
        fundamentals = {}
        for symbol in symbols:
            overview = self.get_company_overview(symbol)
            if overview:
                fundamentals[symbol] = overview
        return fundamentals
    
    def get_api_usage_stats(self):
        """Get API usage statistics"""
        return {
            'calls_made': self.call_count,
            'last_call_time': self.last_call_time,
            'rate_limit_delay': self.rate_limit_delay
        }