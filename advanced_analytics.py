import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_alpha_beta(self, stock_returns, market_returns):
        """Calculate alpha and beta using linear regression"""
        try:
            if len(stock_returns) < 30 or len(market_returns) < 30:
                return 0, 1.0
            
            # Align the data
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) < 30:
                return 0, 1.0
            
            stock_aligned = stock_returns[common_dates]
            market_aligned = market_returns[common_dates]
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(market_aligned, stock_aligned)
            
            # Beta is the slope
            beta = slope
            
            # Alpha is the intercept annualized
            alpha = intercept * 252 * 100  # Convert to annual percentage
            
            return alpha, beta
        
        except Exception as e:
            st.error(f"Error calculating alpha/beta: {str(e)}")
            return 0, 1.0
    
    def calculate_information_ratio(self, stock_returns, benchmark_returns):
        """Calculate Information Ratio"""
        try:
            if len(stock_returns) < 30 or len(benchmark_returns) < 30:
                return 0
            
            # Align the data
            common_dates = stock_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 30:
                return 0
            
            stock_aligned = stock_returns[common_dates]
            benchmark_aligned = benchmark_returns[common_dates]
            
            # Calculate excess returns
            excess_returns = stock_aligned - benchmark_aligned
            
            # Calculate Information Ratio
            if excess_returns.std() == 0:
                return 0
            
            ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return ir
        
        except Exception as e:
            st.error(f"Error calculating Information Ratio: {str(e)}")
            return 0
    
    def calculate_calmar_ratio(self, returns, max_drawdown):
        """Calculate Calmar Ratio"""
        try:
            if len(returns) == 0 or max_drawdown == 0:
                return 0
            
            annual_return = returns.mean() * 252
            calmar_ratio = annual_return / (max_drawdown / 100)
            return calmar_ratio
        
        except Exception as e:
            st.error(f"Error calculating Calmar Ratio: {str(e)}")
            return 0
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=None):
        """Calculate Sortino Ratio"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if len(returns) == 0:
                return 0
            
            # Calculate excess returns
            excess_returns = returns - risk_free_rate / 252
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return 0
            
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
            
            if downside_deviation == 0:
                return 0
            
            # Calculate Sortino ratio
            annual_excess_return = excess_returns.mean() * 252
            sortino_ratio = annual_excess_return / downside_deviation
            
            return sortino_ratio
        
        except Exception as e:
            st.error(f"Error calculating Sortino Ratio: {str(e)}")
            return 0
    
    def calculate_treynor_ratio(self, returns, beta, risk_free_rate=None):
        """Calculate Treynor Ratio"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if len(returns) == 0 or beta == 0:
                return 0
            
            annual_return = returns.mean() * 252
            treynor_ratio = (annual_return - risk_free_rate) / beta
            return treynor_ratio
        
        except Exception as e:
            st.error(f"Error calculating Treynor Ratio: {str(e)}")
            return 0
    
    def calculate_portfolio_performance_metrics(self, portfolio, historical_data):
        """Calculate comprehensive portfolio performance metrics"""
        try:
            if historical_data.empty or portfolio.empty:
                return {}
            
            # Get portfolio weights (simplified - equal weights for now)
            symbols = portfolio['Symbol'].tolist()
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
            # Calculate weighted portfolio returns
            portfolio_returns = pd.Series(0, index=historical_data.index)
            
            for symbol, weight in weights.items():
                if symbol in historical_data.columns:
                    stock_returns = historical_data[symbol].pct_change().dropna()
                    portfolio_returns += stock_returns * weight
            
            portfolio_returns = portfolio_returns.dropna()
            
            if len(portfolio_returns) == 0:
                return {}
            
            # Get market data for comparisons
            market_data = yf.download("^GSPC", start=historical_data.index[0], end=historical_data.index[-1], progress=False)
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Calculate metrics
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = (1 + portfolio_returns).prod() - 1
            metrics['annual_return'] = portfolio_returns.mean() * 252
            metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / metrics['volatility'] if metrics['volatility'] != 0 else 0
            
            # Calculate alpha and beta
            alpha, beta = self.calculate_alpha_beta(portfolio_returns, market_returns)
            metrics['alpha'] = alpha
            metrics['beta'] = beta
            
            # More advanced metrics
            metrics['information_ratio'] = self.calculate_information_ratio(portfolio_returns, market_returns)
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(portfolio_returns)
            metrics['treynor_ratio'] = self.calculate_treynor_ratio(portfolio_returns, beta)
            
            # Calculate max drawdown
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            running_max = portfolio_cumulative.expanding().max()
            drawdown = (portfolio_cumulative - running_max) / running_max
            metrics['max_drawdown'] = abs(drawdown.min())
            
            # Calmar ratio
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(portfolio_returns, metrics['max_drawdown'] * 100)
            
            # VaR calculations
            metrics['var_95'] = np.percentile(portfolio_returns, 5)
            metrics['var_99'] = np.percentile(portfolio_returns, 1)
            
            # CVaR (Conditional VaR)
            var_95 = metrics['var_95']
            metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_95].mean()
            
            return metrics
        
        except Exception as e:
            st.error(f"Error calculating portfolio performance metrics: {str(e)}")
            return {}
    
    def generate_performance_attribution(self, portfolio, historical_data):
        """Generate performance attribution analysis"""
        try:
            if historical_data.empty or portfolio.empty:
                return pd.DataFrame()
            
            attribution_data = []
            
            # Calculate individual stock contributions
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                weight = 1 / len(portfolio)  # Equal weight for simplicity
                
                if symbol in historical_data.columns:
                    stock_returns = historical_data[symbol].pct_change().dropna()
                    
                    if len(stock_returns) > 0:
                        # Calculate contribution to portfolio return
                        contribution = stock_returns.mean() * weight * 252 * 100
                        
                        # Calculate individual stock metrics
                        stock_volatility = stock_returns.std() * np.sqrt(252) * 100
                        stock_sharpe = (stock_returns.mean() * 252 - self.risk_free_rate) / (stock_returns.std() * np.sqrt(252)) if stock_returns.std() != 0 else 0
                        
                        attribution_data.append({
                            'Symbol': symbol,
                            'Weight (%)': weight * 100,
                            'Annual Return (%)': stock_returns.mean() * 252 * 100,
                            'Volatility (%)': stock_volatility,
                            'Sharpe Ratio': stock_sharpe,
                            'Contribution to Return (%)': contribution
                        })
            
            return pd.DataFrame(attribution_data)
        
        except Exception as e:
            st.error(f"Error generating performance attribution: {str(e)}")
            return pd.DataFrame()
    
    def create_risk_return_chart(self, portfolio, historical_data):
        """Create risk-return scatter plot"""
        try:
            if historical_data.empty or portfolio.empty:
                return None
            
            returns_data = []
            volatility_data = []
            symbols_data = []
            
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                
                if symbol in historical_data.columns:
                    stock_returns = historical_data[symbol].pct_change().dropna()
                    
                    if len(stock_returns) > 30:
                        annual_return = stock_returns.mean() * 252 * 100
                        volatility = stock_returns.std() * np.sqrt(252) * 100
                        
                        returns_data.append(annual_return)
                        volatility_data.append(volatility)
                        symbols_data.append(symbol)
            
            if not returns_data:
                return None
            
            fig = px.scatter(
                x=volatility_data,
                y=returns_data,
                text=symbols_data,
                title="Risk-Return Analysis",
                labels={'x': 'Volatility (%)', 'y': 'Annual Return (%)'}
            )
            
            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_title="Risk (Volatility %)",
                yaxis_title="Return (Annual %)",
                showlegend=False
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating risk-return chart: {str(e)}")
            return None
    
    def create_correlation_heatmap(self, historical_data):
        """Create correlation heatmap"""
        try:
            if historical_data.empty:
                return None
            
            # Calculate correlation matrix
            returns = historical_data.pct_change().dropna()
            correlation_matrix = returns.corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Portfolio Correlation Matrix",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def calculate_portfolio_optimization_suggestions(self, portfolio, historical_data):
        """Generate portfolio optimization suggestions"""
        try:
            if historical_data.empty or portfolio.empty:
                return []
            
            suggestions = []
            
            # Calculate correlation matrix
            returns = historical_data.pct_change().dropna()
            correlation_matrix = returns.corr()
            
            # Check for high correlations
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                suggestions.append({
                    'type': 'Diversification',
                    'message': f"High correlation detected between {len(high_corr_pairs)} stock pairs. Consider diversifying across different sectors.",
                    'severity': 'Medium'
                })
            
            # Check portfolio concentration
            if len(portfolio) < 5:
                suggestions.append({
                    'type': 'Concentration',
                    'message': "Portfolio has fewer than 5 stocks. Consider adding more stocks to reduce concentration risk.",
                    'severity': 'High'
                })
            
            # Check for extremely volatile stocks
            for symbol in portfolio['Symbol']:
                if symbol in historical_data.columns:
                    stock_returns = historical_data[symbol].pct_change().dropna()
                    volatility = stock_returns.std() * np.sqrt(252) * 100
                    
                    if volatility > 50:
                        suggestions.append({
                            'type': 'Volatility',
                            'message': f"{symbol} has high volatility ({volatility:.1f}%). Consider reducing position size or adding hedging.",
                            'severity': 'Medium'
                        })
            
            return suggestions
        
        except Exception as e:
            st.error(f"Error generating optimization suggestions: {str(e)}")
            return []