import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'portfolio.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT,
            quantity REAL,
            purchase_price REAL,
            date_added TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_stock(symbol, name, quantity, purchase_price, date_added):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO stocks (symbol, name, quantity, purchase_price, date_added)
        VALUES (?, ?, ?, ?, ?)
    ''', (symbol, name, quantity, purchase_price, date_added))
    conn.commit()
    conn.close()

def get_all_stocks():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, symbol, name, quantity, purchase_price, date_added FROM stocks')
    rows = c.fetchall()
    conn.close()
    return rows

def remove_stock(stock_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM stocks WHERE id = ?', (stock_id,))
    conn.commit()
    conn.close()

def sync_db_to_session():
    import pandas as pd
    rows = get_all_stocks()
    df = pd.DataFrame(rows, columns=["ID", "Symbol", "Name", "Shares", "Purchase_Price", "Date_Added"])
    # Only keep relevant columns for session state
    if not df.empty:
        st.session_state.portfolio = df[["Symbol", "Shares", "Purchase_Price", "Date_Added"]].copy()
    else:
        st.session_state.portfolio = pd.DataFrame(columns=["Symbol", "Shares", "Purchase_Price", "Date_Added"])

class PortfolioManager:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase_Price', 'Date_Added'])
    
    def add_stock(self, symbol, shares, purchase_price):
        """Add a stock to the portfolio"""
        try:
            # Check if stock already exists
            existing_stock = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]
            
            if not existing_stock.empty:
                # Update existing stock (add to position)
                current_shares = existing_stock['Shares'].iloc[0]
                current_price = existing_stock['Purchase_Price'].iloc[0]
                
                # Calculate weighted average price
                total_shares = current_shares + shares
                weighted_price = (current_shares * current_price + shares * purchase_price) / total_shares
                
                # Update the record
                st.session_state.portfolio.loc[
                    st.session_state.portfolio['Symbol'] == symbol, 'Shares'
                ] = total_shares
                st.session_state.portfolio.loc[
                    st.session_state.portfolio['Symbol'] == symbol, 'Purchase_Price'
                ] = weighted_price
            else:
                # Add new stock
                new_stock = pd.DataFrame({
                    'Symbol': [symbol],
                    'Shares': [shares],
                    'Purchase_Price': [purchase_price],
                    'Date_Added': [datetime.now()]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_stock], ignore_index=True)
            
            return True
        except Exception as e:
            st.error(f"Error adding stock: {str(e)}")
            return False
    
    def remove_stock(self, symbol):
        """Remove a stock from the portfolio"""
        try:
            st.session_state.portfolio = st.session_state.portfolio[
                st.session_state.portfolio['Symbol'] != symbol
            ].reset_index(drop=True)
            return True
        except Exception as e:
            st.error(f"Error removing stock: {str(e)}")
            return False
    
    def get_portfolio(self):
        """Get the current portfolio"""
        return st.session_state.portfolio.copy()
    
    def calculate_portfolio_metrics(self, portfolio, current_data):
        """Calculate portfolio value, cost, and return"""
        try:
            total_value = 0
            total_cost = 0
            
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                purchase_price = stock['Purchase_Price']
                
                # Calculate cost basis
                cost_basis = shares * purchase_price
                total_cost += cost_basis
                
                # Calculate current value
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    current_value = shares * current_price
                    total_value += current_value
                else:
                    # If we can't get current price, use purchase price
                    total_value += cost_basis
            
            total_return = total_value - total_cost
            
            return total_value, total_cost, total_return
        
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return 0, 0, 0
    
    def get_portfolio_weights(self, portfolio, current_data):
        """Calculate portfolio weights based on current values"""
        try:
            weights = {}
            total_value = 0
            
            # Calculate total portfolio value
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    current_value = shares * current_price
                    total_value += current_value
                    weights[symbol] = current_value
            
            # Convert to percentages
            if total_value > 0:
                for symbol in weights:
                    weights[symbol] = weights[symbol] / total_value
            
            return weights
        
        except Exception as e:
            st.error(f"Error calculating portfolio weights: {str(e)}")
            return {}
    
    def generate_portfolio_report(self):
        """Generate a comprehensive portfolio report"""
        try:
            from data_fetcher import DataFetcher
            
            portfolio = self.get_portfolio()
            if portfolio.empty:
                return pd.DataFrame()
            
            data_fetcher = DataFetcher()
            current_data = data_fetcher.get_portfolio_data(portfolio['Symbol'].tolist())
            
            report_data = []
            
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                purchase_price = stock['Purchase_Price']
                date_added = stock['Date_Added']
                
                # Calculate metrics
                cost_basis = shares * purchase_price
                
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    current_value = shares * current_price
                    gain_loss = current_value - cost_basis
                    gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                    
                    report_data.append({
                        'Symbol': symbol,
                        'Shares': shares,
                        'Purchase Price': purchase_price,
                        'Current Price': current_price,
                        'Cost Basis': cost_basis,
                        'Current Value': current_value,
                        'Gain/Loss ($)': gain_loss,
                        'Gain/Loss (%)': gain_loss_pct,
                        'Date Added': date_added.strftime('%Y-%m-%d') if isinstance(date_added, datetime) else date_added
                    })
            
            return pd.DataFrame(report_data)
        
        except Exception as e:
            st.error(f"Error generating portfolio report: {str(e)}")
            return pd.DataFrame()
    
    def generate_performance_report(self):
        """Generate a performance report with key metrics"""
        try:
            from data_fetcher import DataFetcher
            from risk_calculator import RiskCalculator
            
            portfolio = self.get_portfolio()
            if portfolio.empty:
                return pd.DataFrame()
            
            data_fetcher = DataFetcher()
            risk_calculator = RiskCalculator()
            
            symbols = portfolio['Symbol'].tolist()
            historical_data = data_fetcher.get_portfolio_history(symbols, "1y")
            current_data = data_fetcher.get_portfolio_data(symbols)
            
            if historical_data.empty or not current_data:
                return pd.DataFrame()
            
            # Calculate portfolio metrics
            portfolio_value, total_cost, total_return = self.calculate_portfolio_metrics(portfolio, current_data)
            
            # Calculate risk metrics
            risk_metrics = risk_calculator.calculate_portfolio_risk(historical_data, portfolio)
            
            # Create performance summary
            performance_data = {
                'Metric': [
                    'Total Portfolio Value',
                    'Total Cost Basis',
                    'Total Return ($)',
                    'Total Return (%)',
                    'Portfolio Volatility (%)',
                    'Sharpe Ratio',
                    'Maximum Drawdown (%)',
                    'Number of Holdings'
                ],
                'Value': [
                    f"${portfolio_value:.2f}",
                    f"${total_cost:.2f}",
                    f"${total_return:.2f}",
                    f"{(total_return / total_cost * 100):.2f}%" if total_cost > 0 else "0.00%",
                    f"{risk_metrics['volatility']:.2f}%",
                    f"{risk_metrics['sharpe_ratio']:.2f}",
                    f"{risk_metrics['max_drawdown']:.2f}%",
                    str(len(portfolio))
                ]
            }
            
            return pd.DataFrame(performance_data)
        
        except Exception as e:
            st.error(f"Error generating performance report: {str(e)}")
            return pd.DataFrame()
