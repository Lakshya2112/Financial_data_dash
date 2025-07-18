import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from data_fetcher import DataFetcher

class AlertSystem:
    def __init__(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        
        self.data_fetcher = DataFetcher()
    
    def add_alert(self, symbol, alert_type, threshold):
        """Add a new alert"""
        try:
            alert = {
                'symbol': symbol.upper(),
                'type': alert_type,
                'threshold': threshold,
                'created_date': datetime.now(),
                'triggered': False
            }
            
            st.session_state.alerts.append(alert)
            return True
        
        except Exception as e:
            st.error(f"Error adding alert: {str(e)}")
            return False
    
    def remove_alert(self, index):
        """Remove an alert by index"""
        try:
            if 0 <= index < len(st.session_state.alerts):
                st.session_state.alerts.pop(index)
                return True
            return False
        
        except Exception as e:
            st.error(f"Error removing alert: {str(e)}")
            return False
    
    def check_alerts(self, portfolio):
        """Check for triggered alerts"""
        try:
            triggered_alerts = []
            
            if not st.session_state.alerts:
                return triggered_alerts
            
            # Get current data for all portfolio stocks
            symbols = portfolio['Symbol'].tolist()
            current_data = self.data_fetcher.get_portfolio_data(symbols)
            
            for alert in st.session_state.alerts:
                symbol = alert['symbol']
                alert_type = alert['type']
                threshold = alert['threshold']
                
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    change_percent = current_data[symbol]['change_percent']
                    volume = current_data[symbol]['volume']
                    
                    # Check different alert types
                    if alert_type == "Price Drop" and change_percent <= -threshold:
                        triggered_alerts.append(
                            f"Price Drop Alert: {symbol} dropped {abs(change_percent):.2f}% "
                            f"(threshold: {threshold}%)"
                        )
                        alert['triggered'] = True
                    
                    elif alert_type == "Price Rise" and change_percent >= threshold:
                        triggered_alerts.append(
                            f"Price Rise Alert: {symbol} rose {change_percent:.2f}% "
                            f"(threshold: {threshold}%)"
                        )
                        alert['triggered'] = True
                    
                    elif alert_type == "Volume Spike":
                        # For volume spike, we need average volume comparison
                        try:
                            stock_info = self.data_fetcher.get_stock_info(symbol)
                            avg_volume = stock_info.get('average_volume', 0)
                            
                            if avg_volume > 0:
                                volume_change = ((volume - avg_volume) / avg_volume) * 100
                                if volume_change >= threshold:
                                    triggered_alerts.append(
                                        f"Volume Spike Alert: {symbol} volume increased {volume_change:.2f}% "
                                        f"(threshold: {threshold}%)"
                                    )
                                    alert['triggered'] = True
                        except:
                            pass  # Skip if we can't get volume data
            
            return triggered_alerts
        
        except Exception as e:
            st.error(f"Error checking alerts: {str(e)}")
            return []
    
    def get_active_alerts(self):
        """Get all active alerts"""
        return [alert for alert in st.session_state.alerts if not alert['triggered']]
    
    def get_triggered_alerts(self):
        """Get all triggered alerts"""
        return [alert for alert in st.session_state.alerts if alert['triggered']]
    
    def clear_triggered_alerts(self):
        """Clear all triggered alerts"""
        try:
            st.session_state.alerts = [alert for alert in st.session_state.alerts if not alert['triggered']]
            return True
        except Exception as e:
            st.error(f"Error clearing triggered alerts: {str(e)}")
            return False
    
    def check_portfolio_alerts(self, portfolio):
        """Check for portfolio-wide alerts"""
        try:
            portfolio_alerts = []
            
            if portfolio.empty:
                return portfolio_alerts
            
            # Get current portfolio data
            symbols = portfolio['Symbol'].tolist()
            current_data = self.data_fetcher.get_portfolio_data(symbols)
            
            if not current_data:
                return portfolio_alerts
            
            # Calculate portfolio metrics
            total_value = 0
            total_cost = 0
            
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                purchase_price = stock['Purchase_Price']
                
                cost_basis = shares * purchase_price
                total_cost += cost_basis
                
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    current_value = shares * current_price
                    total_value += current_value
                else:
                    total_value += cost_basis
            
            # Calculate portfolio performance
            if total_cost > 0:
                portfolio_return_pct = ((total_value - total_cost) / total_cost) * 100
                
                # Check for significant portfolio changes
                if portfolio_return_pct <= -10:
                    portfolio_alerts.append(
                        f"Portfolio Alert: Portfolio down {abs(portfolio_return_pct):.2f}% "
                        f"(Current value: ${total_value:.2f})"
                    )
                elif portfolio_return_pct >= 20:
                    portfolio_alerts.append(
                        f"Portfolio Alert: Portfolio up {portfolio_return_pct:.2f}% "
                        f"(Current value: ${total_value:.2f})"
                    )
            
            # Check for individual stock alerts
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                purchase_price = stock['Purchase_Price']
                
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    price_change_pct = ((current_price - purchase_price) / purchase_price) * 100
                    
                    # Alert for significant individual stock changes
                    if price_change_pct <= -15:
                        portfolio_alerts.append(
                            f"Stock Alert: {symbol} down {abs(price_change_pct):.2f}% "
                            f"from your purchase price"
                        )
                    elif price_change_pct >= 25:
                        portfolio_alerts.append(
                            f"Stock Alert: {symbol} up {price_change_pct:.2f}% "
                            f"from your purchase price"
                        )
            
            return portfolio_alerts
        
        except Exception as e:
            st.error(f"Error checking portfolio alerts: {str(e)}")
            return []
    
    def generate_alert_report(self):
        """Generate a report of all alerts"""
        try:
            if not st.session_state.alerts:
                return pd.DataFrame()
            
            alert_data = []
            for i, alert in enumerate(st.session_state.alerts):
                alert_data.append({
                    'Index': i,
                    'Symbol': alert['symbol'],
                    'Type': alert['type'],
                    'Threshold': f"{alert['threshold']}%",
                    'Status': 'Triggered' if alert['triggered'] else 'Active',
                    'Created': alert['created_date'].strftime('%Y-%m-%d %H:%M')
                })
            
            return pd.DataFrame(alert_data)
        
        except Exception as e:
            st.error(f"Error generating alert report: {str(e)}")
            return pd.DataFrame()
    
    def set_default_portfolio_alerts(self, portfolio):
        """Set default alerts for portfolio stocks"""
        try:
            default_alerts = [
                {'type': 'Price Drop', 'threshold': 5.0},
                {'type': 'Price Rise', 'threshold': 10.0},
                {'type': 'Volume Spike', 'threshold': 50.0}
            ]
            
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                
                for alert_config in default_alerts:
                    # Check if alert already exists
                    existing_alert = any(
                        alert['symbol'] == symbol and 
                        alert['type'] == alert_config['type']
                        for alert in st.session_state.alerts
                    )
                    
                    if not existing_alert:
                        self.add_alert(symbol, alert_config['type'], alert_config['threshold'])
            
            return True
        
        except Exception as e:
            st.error(f"Error setting default alerts: {str(e)}")
            return False
    
    def get_alert_summary(self):
        """Get a summary of alert statistics"""
        try:
            if not st.session_state.alerts:
                return {
                    'total_alerts': 0,
                    'active_alerts': 0,
                    'triggered_alerts': 0,
                    'symbols_monitored': 0
                }
            
            total_alerts = len(st.session_state.alerts)
            active_alerts = len(self.get_active_alerts())
            triggered_alerts = len(self.get_triggered_alerts())
            
            symbols_monitored = len(set(alert['symbol'] for alert in st.session_state.alerts))
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'triggered_alerts': triggered_alerts,
                'symbols_monitored': symbols_monitored
            }
        
        except Exception as e:
            st.error(f"Error getting alert summary: {str(e)}")
            return {
                'total_alerts': 0,
                'active_alerts': 0,
                'triggered_alerts': 0,
                'symbols_monitored': 0
            }
