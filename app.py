import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from portfolio_manager import PortfolioManager
from risk_calculator import RiskCalculator
from data_fetcher import DataFetcher
from alert_system import AlertSystem
from report_generator import ReportGenerator
from notification_system import NotificationSystem
from alpha_vantage_fetcher import AlphaVantageFetcher
from advanced_analytics import AdvancedAnalytics
import utils

# Configure page
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'risk_calculator' not in st.session_state:
    st.session_state.risk_calculator = RiskCalculator()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator()
if 'notification_system' not in st.session_state:
    st.session_state.notification_system = NotificationSystem()
if 'alpha_vantage' not in st.session_state:
    st.session_state.alpha_vantage = AlphaVantageFetcher()
if 'advanced_analytics' not in st.session_state:
    st.session_state.advanced_analytics = AdvancedAnalytics()

def main():
    st.title("📊 Financial Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.selectbox(
            "Select Dashboard View",
            ["Portfolio Overview", "Risk Analysis", "Market Data", "Advanced Analytics", "Alerts & Reports", "Market News & Sentiment"]
        )
        
        st.markdown("---")
        
        # Portfolio Management
        st.header("Portfolio Management")
        
        # Add stock to portfolio
        with st.expander("Add Stock to Portfolio"):
            symbol = st.text_input("Stock Symbol (e.g., AAPL, GOOGL)")
            shares = st.number_input("Number of Shares", min_value=0.0, step=0.1)
            purchase_price = st.number_input("Purchase Price ($)", min_value=0.0, step=0.01)
            
            if st.button("Add to Portfolio"):
                if symbol and shares > 0 and purchase_price > 0:
                    try:
                        success = st.session_state.portfolio_manager.add_stock(
                            symbol.upper(), shares, purchase_price
                        )
                        if success:
                            st.success(f"Added {shares} shares of {symbol.upper()} to portfolio")
                            st.rerun()
                        else:
                            st.error("Failed to add stock to portfolio")
                    except Exception as e:
                        st.error(f"Error adding stock: {str(e)}")
                else:
                    st.error("Please fill in all fields with valid values")
        
        # Current portfolio
        portfolio = st.session_state.portfolio_manager.get_portfolio()
        if not portfolio.empty:
            st.subheader("Current Portfolio")
            for _, stock in portfolio.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{stock['Symbol']}**: {stock['Shares']} shares")
                with col2:
                    if st.button("🗑️", key=f"remove_{stock['Symbol']}"):
                        st.session_state.portfolio_manager.remove_stock(stock['Symbol'])
                        st.rerun()
        
        st.markdown("---")
        
        # Data refresh
        if st.button("🔄 Refresh Data"):
            st.session_state.data_fetcher.clear_cache()
            st.success("Data refreshed!")
            st.rerun()
    
    # Main content area
    if page == "Portfolio Overview":
        show_portfolio_overview()
    elif page == "Risk Analysis":
        show_risk_analysis()
    elif page == "Market Data":
        show_market_data()
    elif page == "Advanced Analytics":
        show_advanced_analytics()
    elif page == "Alerts & Reports":
        show_alerts_reports()
    elif page == "Market News & Sentiment":
        show_market_news_sentiment()

def show_portfolio_overview():
    st.header("📈 Portfolio Overview")
    
    portfolio = st.session_state.portfolio_manager.get_portfolio()
    
    if portfolio.empty:
        st.info("No stocks in portfolio. Add some stocks using the sidebar to get started.")
        return
    
    # Get current data for all portfolio stocks
    try:
        portfolio_data = st.session_state.data_fetcher.get_portfolio_data(portfolio['Symbol'].tolist())
        
        # Calculate portfolio metrics
        portfolio_value, total_cost, portfolio_return = st.session_state.portfolio_manager.calculate_portfolio_metrics(
            portfolio, portfolio_data
        )
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        
        with col2:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        
        with col3:
            return_pct = (portfolio_return / total_cost * 100) if total_cost > 0 else 0
            st.metric("Total Return", f"${portfolio_return:,.2f}", f"{return_pct:.2f}%")
        
        with col4:
            st.metric("Number of Stocks", len(portfolio))
        
        st.markdown("---")
        
        # Portfolio composition
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Asset Allocation")
            # Calculate current values for each stock
            current_values = []
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                if symbol in portfolio_data:
                    current_price = portfolio_data[symbol]['current_price']
                    current_value = shares * current_price
                    current_values.append(current_value)
                else:
                    current_values.append(0)
            
            # Create pie chart
            if sum(current_values) > 0:
                fig = px.pie(
                    values=current_values,
                    names=portfolio['Symbol'],
                    title="Portfolio Allocation by Current Value"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to fetch current prices for pie chart")
        
        with col2:
            st.subheader("Stock Performance")
            # Create performance table
            performance_data = []
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                purchase_price = stock['Purchase_Price']
                
                if symbol in portfolio_data:
                    current_price = portfolio_data[symbol]['current_price']
                    current_value = shares * current_price
                    cost_basis = shares * purchase_price
                    gain_loss = current_value - cost_basis
                    gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                    
                    performance_data.append({
                        'Symbol': symbol,
                        'Shares': shares,
                        'Purchase Price': f"${purchase_price:.2f}",
                        'Current Price': f"${current_price:.2f}",
                        'Current Value': f"${current_value:.2f}",
                        'Gain/Loss': f"${gain_loss:.2f}",
                        'Gain/Loss %': f"{gain_loss_pct:.2f}%"
                    })
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Unable to fetch current performance data")
        
        # Historical performance chart
        st.subheader("Portfolio Performance Over Time")
        period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        
        try:
            portfolio_history = st.session_state.data_fetcher.get_portfolio_history(
                portfolio['Symbol'].tolist(), period
            )
            
            if not portfolio_history.empty:
                # Calculate weighted portfolio value over time
                portfolio_values = []
                dates = portfolio_history.index
                
                for date in dates:
                    daily_value = 0
                    for _, stock in portfolio.iterrows():
                        symbol = stock['Symbol']
                        shares = stock['Shares']
                        if symbol in portfolio_history.columns:
                            price = portfolio_history.loc[date, symbol]
                            if pd.notna(price):
                                daily_value += shares * price
                    portfolio_values.append(daily_value)
                
                # Create line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to fetch historical data for portfolio chart")
        
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
    
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")

def show_risk_analysis():
    st.header("⚠️ Risk Analysis")
    
    portfolio = st.session_state.portfolio_manager.get_portfolio()
    
    if portfolio.empty:
        st.info("No stocks in portfolio. Add some stocks to perform risk analysis.")
        return
    
    try:
        # Get historical data for risk calculations
        symbols = portfolio['Symbol'].tolist()
        historical_data = st.session_state.data_fetcher.get_portfolio_history(symbols, "1y")
        
        if historical_data.empty:
            st.warning("Unable to fetch historical data for risk analysis")
            return
        
        # Calculate risk metrics
        risk_metrics = st.session_state.risk_calculator.calculate_portfolio_risk(
            historical_data, portfolio
        )
        
        # Display risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Volatility", f"{risk_metrics['volatility']:.2f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2f}%")
        
        st.markdown("---")
        
        # Individual stock risk metrics
        st.subheader("Individual Stock Risk Metrics")
        
        risk_data = []
        for symbol in symbols:
            if symbol in historical_data.columns:
                stock_data = historical_data[symbol].dropna()
                if len(stock_data) > 30:  # Need sufficient data
                    volatility = st.session_state.risk_calculator.calculate_volatility(stock_data)
                    beta = st.session_state.risk_calculator.calculate_beta(stock_data)
                    
                    risk_data.append({
                        'Symbol': symbol,
                        'Volatility (%)': f"{volatility:.2f}",
                        'Beta': f"{beta:.2f}",
                        'Risk Level': utils.get_risk_level(volatility, beta)
                    })
        
        if risk_data:
            df = pd.DataFrame(risk_data)
            st.dataframe(df, use_container_width=True)
        
        # Risk-Return scatter plot
        st.subheader("Risk-Return Analysis")
        
        returns_data = []
        volatility_data = []
        symbols_data = []
        
        for symbol in symbols:
            if symbol in historical_data.columns:
                stock_data = historical_data[symbol].dropna()
                if len(stock_data) > 30:
                    returns = stock_data.pct_change().dropna()
                    annual_return = (returns.mean() * 252) * 100
                    volatility = st.session_state.risk_calculator.calculate_volatility(stock_data)
                    
                    returns_data.append(annual_return)
                    volatility_data.append(volatility)
                    symbols_data.append(symbol)
        
        if returns_data and volatility_data:
            fig = px.scatter(
                x=volatility_data,
                y=returns_data,
                text=symbols_data,
                title="Risk vs Return",
                labels={'x': 'Volatility (%)', 'y': 'Annual Return (%)'}
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        
        # Value at Risk (VaR) calculation
        st.subheader("Value at Risk (VaR)")
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
        
        portfolio_data = st.session_state.data_fetcher.get_portfolio_data(symbols)
        if portfolio_data:
            portfolio_value = sum(
                portfolio[portfolio['Symbol'] == symbol]['Shares'].iloc[0] * 
                portfolio_data[symbol]['current_price']
                for symbol in symbols if symbol in portfolio_data
            )
            
            var_1d = st.session_state.risk_calculator.calculate_var(
                historical_data, portfolio, confidence_level, portfolio_value
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"1-Day VaR ({confidence_level}%)", f"${var_1d:.2f}")
            with col2:
                st.metric(f"1-Week VaR ({confidence_level}%)", f"${var_1d * np.sqrt(5):.2f}")
    
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")

def show_market_data():
    st.header("📊 Market Data")
    
    # Market overview
    st.subheader("Market Indices")
    
    # Get major indices data
    indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow Jones, NASDAQ, Russell 2000
    index_names = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]
    
    try:
        indices_data = st.session_state.data_fetcher.get_market_indices(indices)
        
        cols = st.columns(4)
        for i, (index, name) in enumerate(zip(indices, index_names)):
            with cols[i]:
                if index in indices_data:
                    current_price = indices_data[index]['current_price']
                    change = indices_data[index]['change']
                    change_pct = indices_data[index]['change_percent']
                    
                    st.metric(
                        name,
                        f"{current_price:.2f}",
                        f"{change:.2f} ({change_pct:.2f}%)"
                    )
                else:
                    st.metric(name, "N/A")
    
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
    
    st.markdown("---")
    
    # Individual stock analysis
    st.subheader("Individual Stock Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analysis_symbol = st.text_input("Enter Stock Symbol", placeholder="AAPL")
        period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        
        if st.button("Analyze Stock"):
            if analysis_symbol:
                st.session_state.analysis_symbol = analysis_symbol.upper()
                st.session_state.analysis_period = period
    
    with col2:
        if hasattr(st.session_state, 'analysis_symbol') and st.session_state.analysis_symbol:
            try:
                # Get stock data
                stock_data = st.session_state.data_fetcher.get_stock_data(
                    st.session_state.analysis_symbol, 
                    st.session_state.analysis_period
                )
                
                if stock_data is not None and not stock_data.empty:
                    # Stock info
                    current_price = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    st.metric(
                        st.session_state.analysis_symbol,
                        f"${current_price:.2f}",
                        f"{change:.2f} ({change_pct:.2f}%)"
                    )
                    
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{st.session_state.analysis_symbol} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    
                    fig_volume.update_layout(
                        title=f"{st.session_state.analysis_symbol} Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                
                else:
                    st.error(f"Unable to fetch data for {st.session_state.analysis_symbol}")
            
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")

def show_alerts_reports():
    st.header("🚨 Alerts & Reports")
    
    # Alert configuration
    st.subheader("Alert Configuration")
    
    portfolio = st.session_state.portfolio_manager.get_portfolio()
    
    if portfolio.empty:
        st.info("No stocks in portfolio. Add some stocks to configure alerts.")
        return
    
    # Configure alerts
    with st.expander("Configure Price Alerts"):
        symbol = st.selectbox("Select Stock", portfolio['Symbol'].tolist())
        alert_type = st.selectbox("Alert Type", ["Price Drop", "Price Rise", "Volume Spike"])
        threshold = st.number_input("Threshold (%)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        
        if st.button("Set Alert"):
            st.session_state.alert_system.add_alert(symbol, alert_type, threshold)
            st.success(f"Alert set for {symbol}: {alert_type} at {threshold}%")
    
    # Check for alerts
    st.subheader("Active Alerts")
    try:
        alerts = st.session_state.alert_system.check_alerts(portfolio)
        
        if alerts:
            for alert in alerts:
                alert_type = "🔴" if "Drop" in alert else "🟢"
                st.warning(f"{alert_type} {alert}")
        else:
            st.info("No active alerts")
    
    except Exception as e:
        st.error(f"Error checking alerts: {str(e)}")
    
    st.markdown("---")
    
    # Report generation
    st.subheader("Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", ["Portfolio Summary", "Risk Analysis", "Performance Report"])
        report_format = st.selectbox("Report Format", ["PDF", "Excel", "CSV"])
        
        if st.button("Generate Report"):
            try:
                # Generate report data
                report_data = st.session_state.report_generator.generate_portfolio_summary_report(
                    st.session_state.portfolio_manager,
                    st.session_state.data_fetcher,
                    st.session_state.risk_calculator
                )
                
                if report_data:
                    if report_format == "PDF":
                        pdf_data = st.session_state.report_generator.create_pdf_report(report_data)
                        if pdf_data:
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_data,
                                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                    
                    elif report_format == "Excel":
                        excel_data = st.session_state.report_generator.create_excel_report(report_data)
                        if excel_data:
                            st.download_button(
                                label="Download Excel Report",
                                data=excel_data,
                                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    elif report_format == "CSV":
                        # Create CSV from holdings data
                        if report_data.get('holdings'):
                            df = pd.DataFrame(report_data['holdings'])
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV Report",
                                data=csv_data,
                                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    
                    st.success("Report generated successfully!")
                else:
                    st.error("Failed to generate report")
            
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    with col2:
        st.subheader("Scheduled Reports")
        
        # Email notification setup
        with st.expander("Email Notification Setup"):
            st.write("Configure email settings for automated reports and alerts")
            
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
            email_username = st.text_input("Email Username")
            email_password = st.text_input("Email Password", type="password")
            
            if st.button("Configure Email"):
                success = st.session_state.notification_system.configure_email_settings(
                    smtp_server, smtp_port, email_username, email_password
                )
                if success:
                    st.success("Email configuration saved!")
                else:
                    st.error("Failed to configure email settings")
            
            # Test email connection
            if st.button("Test Email Connection"):
                success, message = st.session_state.notification_system.test_email_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Schedule reports
        with st.expander("Schedule Automated Reports"):
            recipient_email = st.text_input("Recipient Email")
            report_frequency = st.selectbox("Report Frequency", ["Daily", "Weekly", "Monthly"])
            
            if st.button("Schedule Report"):
                if recipient_email and "@" in recipient_email:
                    # Store scheduled report configuration
                    if 'scheduled_reports' not in st.session_state:
                        st.session_state.scheduled_reports = []
                    
                    scheduled_report = {
                        'recipient': recipient_email,
                        'frequency': report_frequency.lower(),
                        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'active': True
                    }
                    
                    st.session_state.scheduled_reports.append(scheduled_report)
                    st.success(f"Scheduled {report_frequency.lower()} report for {recipient_email}")
                else:
                    st.error("Please enter a valid email address")
        
        # Show scheduled reports
        if 'scheduled_reports' in st.session_state and st.session_state.scheduled_reports:
            st.subheader("Active Scheduled Reports")
            for i, report in enumerate(st.session_state.scheduled_reports):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{report['frequency'].title()}** report to {report['recipient']}")
                    st.write(f"Created: {report['created_date']}")
                with col_b:
                    if st.button("Remove", key=f"remove_report_{i}"):
                        st.session_state.scheduled_reports.pop(i)
                        st.rerun()
    
    st.markdown("---")
    
    # Alert management
    st.subheader("Alert Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Alerts**")
        alert_report = st.session_state.alert_system.generate_alert_report()
        if not alert_report.empty:
            st.dataframe(alert_report, use_container_width=True)
            
            # Remove alerts
            if st.button("Clear All Triggered Alerts"):
                st.session_state.alert_system.clear_triggered_alerts()
                st.success("Triggered alerts cleared")
                st.rerun()
        else:
            st.info("No alerts configured")
    
    with col2:
        st.write("**Portfolio Alerts**")
        try:
            portfolio_alerts = st.session_state.alert_system.check_portfolio_alerts(portfolio)
            if portfolio_alerts:
                for alert in portfolio_alerts:
                    st.warning(alert)
            else:
                st.info("No portfolio alerts")
        except Exception as e:
            st.error(f"Error checking portfolio alerts: {str(e)}")
        
        # Alert summary
        alert_summary = st.session_state.alert_system.get_alert_summary()
        st.metric("Total Alerts", alert_summary['total_alerts'])
        st.metric("Active Alerts", alert_summary['active_alerts'])
        st.metric("Triggered Alerts", alert_summary['triggered_alerts'])
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Send Test Email Alert"):
            # Send test email
            portfolio_data = st.session_state.data_fetcher.get_portfolio_data(portfolio['Symbol'].tolist())
            if portfolio_data:
                portfolio_value, total_cost, total_return = st.session_state.portfolio_manager.calculate_portfolio_metrics(
                    portfolio, portfolio_data
                )
                
                portfolio_summary = {
                    'total_value': portfolio_value,
                    'total_return': total_return,
                    'return_percentage': (total_return / total_cost * 100) if total_cost > 0 else 0,
                    'num_holdings': len(portfolio)
                }
                
                # For demo purposes, we'll show a success message
                st.success("Test email alert would be sent! (Configure email settings to enable)")
    
    with col2:
        if st.button("Generate Market Summary"):
            try:
                market_data = st.session_state.report_generator.generate_market_summary_report(
                    st.session_state.data_fetcher
                )
                if market_data:
                    st.success("Market summary generated!")
                    
                    # Display market summary
                    st.subheader("Market Summary")
                    for index in market_data['indices']:
                        st.metric(
                            index['name'],
                            f"{index['current_price']:.2f}",
                            f"{index['change_percent']:+.2f}%"
                        )
                else:
                    st.error("Failed to generate market summary")
            except Exception as e:
                st.error(f"Error generating market summary: {str(e)}")
    
    with col3:
        if st.button("Set Default Alerts"):
            success = st.session_state.alert_system.set_default_portfolio_alerts(portfolio)
            if success:
                st.success("Default alerts set for all portfolio stocks")
                st.rerun()
            else:
                st.error("Failed to set default alerts")

def show_advanced_analytics():
    st.header("📊 Advanced Analytics")
    
    portfolio = st.session_state.portfolio_manager.get_portfolio()
    
    if portfolio.empty:
        st.info("No stocks in portfolio. Add some stocks to view advanced analytics.")
        return
    
    # Portfolio performance metrics
    st.subheader("Enhanced Portfolio Metrics")
    
    try:
        symbols = portfolio['Symbol'].tolist()
        historical_data = st.session_state.data_fetcher.get_portfolio_history(symbols, "1y")
        
        if not historical_data.empty:
            # Calculate comprehensive performance metrics
            performance_metrics = st.session_state.advanced_analytics.calculate_portfolio_performance_metrics(
                portfolio, historical_data
            )
            
            if performance_metrics:
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Alpha", f"{performance_metrics.get('alpha', 0):.2f}%")
                    st.metric("Beta", f"{performance_metrics.get('beta', 1.0):.2f}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{performance_metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Sortino Ratio", f"{performance_metrics.get('sortino_ratio', 0):.2f}")
                
                with col3:
                    st.metric("Treynor Ratio", f"{performance_metrics.get('treynor_ratio', 0):.2f}")
                    st.metric("Information Ratio", f"{performance_metrics.get('information_ratio', 0):.2f}")
                
                with col4:
                    st.metric("Calmar Ratio", f"{performance_metrics.get('calmar_ratio', 0):.2f}")
                    st.metric("Max Drawdown", f"{performance_metrics.get('max_drawdown', 0)*100:.2f}%")
                
                st.markdown("---")
                
                # Risk metrics
                st.subheader("Risk Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("VaR (95%)", f"{performance_metrics.get('var_95', 0)*100:.2f}%")
                    st.metric("CVaR (95%)", f"{performance_metrics.get('cvar_95', 0)*100:.2f}%")
                
                with col2:
                    st.metric("VaR (99%)", f"{performance_metrics.get('var_99', 0)*100:.2f}%")
                    st.metric("Annual Volatility", f"{performance_metrics.get('volatility', 0)*100:.2f}%")
        
        # Performance attribution
        st.subheader("Performance Attribution")
        
        attribution_data = st.session_state.advanced_analytics.generate_performance_attribution(
            portfolio, historical_data
        )
        
        if not attribution_data.empty:
            st.dataframe(attribution_data, use_container_width=True)
        
        # Risk-Return Chart
        st.subheader("Risk-Return Analysis")
        
        risk_return_chart = st.session_state.advanced_analytics.create_risk_return_chart(
            portfolio, historical_data
        )
        
        if risk_return_chart:
            st.plotly_chart(risk_return_chart, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        
        correlation_chart = st.session_state.advanced_analytics.create_correlation_heatmap(historical_data)
        
        if correlation_chart:
            st.plotly_chart(correlation_chart, use_container_width=True)
        
        # Portfolio Optimization Suggestions
        st.subheader("Portfolio Optimization Suggestions")
        
        suggestions = st.session_state.advanced_analytics.calculate_portfolio_optimization_suggestions(
            portfolio, historical_data
        )
        
        if suggestions:
            for suggestion in suggestions:
                if suggestion['severity'] == 'High':
                    st.error(f"🔴 {suggestion['type']}: {suggestion['message']}")
                elif suggestion['severity'] == 'Medium':
                    st.warning(f"🟡 {suggestion['type']}: {suggestion['message']}")
                else:
                    st.info(f"🟢 {suggestion['type']}: {suggestion['message']}")
        else:
            st.success("No optimization suggestions at this time. Your portfolio appears well-balanced.")
    
    except Exception as e:
        st.error(f"Error calculating advanced analytics: {str(e)}")
    
    st.markdown("---")
    
    # Alpha Vantage Enhanced Data
    st.subheader("Enhanced Market Data (Alpha Vantage)")
    
    selected_symbol = st.selectbox("Select Stock for Detailed Analysis", portfolio['Symbol'].tolist())
    
    if selected_symbol:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Company Overview")
            try:
                company_overview = st.session_state.alpha_vantage.get_company_overview(selected_symbol)
                if company_overview:
                    st.write(f"**Company**: {company_overview.get('Name', 'N/A')}")
                    st.write(f"**Sector**: {company_overview.get('Sector', 'N/A')}")
                    st.write(f"**Industry**: {company_overview.get('Industry', 'N/A')}")
                    st.write(f"**Market Cap**: ${company_overview.get('MarketCapitalization', 0):,.0f}")
                    st.write(f"**P/E Ratio**: {company_overview.get('PERatio', 'N/A')}")
                    st.write(f"**Dividend Yield**: {company_overview.get('DividendYield', 'N/A')}")
                    st.write(f"**Beta**: {company_overview.get('Beta', 'N/A')}")
                    st.write(f"**52 Week High**: ${company_overview.get('52WeekHigh', 0):.2f}")
                    st.write(f"**52 Week Low**: ${company_overview.get('52WeekLow', 0):.2f}")
                else:
                    st.warning("Company overview data not available")
            except Exception as e:
                st.error(f"Error fetching company overview: {str(e)}")
        
        with col2:
            st.subheader("Real-time Quote")
            try:
                quote = st.session_state.alpha_vantage.get_stock_quote(selected_symbol)
                if quote:
                    st.metric("Current Price", f"${quote['price']:.2f}", f"{quote['change_percent']}%")
                    st.write(f"**Volume**: {quote['volume']:,}")
                    st.write(f"**Open**: ${quote['open']:.2f}")
                    st.write(f"**High**: ${quote['high']:.2f}")
                    st.write(f"**Low**: ${quote['low']:.2f}")
                    st.write(f"**Previous Close**: ${quote['previous_close']:.2f}")
                    st.write(f"**Trading Day**: {quote['latest_trading_day']}")
                else:
                    st.warning("Real-time quote not available")
            except Exception as e:
                st.error(f"Error fetching real-time quote: {str(e)}")

def show_market_news_sentiment():
    st.header("📰 Market News & Sentiment")
    
    # Market News
    st.subheader("Latest Market News")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topics = st.multiselect(
            "Select News Topics",
            ["technology", "earnings", "finance", "ipo", "mergers_and_acquisitions", "financial_markets"],
            default=["technology", "earnings"]
        )
    
    with col2:
        news_limit = st.slider("Number of Articles", 10, 100, 50)
    
    if st.button("Fetch Latest News"):
        try:
            topics_str = ",".join(topics) if topics else "technology,earnings"
            news_data = st.session_state.alpha_vantage.get_market_news(topics_str, limit=news_limit)
            
            if news_data:
                st.session_state.current_news = news_data
                st.success(f"Fetched {len(news_data)} news articles")
            else:
                st.error("Failed to fetch news data")
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
    
    # Display news if available
    if hasattr(st.session_state, 'current_news') and st.session_state.current_news:
        st.subheader("News Articles")
        
        # Calculate sentiment statistics
        sentiments = [article['sentiment_score'] for article in st.session_state.current_news]
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        positive_count = sum(1 for score in sentiments if score > 0.1)
        negative_count = sum(1 for score in sentiments if score < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Display sentiment overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        
        with col2:
            st.metric("Positive Articles", positive_count)
        
        with col3:
            st.metric("Negative Articles", negative_count)
        
        with col4:
            st.metric("Neutral Articles", neutral_count)
        
        st.markdown("---")
        
        # Display articles
        for i, article in enumerate(st.session_state.current_news[:20]):  # Show first 20 articles
            with st.expander(f"📄 {article['title'][:100]}..." if len(article['title']) > 100 else f"📄 {article['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Source**: {article['source']}")
                    st.write(f"**Published**: {article['published']}")
                    st.write(f"**Summary**: {article['summary']}")
                    if article['url']:
                        st.write(f"**[Read Full Article]({article['url']})**")
                
                with col2:
                    sentiment_color = "🟢" if article['sentiment_score'] > 0.1 else "🔴" if article['sentiment_score'] < -0.1 else "🟡"
                    st.metric("Sentiment", f"{sentiment_color} {article['sentiment_label']}")
                    st.metric("Score", f"{article['sentiment_score']:.2f}")
                    
                    if article['topics']:
                        st.write("**Topics**:")
                        for topic in article['topics'][:3]:  # Show first 3 topics
                            st.write(f"- {topic}")
    
    st.markdown("---")
    
    # Sector Performance
    st.subheader("Sector Performance")
    
    if st.button("Fetch Sector Performance"):
        try:
            sector_data = st.session_state.alpha_vantage.get_sector_performance()
            if sector_data:
                st.session_state.sector_performance = sector_data
                st.success("Sector performance data fetched")
            else:
                st.error("Failed to fetch sector performance data")
        except Exception as e:
            st.error(f"Error fetching sector performance: {str(e)}")
    
    # Display sector performance
    if hasattr(st.session_state, 'sector_performance') and st.session_state.sector_performance:
        time_period = st.selectbox(
            "Select Time Period",
            list(st.session_state.sector_performance.keys()),
            index=0
        )
        
        if time_period in st.session_state.sector_performance:
            sector_perf = st.session_state.sector_performance[time_period]
            
            # Create DataFrame for display
            sector_df = pd.DataFrame(list(sector_perf.items()), columns=['Sector', 'Performance'])
            sector_df['Performance'] = sector_df['Performance'].str.replace('%', '').astype(float)
            sector_df = sector_df.sort_values('Performance', ascending=False)
            
            # Display as chart
            fig = px.bar(
                sector_df, 
                x='Performance', 
                y='Sector', 
                title=f"Sector Performance - {time_period.replace('Rank ', '').replace(':', '')}",
                orientation='h'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Display as table
            st.dataframe(sector_df, use_container_width=True)
    
    # API Usage Statistics
    st.subheader("API Usage Statistics")
    
    usage_stats = st.session_state.alpha_vantage.get_api_usage_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("API Calls Made", usage_stats['calls_made'])
    
    with col2:
        if usage_stats['last_call_time']:
            last_call = datetime.fromtimestamp(usage_stats['last_call_time']).strftime('%H:%M:%S')
            st.metric("Last API Call", last_call)
        else:
            st.metric("Last API Call", "Never")

if __name__ == "__main__":
    main()
