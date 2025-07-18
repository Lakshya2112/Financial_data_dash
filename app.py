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

def main():
    st.title("📊 Financial Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.selectbox(
            "Select Dashboard View",
            ["Portfolio Overview", "Risk Analysis", "Market Data", "Alerts & Reports"]
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
    elif page == "Alerts & Reports":
        show_alerts_reports()

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
        
        if st.button("Generate Report"):
            try:
                if report_type == "Portfolio Summary":
                    report_data = st.session_state.portfolio_manager.generate_portfolio_report()
                elif report_type == "Risk Analysis":
                    symbols = portfolio['Symbol'].tolist()
                    historical_data = st.session_state.data_fetcher.get_portfolio_history(symbols, "1y")
                    report_data = st.session_state.risk_calculator.generate_risk_report(
                        historical_data, portfolio
                    )
                else:  # Performance Report
                    report_data = st.session_state.portfolio_manager.generate_performance_report()
                
                st.session_state.current_report = report_data
                st.success("Report generated successfully!")
            
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'current_report') and st.session_state.current_report is not None:
            # Display report
            st.subheader("Generated Report")
            
            # Convert to CSV for download
            csv = st.session_state.current_report.to_csv(index=False)
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name=f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Display report preview
            st.dataframe(st.session_state.current_report, use_container_width=True)

if __name__ == "__main__":
    main()
