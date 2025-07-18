import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import portfolio_manager as pm
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
# Remove PortfolioManager session state initialization
# if 'portfolio_manager' not in st.session_state:
#     st.session_state.portfolio_manager = PortfolioManager()
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

# Initialize database at app start
pm.init_db()
pm.sync_db_to_session()

def main():
    st.title("📊 Financial Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.selectbox(
            "Select Dashboard View",
            ["Portfolio Overview", "Market Data", "Market News & Sentiment"]
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
                        pm.add_stock(symbol.upper(), None, shares, purchase_price, str(datetime.now()))
                        pm.sync_db_to_session()
                        st.success(f"Added {shares} shares of {symbol.upper()} to portfolio")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding stock: {str(e)}")
                else:
                    st.error("Please fill in all fields with valid values")
        # Current portfolio (from DB)
        stocks = pm.get_all_stocks()
        import pandas as pd
        if stocks:
            st.subheader("Current Portfolio (DB)")
            df = pd.DataFrame(stocks, columns=pd.Index(["ID", "Symbol", "Name", "Shares", "Purchase Price", "Date Added"]))
            for _, row in df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['Symbol']}**: {row['Shares']} shares @ ${row['Purchase Price']}")
                with col2:
                    if st.button("🗑️", key=f"remove_db_{row['ID']}"):
                        pm.remove_stock(row['ID'])
                        pm.sync_db_to_session()
                        st.rerun()
        else:
            st.info("No stocks in portfolio. Add some stocks using the sidebar to get started.")
        
        st.markdown("---")
        
        # Data refresh
        if st.button("🔄 Refresh Data"):
            st.session_state.data_fetcher.clear_cache()
            st.success("Data refreshed!")
            st.rerun()
    
    # Always set dark mode theme
    from pathlib import Path
    theme_config = {
        "theme": {
            "base": "dark",
            "primaryColor": "#1f77b4",
            "backgroundColor": "#0e1117",
            "secondaryBackgroundColor": "#262730",
            "textColor": "#fafafa"
        }
    }
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.toml"
    import toml
    with open(config_path, "w") as f:
        toml.dump(theme_config, f)

    # Main content area
    if page == "Portfolio Overview":
        show_portfolio_overview()
    elif page == "Market Data":
        show_market_data()
    elif page == "Market News & Sentiment":
        show_market_news_sentiment()

def show_portfolio_overview():
    st.header("📈 Portfolio Overview")
    
    portfolio = st.session_state.portfolio
    
    if portfolio.empty:
        st.info("No stocks in portfolio. Add some stocks using the sidebar to get started.")
        return
    
    # Get current data for all portfolio stocks
    try:
        portfolio_data = st.session_state.data_fetcher.get_portfolio_data(portfolio['Symbol'].tolist())
        
        # Calculate portfolio metrics
        portfolio_value, total_cost, portfolio_return = pm.PortfolioManager().calculate_portfolio_metrics(
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
            symbols = portfolio['Symbol'].tolist()
            if not symbols:
                st.info("Add stocks to your portfolio to see performance charts")
            else:
                portfolio_history = st.session_state.data_fetcher.get_portfolio_history(
                    symbols, period
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
                    if portfolio_values and len(portfolio_values) > 0:
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
                        st.warning("No portfolio data to display")
                else:
                    st.warning("Unable to fetch historical data for portfolio chart")
        
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
    
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")

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
            sector_df = pd.DataFrame(list(sector_perf.items()), columns=pd.Index(['Sector', 'Performance']))
            if sector_df['Performance'].dtype == object:
                sector_df['Performance'] = sector_df['Performance'].fillna('0')
                sector_df['Performance'] = sector_df['Performance'].apply(lambda x: str(x).replace('%', '') if x is not None else '0').astype(float)
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
