# Financial Dashboard

## Overview

This is a comprehensive financial dashboard application built with Streamlit that provides real-time portfolio management, risk analysis, and market monitoring capabilities. The application integrates with Yahoo Finance API to fetch live market data and offers automated alerting functionality for investment monitoring.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web-based dashboard interface
- **Layout**: Wide layout with expandable sidebar for navigation and controls
- **State Management**: Streamlit session state for maintaining user data across interactions
- **Visualization**: Plotly for interactive charts and graphs (integrated but not shown in current files)

### Backend Architecture
- **Modular Design**: Object-oriented architecture with separate classes for different functionalities
- **Data Layer**: Real-time data fetching from Yahoo Finance API with built-in caching
- **Business Logic**: Separated into specialized managers for portfolio, risk, and alerts
- **Utilities**: Helper functions for formatting and calculations

## Key Components

### 1. Portfolio Management (`portfolio_manager.py`)
- **Purpose**: Manages user's stock portfolio with add/remove functionality
- **Features**: 
  - Weighted average price calculation for existing positions
  - DataFrame-based storage using Streamlit session state
  - Date tracking for portfolio additions

### 2. Risk Analysis (`risk_calculator.py`)
- **Purpose**: Calculates financial risk metrics for portfolio assessment
- **Metrics**:
  - Volatility calculation (annualized)
  - Beta calculation relative to S&P 500
  - Risk-free rate consideration (2% annual)

### 3. Data Fetching (`data_fetcher.py`)
- **Purpose**: Handles all external API interactions with Yahoo Finance
- **Features**:
  - Cached data retrieval (5-minute TTL)
  - Historical and real-time stock data
  - Error handling for invalid symbols
  - Portfolio-wide data aggregation

### 4. Alert System (`alert_system.py`)
- **Purpose**: Manages price and threshold-based alerts
- **Features**:
  - Configurable alert types and thresholds
  - Alert creation and removal
  - Trigger checking against portfolio data

### 5. Utilities (`utils.py`)
- **Purpose**: Common formatting and calculation functions
- **Functions**:
  - Currency formatting with K/M/B suffixes
  - Percentage formatting
  - Risk level determination
  - Date calculations

## Data Flow

1. **User Input**: Users interact with Streamlit interface to add stocks, set alerts, and configure views
2. **Data Fetching**: DataFetcher class retrieves real-time data from Yahoo Finance API
3. **Processing**: Risk calculations and portfolio analytics are performed using fetched data
4. **Storage**: All user data is maintained in Streamlit session state (temporary storage)
5. **Display**: Results are rendered through Streamlit components with Plotly visualizations
6. **Alerts**: Alert system continuously monitors portfolio against user-defined thresholds

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **yfinance**: Yahoo Finance API wrapper for stock data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive visualizations

### Data Sources
- **Yahoo Finance**: Primary data source for stock prices, historical data, and company information
- **S&P 500 Index**: Used as market benchmark for beta calculations

## Deployment Strategy

### Current Implementation
- **Local Development**: Designed for local Streamlit execution
- **Session-based Storage**: No persistent database, relies on browser session
- **Real-time Data**: API calls made on-demand with caching

### Scalability Considerations
- **Caching**: 5-minute TTL on data fetching to reduce API calls
- **Error Handling**: Comprehensive error handling for API failures
- **Modular Design**: Easy to extend with additional features or data sources

### Potential Enhancements
- **Database Integration**: Could add PostgreSQL for persistent storage
- **Authentication**: Currently no user authentication system
- **Email Alerts**: Alert system structure exists but email functionality not implemented
- **Scheduled Reports**: Framework ready for automated reporting features

## Development Notes

The application follows a clean separation of concerns with each component handling specific functionality. The codebase is designed to be easily extensible, with clear interfaces between components. The lack of persistent storage makes this suitable for demo purposes but would need database integration for production use.