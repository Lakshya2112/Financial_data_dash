import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def generate_portfolio_summary_report(self, portfolio_manager, data_fetcher, risk_calculator):
        """Generate a comprehensive portfolio summary report"""
        try:
            portfolio = portfolio_manager.get_portfolio()
            if portfolio.empty:
                return None
            
            # Get current data
            symbols = portfolio['Symbol'].tolist()
            current_data = data_fetcher.get_portfolio_data(symbols)
            historical_data = data_fetcher.get_portfolio_history(symbols, "1y")
            
            # Calculate metrics
            portfolio_value, total_cost, total_return = portfolio_manager.calculate_portfolio_metrics(
                portfolio, current_data
            )
            
            risk_metrics = risk_calculator.calculate_portfolio_risk(historical_data, portfolio)
            
            # Create report data
            report_data = {
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'portfolio_value': portfolio_value,
                'total_cost': total_cost,
                'total_return': total_return,
                'return_percentage': (total_return / total_cost * 100) if total_cost > 0 else 0,
                'num_holdings': len(portfolio),
                'risk_metrics': risk_metrics,
                'holdings': []
            }
            
            # Add individual holdings
            for _, stock in portfolio.iterrows():
                symbol = stock['Symbol']
                shares = stock['Shares']
                purchase_price = stock['Purchase_Price']
                
                if symbol in current_data:
                    current_price = current_data[symbol]['current_price']
                    current_value = shares * current_price
                    cost_basis = shares * purchase_price
                    gain_loss = current_value - cost_basis
                    gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                    
                    report_data['holdings'].append({
                        'symbol': symbol,
                        'shares': shares,
                        'purchase_price': purchase_price,
                        'current_price': current_price,
                        'current_value': current_value,
                        'cost_basis': cost_basis,
                        'gain_loss': gain_loss,
                        'gain_loss_pct': gain_loss_pct
                    })
            
            return report_data
        
        except Exception as e:
            st.error(f"Error generating portfolio summary: {str(e)}")
            return None
    
    def create_pdf_report(self, report_data):
        """Create a PDF report from report data"""
        try:
            if not report_data:
                return None
            
            # Create PDF buffer
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            
            # Title
            title = Paragraph("Portfolio Performance Report", self.styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Report date
            date_text = f"Generated on: {report_data['generation_date']}"
            date_para = Paragraph(date_text, self.styles['Normal'])
            story.append(date_para)
            story.append(Spacer(1, 12))
            
            # Portfolio summary
            summary_title = Paragraph("Portfolio Summary", self.styles['Heading2'])
            story.append(summary_title)
            
            summary_data = [
                ['Metric', 'Value'],
                ['Portfolio Value', f"${report_data['portfolio_value']:,.2f}"],
                ['Total Cost', f"${report_data['total_cost']:,.2f}"],
                ['Total Return', f"${report_data['total_return']:,.2f}"],
                ['Return Percentage', f"{report_data['return_percentage']:.2f}%"],
                ['Number of Holdings', str(report_data['num_holdings'])]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Risk metrics
            risk_title = Paragraph("Risk Analysis", self.styles['Heading2'])
            story.append(risk_title)
            
            risk_data = [
                ['Risk Metric', 'Value'],
                ['Portfolio Volatility', f"{report_data['risk_metrics']['volatility']:.2f}%"],
                ['Sharpe Ratio', f"{report_data['risk_metrics']['sharpe_ratio']:.2f}"],
                ['Maximum Drawdown', f"{report_data['risk_metrics']['max_drawdown']:.2f}%"],
                ['Beta', f"{report_data['risk_metrics']['beta']:.2f}"]
            ]
            
            risk_table = Table(risk_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(risk_table)
            story.append(Spacer(1, 12))
            
            # Holdings detail
            holdings_title = Paragraph("Holdings Detail", self.styles['Heading2'])
            story.append(holdings_title)
            
            holdings_data = [['Symbol', 'Shares', 'Purchase Price', 'Current Price', 'Current Value', 'Gain/Loss', 'Gain/Loss %']]
            
            for holding in report_data['holdings']:
                holdings_data.append([
                    holding['symbol'],
                    f"{holding['shares']:.2f}",
                    f"${holding['purchase_price']:.2f}",
                    f"${holding['current_price']:.2f}",
                    f"${holding['current_value']:.2f}",
                    f"${holding['gain_loss']:.2f}",
                    f"{holding['gain_loss_pct']:.2f}%"
                ])
            
            holdings_table = Table(holdings_data)
            holdings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(holdings_table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return buffer.getvalue()
        
        except Exception as e:
            st.error(f"Error creating PDF report: {str(e)}")
            return None
    
    def create_excel_report(self, report_data):
        """Create an Excel report from report data"""
        try:
            if not report_data:
                return None
            
            # Create Excel buffer
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Portfolio Value', 'Total Cost', 'Total Return', 'Return Percentage', 'Number of Holdings'],
                    'Value': [
                        f"${report_data['portfolio_value']:,.2f}",
                        f"${report_data['total_cost']:,.2f}",
                        f"${report_data['total_return']:,.2f}",
                        f"{report_data['return_percentage']:.2f}%",
                        str(report_data['num_holdings'])
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Portfolio Summary', index=False)
                
                # Risk metrics sheet
                risk_df = pd.DataFrame({
                    'Risk Metric': ['Portfolio Volatility', 'Sharpe Ratio', 'Maximum Drawdown', 'Beta'],
                    'Value': [
                        f"{report_data['risk_metrics']['volatility']:.2f}%",
                        f"{report_data['risk_metrics']['sharpe_ratio']:.2f}",
                        f"{report_data['risk_metrics']['max_drawdown']:.2f}%",
                        f"{report_data['risk_metrics']['beta']:.2f}"
                    ]
                })
                risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
                
                # Holdings sheet
                holdings_df = pd.DataFrame(report_data['holdings'])
                if not holdings_df.empty:
                    holdings_df.to_excel(writer, sheet_name='Holdings Detail', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
        
        except Exception as e:
            st.error(f"Error creating Excel report: {str(e)}")
            return None
    
    def generate_market_summary_report(self, data_fetcher):
        """Generate a market summary report"""
        try:
            # Get market indices
            indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
            index_names = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]
            
            indices_data = data_fetcher.get_market_indices(indices)
            
            # Get trending stocks
            trending_stocks = data_fetcher.get_trending_stocks(10)
            
            market_report = {
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'indices': [],
                'trending_stocks': trending_stocks
            }
            
            for index, name in zip(indices, index_names):
                if index in indices_data:
                    market_report['indices'].append({
                        'name': name,
                        'symbol': index,
                        'current_price': indices_data[index]['current_price'],
                        'change': indices_data[index]['change'],
                        'change_percent': indices_data[index]['change_percent']
                    })
            
            return market_report
        
        except Exception as e:
            st.error(f"Error generating market summary: {str(e)}")
            return None
    
    def create_chart_image(self, chart_data, chart_type="line"):
        """Create a chart image for reports"""
        try:
            if chart_type == "line":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data['x'],
                    y=chart_data['y'],
                    mode='lines',
                    name=chart_data.get('name', 'Data')
                ))
                fig.update_layout(
                    title=chart_data.get('title', 'Chart'),
                    xaxis_title=chart_data.get('x_title', 'X'),
                    yaxis_title=chart_data.get('y_title', 'Y')
                )
            
            elif chart_type == "pie":
                fig = px.pie(
                    values=chart_data['values'],
                    names=chart_data['names'],
                    title=chart_data.get('title', 'Pie Chart')
                )
            
            # Convert to image
            img_bytes = fig.to_image(format="png", width=800, height=600)
            
            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode()
            
            return img_base64
        
        except Exception as e:
            st.error(f"Error creating chart image: {str(e)}")
            return None
    
    def schedule_reports(self, frequency="daily"):
        """Schedule automatic report generation"""
        try:
            # This is a simplified implementation
            # In production, you would use a proper scheduler like APScheduler
            
            report_config = {
                'frequency': frequency,
                'last_generated': datetime.now(),
                'enabled': True
            }
            
            # Store in session state
            if 'scheduled_reports' not in st.session_state:
                st.session_state.scheduled_reports = []
            
            st.session_state.scheduled_reports.append(report_config)
            
            return True
        
        except Exception as e:
            st.error(f"Error scheduling reports: {str(e)}")
            return False
    
    def get_scheduled_reports(self):
        """Get list of scheduled reports"""
        try:
            if 'scheduled_reports' not in st.session_state:
                return []
            
            return st.session_state.scheduled_reports
        
        except Exception as e:
            st.error(f"Error getting scheduled reports: {str(e)}")
            return []