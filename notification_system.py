import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
from datetime import datetime
import json

class NotificationSystem:
    def __init__(self):
        self.smtp_server = None
        self.smtp_port = None
        self.email_username = None
        self.email_password = None
        self.setup_email_config()
    
    def setup_email_config(self):
        """Setup email configuration from environment or session state"""
        try:
            # Check if email configuration exists in session state
            if 'email_config' not in st.session_state:
                st.session_state.email_config = {
                    'smtp_server': '',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'enabled': False
                }
            
            config = st.session_state.email_config
            if config['enabled']:
                self.smtp_server = config['smtp_server']
                self.smtp_port = config['smtp_port']
                self.email_username = config['username']
                self.email_password = config['password']
        
        except Exception as e:
            st.error(f"Error setting up email config: {str(e)}")
    
    def configure_email_settings(self, smtp_server, smtp_port, username, password):
        """Configure email settings"""
        try:
            st.session_state.email_config = {
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'username': username,
                'password': password,
                'enabled': True
            }
            
            self.smtp_server = smtp_server
            self.smtp_port = smtp_port
            self.email_username = username
            self.email_password = password
            
            return True
        
        except Exception as e:
            st.error(f"Error configuring email settings: {str(e)}")
            return False
    
    def send_email(self, to_email, subject, body, attachment=None):
        """Send email notification"""
        try:
            if not self.smtp_server or not self.email_username:
                st.warning("Email configuration not set up. Please configure email settings first.")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachment if provided
            if attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment['data'])
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment["filename"]}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            server.send_message(msg)
            server.quit()
            
            return True
        
        except Exception as e:
            st.error(f"Error sending email: {str(e)}")
            return False
    
    def send_portfolio_alert(self, to_email, alert_message, portfolio_summary=None):
        """Send portfolio alert email"""
        try:
            subject = f"Portfolio Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            body = f"""
Financial Dashboard Alert

{alert_message}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            
            if portfolio_summary:
                body += f"""
Portfolio Summary:
- Total Value: ${portfolio_summary.get('total_value', 0):,.2f}
- Total Return: ${portfolio_summary.get('total_return', 0):,.2f}
- Return Percentage: {portfolio_summary.get('return_percentage', 0):.2f}%
- Number of Holdings: {portfolio_summary.get('num_holdings', 0)}
"""
            
            body += """
Please log into your financial dashboard for more details.

Best regards,
Financial Dashboard System
"""
            
            return self.send_email(to_email, subject, body)
        
        except Exception as e:
            st.error(f"Error sending portfolio alert: {str(e)}")
            return False
    
    def send_daily_report(self, to_email, report_data, pdf_report=None):
        """Send daily portfolio report"""
        try:
            subject = f"Daily Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
Daily Portfolio Report

Generated on: {report_data.get('generation_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

Portfolio Summary:
- Portfolio Value: ${report_data.get('portfolio_value', 0):,.2f}
- Total Cost: ${report_data.get('total_cost', 0):,.2f}
- Total Return: ${report_data.get('total_return', 0):,.2f}
- Return Percentage: {report_data.get('return_percentage', 0):.2f}%
- Number of Holdings: {report_data.get('num_holdings', 0)}

Risk Metrics:
- Portfolio Volatility: {report_data.get('risk_metrics', {}).get('volatility', 0):.2f}%
- Sharpe Ratio: {report_data.get('risk_metrics', {}).get('sharpe_ratio', 0):.2f}
- Maximum Drawdown: {report_data.get('risk_metrics', {}).get('max_drawdown', 0):.2f}%
- Beta: {report_data.get('risk_metrics', {}).get('beta', 1.0):.2f}

Please find the detailed report attached.

Best regards,
Financial Dashboard System
"""
            
            attachment = None
            if pdf_report:
                attachment = {
                    'data': pdf_report,
                    'filename': f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf"
                }
            
            return self.send_email(to_email, subject, body, attachment)
        
        except Exception as e:
            st.error(f"Error sending daily report: {str(e)}")
            return False
    
    def send_market_summary(self, to_email, market_data):
        """Send market summary email"""
        try:
            subject = f"Market Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
Market Summary Report

Generated on: {market_data.get('generation_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

Market Indices:
"""
            
            for index in market_data.get('indices', []):
                body += f"- {index['name']}: {index['current_price']:.2f} ({index['change_percent']:+.2f}%)\n"
            
            body += """
Trending Stocks:
"""
            
            for stock in market_data.get('trending_stocks', [])[:5]:
                body += f"- {stock['symbol']}: ${stock['price']:.2f} ({stock['change_percent']:+.2f}%)\n"
            
            body += """
Please log into your financial dashboard for more details.

Best regards,
Financial Dashboard System
"""
            
            return self.send_email(to_email, subject, body)
        
        except Exception as e:
            st.error(f"Error sending market summary: {str(e)}")
            return False
    
    def test_email_connection(self):
        """Test email connection"""
        try:
            if not self.smtp_server or not self.email_username:
                return False, "Email configuration not set up"
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            server.quit()
            
            return True, "Email connection successful"
        
        except Exception as e:
            return False, f"Email connection failed: {str(e)}"
    
    def get_notification_history(self):
        """Get notification history"""
        try:
            if 'notification_history' not in st.session_state:
                st.session_state.notification_history = []
            
            return st.session_state.notification_history
        
        except Exception as e:
            st.error(f"Error getting notification history: {str(e)}")
            return []
    
    def add_notification_to_history(self, notification_type, recipient, status):
        """Add notification to history"""
        try:
            if 'notification_history' not in st.session_state:
                st.session_state.notification_history = []
            
            notification = {
                'type': notification_type,
                'recipient': recipient,
                'status': status,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.session_state.notification_history.append(notification)
            
            # Keep only last 100 notifications
            if len(st.session_state.notification_history) > 100:
                st.session_state.notification_history = st.session_state.notification_history[-100:]
        
        except Exception as e:
            st.error(f"Error adding notification to history: {str(e)}")
    
    def setup_notification_preferences(self, preferences):
        """Setup notification preferences"""
        try:
            if 'notification_preferences' not in st.session_state:
                st.session_state.notification_preferences = {}
            
            st.session_state.notification_preferences.update(preferences)
            return True
        
        except Exception as e:
            st.error(f"Error setting notification preferences: {str(e)}")
            return False
    
    def get_notification_preferences(self):
        """Get notification preferences"""
        try:
            if 'notification_preferences' not in st.session_state:
                return {
                    'email_alerts': True,
                    'daily_reports': False,
                    'weekly_reports': True,
                    'threshold_alerts': True,
                    'market_summaries': False
                }
            
            return st.session_state.notification_preferences
        
        except Exception as e:
            st.error(f"Error getting notification preferences: {str(e)}")
            return {}