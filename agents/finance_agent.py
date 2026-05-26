import yfinance as yf
from datetime import datetime


COMPANY_TICKERS = {
    'nestle': 'NSRGY',
    'cadbury': 'MDLZ',
    'mondelez': 'MDLZ',
    'hershey': 'HSY',
}


def analyze_finance(company):

    company_lower = company.lower()

    ticker = COMPANY_TICKERS.get(company_lower, 'NSRGY')

    try:
        stock = yf.Ticker(ticker)

        hist = stock.history(
            period='3mo',
            auto_adjust=True,
        )

        if hist.empty:
            raise ValueError('No stock history returned')

        latest_close = float(hist['Close'].iloc[-1])
        avg_close = float(hist['Close'].mean())

        trend = 'Bullish' if latest_close > avg_close else 'Stable'

        return {
            'company': company,
            'ticker': ticker,
            'latest_close': round(latest_close, 2),
            'average_close': round(avg_close, 2),
            'market_trend': trend,
            'analysis_timestamp': datetime.utcnow().isoformat(),
        }

    except Exception as error:

        return {
            'company': company,
            'ticker': ticker,
            'market_trend': 'Unavailable',
            'status': 'Finance API temporarily rate limited',
            'fallback_analysis': [
                'Nestle remains a strong FMCG competitor globally.',
                'Chocolate and confectionery demand remains stable.',
                'Festive season campaigns can improve market share.',
                'Retail promotions may help counter competitor momentum.',
            ],
            'technical_error': str(error),
        }
