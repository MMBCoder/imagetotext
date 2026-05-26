import yfinance as yf


def analyze_finance(company):
    ticker = 'NSRGY'
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')

    latest_close = hist['Close'].iloc[-1]
    avg_close = hist['Close'].mean()

    return {
        'ticker': ticker,
        'latest_close': round(latest_close, 2),
        'average_close': round(avg_close, 2),
    }
