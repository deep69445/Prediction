import pandas as pd
import requests

def fetch_stock_data(symbol: str, api_key: str, interval: str = "60min") -> pd.DataFrame:
    """
    Fetch intraday stock data from Alpha Vantage and clean it.

    Args:
        symbol (str): Stock ticker symbol.
        api_key (str): Your Alpha Vantage API key.
        interval (str): Time interval between data points.

    Returns:
        pd.DataFrame: Cleaned stock data.
    """
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}&interval={interval}&outputsize=full&apikey={api_key}"
    )
    
    response = requests.get(url)
    data = response.json()
    
    # Extract data dictionary
    time_series_key = f"Time Series ({interval})"
    if time_series_key not in data:
        raise Exception("Invalid API response or API limit reached.")

    df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
    
    # Rename columns for clarity
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    # Reset index and format datetime
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Datetime'}, inplace=True)
    
    # Split date and time
    df['Date'] = df['Datetime'].dt.date
    df['Time'] = df['Datetime'].dt.time

    # Convert all numeric columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df