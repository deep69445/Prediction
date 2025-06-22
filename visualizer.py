import plotly.graph_objs as go
import pandas as pd

def plot_recent_prices(df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Datetime'],
        y=df['Close'],
        mode='lines+markers',
        name='Close Price'
    ))
    fig.update_layout(
        title=f"{symbol} - Recent Closing Prices",
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=400
    )
    return fig

def plot_with_moving_averages(df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'], name='Close Price'))
    
    if 'MA_3' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MA_3'], name='MA 3'))
    if 'MA_7' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MA_7'], name='MA 7'))

    fig.update_layout(
        title=f"{symbol} - Moving Averages",
        xaxis_title="Time",
        yaxis_title="Price",
        template='plotly_dark',
        height=400
    )
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))

    fig.update_layout(
        title="Actual vs Predicted Close Prices",
        xaxis_title="Index",
        yaxis_title="Price",
        template='plotly_dark',
        height=400
    )
    return fig