from data_loader import fetch_stock_data
from predictor import train_model, predict_next_day

symbol = "AAPL"
api_key = "UXS0VUB2WCZ5HTYH"

df = fetch_stock_data(symbol, api_key)
model, prediction, mae, rmse = train_model(df)
next_price = predict_next_day(model, df)

print(f"Predicted Close: {next_price:.2f}")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")