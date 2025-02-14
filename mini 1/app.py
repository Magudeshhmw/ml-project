from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # ARIMA import

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        company1 = request.form["company1"]
        company2 = request.form["company2"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Convert input strings to datetime objects
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return render_template("index.html", error_message="Error: Date format is incorrect. Please use YYYY-MM-DD.")

        # Download historical data for the given companies
        data1 = yf.download(company1, start=start_date, end=end_date)
        data2 = yf.download(company2, start=start_date, end=end_date)

        if data1.empty or data2.empty:
            return render_template("index.html", error_message="No data available for this symbol or date range.")

        # Calculate the percentage difference in the last closing price between the two companies
        last_price1 = data1['Close'].iloc[-1]
        last_price2 = data2['Close'].iloc[-1]
        price_diff_pct = ((last_price2 - last_price1) / last_price1) * 100  # Percentage difference

        # Get current share prices for both companies
        company1_info = yf.Ticker(company1).history(period="1d")
        company2_info = yf.Ticker(company2).history(period="1d")
        current_price1 = company1_info['Close'].iloc[-1]
        current_price2 = company2_info['Close'].iloc[-1]

        # Suggest which company is the best based on the most recent price
        best_company = company1 if current_price1 > current_price2 else company2

        # ARIMA Model Forecasting for Company 1
        df1 = data1[['Close']]
        train_size1 = int(len(df1) * 0.8)
        train1, test1 = df1.iloc[:train_size1], df1.iloc[train_size1:]
        model1 = ARIMA(train1['Close'], order=(1, 1, 2))  # (p, d, q)
        model_fit1 = model1.fit()

        # ARIMA Model Forecasting for Company 2
        df2 = data2[['Close']]
        train_size2 = int(len(df2) * 0.8)
        train2, test2 = df2.iloc[:train_size2], df2.iloc[train_size2:]
        model2 = ARIMA(train2['Close'], order=(1, 1, 2))  # (p, d, q)
        model_fit2 = model2.fit()

        # Forecast for Company 1
        forecast1 = model_fit1.forecast(steps=len(test1))

        # Forecast for Company 2
        forecast2 = model_fit2.forecast(steps=len(test2))

        # Plotting combined forecast for both companies
        plt.figure(figsize=(14, 7))
        plt.plot(train1.index, train1['Close'], label=f'{company1} Train', color='#203147')
        plt.plot(test1.index, test1['Close'], label=f'{company1} Test', color='#01ef63')
        plt.plot(test1.index, forecast1, label=f'{company1} Forecast', color='orange')

        plt.plot(train2.index, train2['Close'], label=f'{company2} Train', color='#0044FF')
        plt.plot(test2.index, test2['Close'], label=f'{company2} Test', color='#FF6600')
        plt.plot(test2.index, forecast2, label=f'{company2} Forecast', color='purple')

        plt.title(f'{company1} vs {company2} Close Price Forecast Comparison')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Save combined forecast plot to a string
        img_combined = io.BytesIO()
        plt.savefig(img_combined, format='png')
        img_combined.seek(0)
        combined_forecast_plot_url = base64.b64encode(img_combined.getvalue()).decode('utf8')

        # Plotting forecast for Company 1
        plt.figure(figsize=(14, 7))
        plt.plot(train1.index, train1['Close'], label=f'{company1} Train', color='#203147')
        plt.plot(test1.index, test1['Close'], label=f'{company1} Test', color='#01ef63')
        plt.plot(test1.index, forecast1, label=f'{company1} Forecast', color='orange')
        plt.title(f'{company1} Close Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Save Company 1 forecast plot to a string
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        company1_forecast_plot_url = base64.b64encode(img1.getvalue()).decode('utf8')

        # Plotting forecast for Company 2
        plt.figure(figsize=(14, 7))
        plt.plot(train2.index, train2['Close'], label=f'{company2} Train', color='#0044FF')
        plt.plot(test2.index, test2['Close'], label=f'{company2} Test', color='#FF6600')
        plt.plot(test2.index, forecast2, label=f'{company2} Forecast', color='purple')
        plt.title(f'{company2} Close Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Save Company 2 forecast plot to a string
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        company2_forecast_plot_url = base64.b64encode(img2.getvalue()).decode('utf8')

        # Return the plots, price difference, and the best company suggestion
        return render_template(
            "index.html",
            combined_forecast_plot_url=combined_forecast_plot_url,
            company1_forecast_plot_url=company1_forecast_plot_url,
            company2_forecast_plot_url=company2_forecast_plot_url,
            price_diff_pct=price_diff_pct,
            company1=company1,
            company2=company2,
            current_price1=current_price1,
            current_price2=current_price2,
            best_company=best_company
        )

    return render_template("index.html", combined_forecast_plot_url=None)

if __name__ == "__main__":
    app.run(debug=True)
