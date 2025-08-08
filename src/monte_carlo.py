import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st


class Stock():
    def __init__(self, ticker="GC=F", period="max"):
        self.data = yf.download(tickers=ticker, period=period, auto_adjust=True)
    
    def average_log_returns_and_volatility(self):
        log_returns = []
        previous_close = None
        for index, row in self.data.iterrows():
            if previous_close is not None: 
                log_returns.append(np.log(row["Close"] / previous_close))
            previous_close = row["Close"]
        average_log_returns = np.mean(log_returns)
        volatility = np.std(log_returns, ddof=1)
        return average_log_returns, volatility

            


gold = Stock()
print(gold.average_log_returns_and_volatility())



    
