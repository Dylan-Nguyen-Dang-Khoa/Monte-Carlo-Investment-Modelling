import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st


class Stock:
    def __init__(self, ticker="GC=F", period="3y"):
        self.data = yf.download(tickers=ticker, period=period, auto_adjust=True)
        self.average_log_returns_and_volatility()

    def average_log_returns_and_volatility(self) -> None:
        log_returns = []
        previous_close = None
        for index, row in self.data.iterrows():
            if previous_close is not None:
                log_returns.append(np.log(row["Close"] / previous_close))
            previous_close = row["Close"]
        self.average_log_returns = np.mean(log_returns)
        self.sigma = np.std(log_returns, ddof=1)


class MonteCarlo:
    def __init__(self, last_price, average_log_returns, sigma, T=1, N=252) -> None:
        self.N = N
        self.dT = T / N
        self.S0 = last_price
        self.drift = average_log_returns - 0.5 * sigma**2
        self.scaled_sigma = sigma * N**0.5

    def geometric_brownian_motion(self, num_simulations=100):
        self.W_t = np.random.normal(size=(num_simulations, self.N))
        prices = np.full(shape=(num_simulations, 1), fill_value=self.S0)
        for step in range(self.N):
            step_prices = prices[:, step] * np.e ** (
                (self.drift - self.scaled_sigma**2 / 2) * self.dT
                + self.scaled_sigma
                * self.dT**0.5
                * np.random.normal(size=(num_simulations,))
            )
            step_prices = step_prices.reshape(-1, 1)
            prices = np.hstack((prices, step_prices))


gold = Stock()
simulation = MonteCarlo(
    last_price=gold.data["Close"].iloc[-1],
    average_log_returns=gold.average_log_returns,
    sigma=gold.sigma,
)
simulation.geometric_brownian_motion()
