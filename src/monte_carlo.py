import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st


class Stock:
    def __init__(self, ticker="GC=F", period="3y", interval="1d"):
        self.data = yf.download(tickers=ticker, period=period, auto_adjust=True)
        self.calculate_metrics()

    def calculate_metrics(self) -> None:
        self.data["LogReturn"] = np.log(
            self.data["Close"] / self.data["Close"].shift(1)
        )
        log_returns = self.data["LogReturn"].dropna()
        self.average_log_returns = log_returns.mean()
        self.sigma = log_returns.std(ddof=1)


class MonteCarlo:
    def __init__(self, last_price, average_log_returns, sigma, T=1, N=252) -> None:
        self.N = N
        self.dT = T / N
        self.S0 = last_price
        self.drift = average_log_returns
        self.sigma = sigma

    def geometric_brownian_motion(self, num_simulations=100):
        self.simulated_prices = np.full(shape=(num_simulations, 1), fill_value=self.S0)
        for step in range(self.N):
            step_prices = self.simulated_prices[:, step] * np.exp(
                (self.drift - self.sigma**2 / 2) * self.dT
                + self.sigma * self.dT**0.5 * np.random.normal(size=(num_simulations,))
            )
            step_prices = step_prices.reshape(-1, 1)
            self.simulated_prices = np.hstack((self.simulated_prices, step_prices))


gold = Stock()
simulation = MonteCarlo(
    last_price=gold.data["Close"].iloc[-1],
    average_log_returns=gold.average_log_returns,
    sigma=gold.sigma,
)
simulation.geometric_brownian_motion()
print(simulation.simulated_prices[:5, -1])
