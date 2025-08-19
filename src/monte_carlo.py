import pandas as pd
import numpy as np
from numpy.typing import NDArray
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st


class Stock:
    def __init__(self, period: str, ticker: str) -> None:
        self.data = yf.download(
            tickers=ticker, period=period, interval="1d", auto_adjust=True
        )
        self.calculate_log_return()

    def calculate_log_return(self) -> None:
        self.data["LogReturn"] = np.log(
            self.data["Close"] / self.data["Close"].shift(1)
        )
        self.log_returns = self.data["LogReturn"].dropna()


class MonteCarlo:
    def __init__(self, S0: float, log_returns: list, T: int) -> None:  # T is in years
        self.dT = 1 / 252
        self.N = int(T / self.dT)
        self.S0 = S0
        self.log_returns = log_returns
        self.squared_log_returns = np.square(log_returns)
        self.drift = (np.mean(log_returns) - np.var(log_returns, ddof=1) / 2) * 252
        self.heston_model_init()

    def heston_model(self, num_simulations: int) -> None:
        self.num_simulations = num_simulations
        self.v_t = np.full(
            shape=(num_simulations,), fill_value=np.var(self.log_returns[-30:], ddof=1)
        )
        self.simulated_prices = np.full(shape=(num_simulations, 1), fill_value=self.S0)
        for step in range(self.N):
            Z = np.random.normal(size=self.simulated_prices[:, step].shape)
            step_prices = self.simulated_prices[:, step] * np.exp(
                self.drift * self.dT + np.sqrt(self.v_t) * np.sqrt(self.dT) * Z
            )
            step_prices = step_prices.reshape(-1, 1)
            self.simulated_prices = np.hstack((self.simulated_prices, step_prices))
            self.calculate_volatility()

    def calculate_xi(self) -> float:
        X = self.squared_log_returns[:-1]
        Y = self.squared_log_returns[1:]
        a, b = self.linear_regression(X=X, Y=Y)
        residuals = Y - (a + b * X)
        xi = np.std(residuals, ddof=1) / np.sqrt(self.dT)
        return xi

    def heston_model_init(self) -> None:
        X = self.squared_log_returns[:-1]
        Y = self.squared_log_returns[1:]
        a, b = self.linear_regression(X=X, Y=Y)
        self.theta = a / (1 - b)
        self.kappa = -np.log(b) / self.dT
        self.xi = self.calculate_xi()

    def calculate_volatility(self) -> None:
        Z = np.random.normal(size=self.num_simulations)
        self.v_t = (
            self.v_t
            + self.kappa * (self.theta - self.v_t) * self.dT
            + self.xi * np.sqrt(self.v_t * self.dT) * Z
        )

    def linear_regression(
        self, X: NDArray[np.float64], Y: NDArray[np.float64]
    ) -> tuple[float, float]:
        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        cov_matrix = np.cov(X, Y, ddof=1)
        var_x = cov_matrix[0, 0]
        cov_xy = cov_matrix[0, 1]
        a = cov_xy / var_x
        b = mean_Y - a * mean_X
        return float(a), float(b)

    def plot_monte_carlo(self):
        ...


def is_valid_ticker(ticker):
    data = yf.Ticker(ticker).history(period="1d")
    return not data.empty


def main() -> None:
    st.title("Ticker Input")
    ticker = st.text_input("Enter a stock ticker symbol")
    if ticker:
        if is_valid_ticker(ticker):
            st.success("Ticker is go!")
            financial_data_period = st.radio(
                "Select length of historical data to use:",
                (
                    "5d",
                    "1mo",
                    "3mo",
                    "6mo",
                    "1y",
                    "2y",
                    "5y",
                    "10y",
                    "ytd",
                    "max",
                ),
            )
            stock = Stock(ticker=ticker, period=financial_data_period)
            simulation_units = st.radio(
                "Select the unit for the length of the simulation:",
                ("Days", "Months", "Years"),
            )
            if simulation_units == "Days":
                days = st.number_input("Enter number of days to simulate:", min_value=1)
                T = days / 252
            elif simulation_units == "Months":
                months = st.number_input(
                    "Enter number of months to simulate:", min_value=1
                )
                T = months / 12
            else:
                T = st.number_input("Enter number of years to simulate:", min_value=1)
            simulation = MonteCarlo(
                T=T,
                S0=stock.data["Close"].iloc[-1],
                log_returns=stock.log_returns,
            )
            num_simulations = st.number_input(
                "Please input the number of paths you want:", min_value=1
            )
            simulation.heston_model(num_simulations=num_simulations)
        else:
            st.error("Please input a valid ticker value")


main()
