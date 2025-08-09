import pandas as pd
import numpy as np
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
    def __init__(
        self, T: int, dT: float, S0: float, log_returns: list  # T is in years
    ) -> None:
        self.T = T
        self.dT = dT
        self.N = int(self.T / self.dT)
        self.S0 = S0
        self.drift = np.mean(log_returns)
        self.sigma = np.std(log_returns, ddof=1)

    def geometric_brownian_motion(self, num_simulations=100) -> None:
        self.num_simulations = st.number_input("Please input the number of paths you want:", min_value=0)
        self.simulated_prices = np.full(shape=(num_simulations, 1), fill_value=self.S0)
        for step in range(self.N):
            step_prices = self.simulated_prices[:, step] * np.exp(
                (self.drift - self.sigma**2 / 2) * self.dT
                + self.sigma * self.dT**0.5 * np.random.normal(size=(num_simulations,))
            )
            step_prices = step_prices.reshape(-1, 1)
            self.simulated_prices = np.hstack((self.simulated_prices, step_prices))


def is_valid_ticker(ticker):
    data = yf.Ticker(ticker).history(period="1d")
    return not data.empty


def show_plot(simulation_object):
    
    t = np.linspace(0, simulation_object.T, simulation_object.simulated_prices.shape[1])
    for i in range(simulation_object.num_simulations):
        plt.plot(t, simulation_object.simulated_prices[i, :], alpha=0.2)
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.title("Monte Carlo Simulated Paths")
    st.pyplot(plt)


def main() -> None:
    st.title("Ticker Input")
    ticker = st.text_input("Enter a stock ticker symbol")
    if ticker:
        if is_valid_ticker(ticker):
            st.success("Ticker is go!")
            financial_data_period = st.radio(
                "Select historical data to use:",
                (
                    "1d",
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
            T = st.number_input("Enter number of years to simulate:", min_value=1)
            time_unit = st.radio(
                "Select time unit for intervals:",
                ("Seconds", "Minutes", "Hours", "Days"),
            )
            amount = st.number_input(
                f"Enter amount of {time_unit.lower()} for intervals:", min_value=1
            )
            if amount == 1:
                st.write(
                    f"You selected {amount} {time_unit.lower().removesuffix("s")}."
                )
            else:
                st.write(f"You selected {amount} {time_unit.lower()}.")
            fractions_of_year_multipliers = {
                "Seconds": 1 / (252 * 6.5 * 3600),
                "Minutes": 1 / (252 * 6.5 * 60),
                "Hours": 1 / (252 * 6.5),
                "Days": 1 / 252,
            }
            simulation = MonteCarlo(
                T=T,
                dT=amount * fractions_of_year_multipliers.get(time_unit),
                S0=stock.data["Close"].iloc[-1],
                log_returns=stock.log_returns,
            )
            simulation.geometric_brownian_motion()
            show_plot(simulation)
        else:
            st.error("Please input a valid ticker value")


main()
