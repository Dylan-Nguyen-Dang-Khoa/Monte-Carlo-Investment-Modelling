import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st


class Stock():
    def __init__(self, ticker="GC=F", start_date=pd.to_datetime("today").date() - pd.Timedelta(days=1), end_date=pd.to_datetime("today").date()):
        print(start_date, end_date)
        self.data = yf.download(ticker, start_date, end_date)
    
