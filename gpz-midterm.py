#This flask asks users to select a coin, investment value, and investment date and will return the current worth
#This flask will also return a plot of the coin returns from the initial investment date to today

#I import all the packages and libraries needed for this app..
from flask import Flask, render_template, request, url_for, redirect
import time
import datetime
from datetime import datetime
import calendar
import requests
from money import Money
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np
import seaborn as sns

import io
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64

# packages being set up
import pandas as pd
import numpy as np
import matplotlib
from cryptocmd import CmcScraper
from scipy import stats
import statsmodels.api as sm
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#I set the app name
app = Flask(__name__)


#Here I create an entry page that allows the user to identify their investment level
@app.route('/')
def start_page():
    return render_template('index.html')

@app.route('/latestprice', methods=['POST'])
def latest_result():
    # Create the variable for the chosen cryptocurrency
    coin = request.form.get('ChooseCoin')

    # initialise scraper without passing time interval
    scraper = CmcScraper(coin)

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    # Create the latest of the table
    current_results = df.head(5)

    # The return below shows the latest results in a table format.
    return render_template('currentvalues.html', coin=coin, current_results=current_results.to_html())

@app.route('/predictions', methods=['POST'])
def crypto_predict():
    # Create the variable for the chosen cryptocurrency
    coin = request.form.get('ChooseCoin')

    # initialise scraper without passing time interval
    scraper = CmcScraper(coin)

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    # Format the dates in the dataframe
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year

    # Update the format of the Volume column
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Provide the log returns using the close price in the dataframe
    # Use the value to provide a volatility (rolling standard deviation function).
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # We're using a window of 365 trading days as the cryptocurrency market doesn't open/close.
    df['volatility'] = df['log_ret'].rolling(365).std() * np.sqrt(365)
    df['dv'] = (df['Close'] * df['Volume'] / 1e6)[1:]
    df['lret'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    df['daily_illiq'] = np.abs(df['lret']) / df['dv']

    # Creating a variable to display the adjusted results
    df_display = df.head(5)

    # Aligning the date to the index in the dataframe
    df.index = df.Date
    df = df.resample('D').mean()  # Resampling to daily frequency for the cryptocurrency
    df_month = df.resample('M').mean()  # Resampling to monthly frequency for cryptocurrency

    # Creating the Box-Cox Transformations
    df_month['high_box'], lmbda = stats.boxcox(df_month.High)

    # Initial approximation of parameters
    Qs = range(0, 2)
    qs = range(0, 3)
    Ps = range(0, 3)
    ps = range(0, 3)
    D = 1
    d = 1
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Model Selection for BTC
    results = []
    best_aic = float("inf")
    for param in parameters_list:
        try:
            model = sm.tsa.statespace.SARIMAX(df_month.high_box, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
        except ValueError:
            print('wrong parameters:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    # Best Models
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']

    # Creating a function for the Inverse Box-Cox Transformation
    def invboxcox(y, lmbda):
        if lmbda == 0:
            return (np.exp(y))
        else:
            return (np.exp(np.log(lmbda * y + 1) / lmbda))

    # Creating the Prediction for BTC
    df_month2 = df_month[['High']]
    date_list = [datetime(2018, 6, 30),
                 datetime(2018, 7, 31),
                 datetime(2018, 8, 31),
                 datetime(2018, 9, 30),
                 datetime(2018, 10, 31),
                 datetime(2018, 11, 30),
                 datetime(2018, 12, 31),
                 datetime(2019, 1, 31),
                 datetime(2019, 2, 28),
                 datetime(2019, 3, 31),
                 datetime(2019, 4, 30),
                 datetime(2019, 5, 31),
                 datetime(2019, 6, 30),
                 datetime(2019, 7, 31)]

    future = pd.DataFrame(index=date_list, columns=df_month.columns)
    df_month2 = pd.concat([df_month2, future])
    df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=75), lmbda)
    plt.figure(figsize=(15, 7))
    df_month2.High.plot()
    df_month2.forecast.plot(color='r', ls='--', label='predicted high')
    plt.legend()
    plt.title('Cryptocurrency Prediction, by months')
    plt.ylabel('mean USD')

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # The return below shows the latest results in a table format.
    return render_template('predictions.html', coin=coin, result=figdata_png.decode('utf8'), adjusted_results=df_display.to_html())

@app.errorhandler(500)
def second_error_page(e):
    return render_template('error_handling.html'), 500

@app.errorhandler(404)
def error_page(e):
    return render_template('error_handling.html'), 404

# Set port, host, and debug status for the flask app
if __name__ == "__main__":
    app.run(debug=True, port=9100)