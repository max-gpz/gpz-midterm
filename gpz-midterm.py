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
import base64
import io
import matplotlib
matplotlib.use('Agg')

# packages being set up
import pandas as pd
import numpy as np
import matplotlib
from cryptocmd import CmcScraper
from scipy import stats
import statsmodels.api as sm
from itertools import product
from datetime import datetime

#I set the app name
app = Flask(__name__)


#Here I create an entry page that allows the user to identify their investment level
@app.route('/')
def start_page():
    return render_template('index.html')

@app.route('/predictions', methods=['POST'])
def latest_result():
    coin = request.form.get('ChooseCoin')

    # initialise scraper without passing time interval
    scraper = CmcScraper(coin)

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    # Print the latest of the table
    current_results = df.head(5)

    #Here I combine the inputs with the comparision calculations to respond to the user
    return render_template('currentvalues.html', coin=coin, current_results=current_results)

@app.errorhandler(404)
def error_page(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def second_error_page(e):
    return render_template('404.html'), 500

# Set port, host, and debug status for the flask app
if __name__ == "__main__":
    app.run(debug=False, port=9100)