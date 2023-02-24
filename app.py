# Import libraries
from http.client import RemoteDisconnected
from urllib.request import urlopen, Request
from pandas_datareader._utils import RemoteDataError
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import nltk
# NLTK VADER for sentiment analysis

nltk.downloader.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
##
import numpy as np 

import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
#from urllib3 import *
#import nltk




# for extracting data from finviz
finviz_url = 'https://finviz.com/quote.ashx?t='
#2010-01-01
#2019-12-31
start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend-Price and Sentiment Prediction')
while True:
    try:
        user_input = st.text_input('Enter Stock Ticker','AAPL')
        df = data.DataReader(user_input, 'yahoo', start, end)
        break
        #df.head()
    except RemoteDataError:
        st.write("Oops!  That was no valid ticker  Try again...")

#describing data 
st.subheader('Data from 2010-2019')
st.write(df.describe())

#visualizations 
st.subheader('Closing Price vs Time chart ')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#splitting data into training and testing  

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training) # this will return an array which we are storing in a variable


#load my model 
model = load_model('keras_model.h5')

#testing part 
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

 #defining x_test and y_test 
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
st.subheader('predicted Prices')
st.write(y_predicted)
st.subheader('Original Prices')
st.write(y_test)
#final visualization
st.subheader('predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='original price')
plt.plot(y_predicted, 'r', label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#sentiment analysis 

#get html headlinestables from finviz
def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    return news_table

ticker = user_input
news_table = get_news(ticker)

#parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() 
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])        
        # Set column names
        columns = ['date', 'time', 'headline']
        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)        
        # Create a pandas datetime object from the strings in 'date' and 'time' column
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        
    return parsed_news_df
        
parsed_news_df = parse_news(news_table)
st.subheader('Parsed News Data')
st.write(parsed_news_df.head())
#Score News Sentiment and Save Results into DataFrame
def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')        
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')    
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)          
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    return parsed_and_scored_news

parsed_and_scored_news = score_news(parsed_news_df)
st.subheader('Scored News Sentiment')
st.write(parsed_and_scored_news.head())

#Resample Sentiment by Hour and Date and Use Plotly to Plot It
st.subheader('Hourly Sentiment Scores')
def plot_hourly_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('H').mean()

    # Plot a bar chart with plotly 
    fig4 = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
    st.plotly_chart(fig4)
    
plot_hourly_sentiment(parsed_and_scored_news, ticker)

st.subheader('Daily Sentiment Scores')
def plot_daily_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('D').mean()

    # Plot a bar chart with plotly
    fig5 = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
    st.plotly_chart(fig5)
    
plot_daily_sentiment(parsed_and_scored_news, ticker)




