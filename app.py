# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:12:37 2022

@author: Kunal
"""

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import codecs
import streamlit.components.v1 as components
#from streamlit_lottie import st_lottie
import json
import requests
from  PIL import Image
from darts import TimeSeries
from darts.models import (Theta,ExponentialSmoothing,AutoARIMA,Prophet)
from darts.metrics import mape
import matplotlib.pyplot as plt


##@st.experimental_singleton()
## Define Functions to call lottie Animations
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

st.header("Demand Prediction Application")
st.caption("a web based prototype demand prediction application")
lottie_hello=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_7c1e8erd.json")
st.sidebar.title(body="Prediction Application")
with st.sidebar:
    st_lottie(lottie_hello,speed=1,reverse=True,loop=True,quality="High",height=200,width=300,key=None)


#Add a logo (optional) in the sidebar
logo = Image.open(r'image3.jpg')
#st.sidebar.image(logo,  width=320)
linkedIN=("https://www.linkedin.com/in/aadityahamine")
st.sidebar.caption("Ideal for : Demand Prediction on TimeSeries")
#st.sidebar.caption("Library Used - Darts")
st.sidebar.caption("Designed by : Aaditya Hamine")
st.sidebar.caption("LinkedIN Details:")
st.sidebar.caption(linkedIN)
menu=["Theta Forecasting","Exponential Smoothing","FbProphet","ARIMA"]
choice=st.radio(label='Select a forecasting technique to predict your demands',options=menu)
#st.button("Give a try with our example dataset")
fh=st.slider('How  many days of data you want to predict?',value=10,min_value=10,max_value=180,step=10)


## Read Data - Our Example Dataset
df=pd.read_csv('data.csv')
series=TimeSeries.from_dataframe(df,'Date','Total')
##train test split
train=int(0.8*len(df['Total']))
test=int(len(df['Total']))-train
## training & validation data
train2,val=series.split_before(train)
with st.expander("Click here to see our sample dataset"):
    st.dataframe(df)

## List of Models
if st.button("Click here to start prediction with our sample dataset"):
    if choice=='Theta Forecasting':
        model=Theta()
        model.fit(train2)
        forecast=model.predict(fh)
        historical=model.historical_forecasts(series=train2,start=0.1,forecast_horizon=fh)
        plt.style.use('seaborn-dark')
        fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
        train2.plot(label='Actual Values')
        historical.plot(label='Predicted Values')
        forecast.plot(label='Forecast')
        error=mape(val,forecast)
        Accuracy=100-error

        plt.legend()
        plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
        st.pyplot(fig,figsize=(8,8))
        
    if choice=='Exponential Smoothing':
        model=ExponentialSmoothing()
        model.fit(train2)
        forecast=model.predict(fh)
        historical=model.historical_forecasts(series=train2,forecast_horizon=fh)
        plt.style.use('seaborn-dark')
        fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
        train2.plot(label='Actual Values')
        historical.plot(label='Predicted Values')
        forecast.plot(label='Forecast')
        error=mape(val,forecast)
        Accuracy=100-error

        plt.legend()
        plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
        st.pyplot(fig,figsize=(8,8))
    
    if choice=='FbProphet':
        model=Prophet()
        model.fit(train2)
        forecast=model.predict(fh)
        historical=model.historical_forecasts(series=train2,forecast_horizon=fh)
        plt.style.use('seaborn-dark')
        fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
        train2.plot(label='Actual Values')
        historical.plot(label='Predicted Values')
        forecast.plot(label='Forecast')
        error=mape(val,forecast)
        Accuracy=100-error

        plt.legend()
        plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
        st.pyplot(fig,figsize=(8,8))  

    if choice=='ARIMA':
        model=AutoARIMA()
        model.fit(train2)
        forecast=model.predict(fh)
        historical=model.historical_forecasts(series=train2,forecast_horizon=fh)
        plt.style.use('seaborn-dark')
        fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
        train2.plot(label='Actual Values')
        historical.plot(label='Predicted Values')
        forecast.plot(label='Forecast')
        error=mape(val,forecast)
        Accuracy=100-error

        plt.legend()
        plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
        st.pyplot(fig,figsize=(8,8))
  
        
else:
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.subheader("Go a step further , try with your own CSV file")
    df=st.file_uploader("Want to try this out with your own data ? - upload a csv file here",type=['csv']) 
    st.caption("CSV file should have first column named as Date. Target figures should be in next consecutive column labelled as 'Total'")
    st.caption("Data format for Date Column can be either mm/dd/yyyy or dd/mm/yyy")
    if df is not None:
        df=pd.read_csv(df)
        df['Date']=pd.to_datetime(df['Date'])
        series2=TimeSeries.from_dataframe(df,'Date','Total')
        #series2=TimeSeries.from_dataframe(df,'Date','Total')
        train2=int(0.8*len(df['Total']))
        test2=int(len(df['Total']))-train
        train_data2,val=series2.split_before(train2)

        
        if choice=='Theta Forecasting':
            model=Theta()
            model.fit(train_data2)
            forecast=model.predict(fh)
            historical=model.historical_forecasts(series=train__data2,forecast_horizon=fh)
            plt.style.use('seaborn-dark')
            fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
            train2.plot(label='Actual Values')
            historical.plot(label='Predicted Values')
            forecast.plot(label='Forecast')
            error=mape(val,forecast)
            Accuracy=100-error

            plt.legend()
            plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
            st.pyplot(fig,figsize=(8,8))
            
        if choice=='Exponential Smoothing':
            model=ExponentialSmoothing()
            model.fit(train_data2)
            forecast=model.predict(fh)
            historical=model.historical_forecasts(series=train_data2,forecast_horizon=fh)
            plt.style.use('seaborn-dark')
            fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
            train2.plot(label='Actual Values')
            historical.plot(label='Predicted Values')
            forecast.plot(label='Forecast')
            error=mape(val,forecast)
            Accuracy=100-error

            plt.legend()
            plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
            st.pyplot(fig,figsize=(8,8))        

        if choice=='FbProphet':
            model=Prophet()
            model.fit(train_data2)
            forecast=model.predict(fh)
            historical=model.historical_forecasts(series=train_data2,forecast_horizon=fh)
            plt.style.use('seaborn-dark')
            fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
            train2.plot(label='Actual Values')
            historical.plot(label='Predicted Values')
            forecast.plot(label='Forecast')
            error=mape(val,forecast)
            Accuracy=100-error

            plt.legend()
            plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
            st.pyplot(fig,figsize=(8,8))
      
        if choice=='ARIMA':
            model=AutoARIMA()
            model.fit(train_data2)
            forecast=model.predict(fh)
            historical=model.historical_forecasts(series=train_data2,forecast_horizon=fh)
            plt.style.use('seaborn-dark')
            fig=plt.figure(figsize=(10,8),facecolor='white',edgecolor=('white'))
            train2.plot(label='Actual Values')
            historical.plot(label='Predicted Values')
            forecast.plot(label='Forecast')
            error=mape(val,forecast)
            Accuracy=100-error

            plt.legend()
            plt.title('Prediction Accuracy   : {:.2f}%'.format(Accuracy))
            st.pyplot(fig,figsize=(8,8))



