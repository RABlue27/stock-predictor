from pandas.core.algorithms import value_counts
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime
import yfinance as yf
from datetime import datetime

def predict(data, company, prediction_days): 
    model = keras.models.load_model(data)

    test_start = dt.datetime(2021,1,2)
    test_end = dt.datetime.now()

    
    data = web.DataReader(company, 'yahoo', test_start, test_end)

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))


    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs) 

    real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction=model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    prediction=model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    
    return(prediction)


def train(company, epochs, batch): 

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2021, 1, 1)

    data = web.DataReader(company, 'yahoo', start, end)

    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days = 45



    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=85, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=85, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=85))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) #predict closing value

    model.compile(optimizer = 'adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs = epochs, batch_size = batch)

    #Test 

    test_start = dt.datetime(2021,1,2)
    test_end = dt.datetime.now()

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(predicted_prices)


    #plot 

    plt.plot(actual_prices, color="black", label=f"Actual Price {company}")
    plt.plot(prediction_prices, color="green", label = f"Predicted Price {company}")
    plt.title(f"{company} Share Price")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Share price")
    plt.show()

    save_location = input("Name the folder you'd like to create and save the model to: ")
    model.save(save_format="h5", filepath=save_location)


company_ticker = input("Enter the ticker of the stock: ")
company_ticker = company_ticker.upper()
predict_data_bool = input("Train or predict? ") #Probably dont use different training data than the ticker you're using 

if (predict_data_bool.lower() == "predict"):
    root = tk.Tk()
    tk.Tk().withdraw()
    data = filedialog.askopenfilename()
    root.destroy()
    predict_price = predict(data, company_ticker, 45)
    print("The price of the stock next close is predicted to be: ", round(float(predict_price), 2))
else:
    epochs = int(input("Enter epochs: "))
    batch = int(input("Enter batch size: "))
    train(company_ticker, epochs, batch)

