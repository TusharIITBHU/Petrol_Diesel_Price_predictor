from django.shortcuts import render;
import numpy as np 
import pandas as pd 
import pickle
import math
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model


def home(request):
    return render(request, "home.html")

def result(request):

    city=int(request.GET['n1'])
    fuel=int(request.GET['n2'])
    days=int(request.GET['n3'])

    df=pd.read_csv(r"./datasetandmodel/prices.csv")
    df=df.iloc[: :-1].reset_index(drop=True)

    if(fuel==1):
        model = load_model(r"./LSTM3.h5")
    elif(fuel==2):
        model = load_model(r"./LSTMDSL.h5")

    data=df.filter(['Delhi-petrol'])
    if(city==1):
        if(fuel==1):
            data=df.filter(['Delhi-petrol'])
        else:
            data=df.filter(['Delhi-diesel'])

    elif(city==2):
        if(fuel==1):
            data=df.filter(['Mumbai-petrol'])
        else:
            data=df.filter(['Mumbai-diesel'])

    elif(city==3):
        if(fuel==1):
            data=df.filter(['Chennai-petrol'])
        else:
            data=df.filter(['Chennai-diesel'])

    elif(city==4):
        if(fuel==1):
            data=df.filter(['Kolkata-petrol'])
        else:
            data=df.filter(['Kolkata-diesel'])

    dataset=data.values

    training_data_len=int(len(dataset)*0.9)

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)


    test_data=scaled_data[training_data_len- 100: , :]

    x_input=test_data[len(test_data)-100:].reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    from numpy import array

    lst_output=[]
    n_steps=100
    i=0
    while(i<days):
    
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
    
            yhat = model.predict(x_input)
    
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    day_new=np.arange(1,101)
    day_pred=np.arange(101,101+days)

    l1=scaler.inverse_transform(lst_output)  
    l1=np.ndarray.tolist(l1)

    mylist=[]

    for i in range(len(l1)):
        mylist.append(l1[i][0])

    for i in range(len(mylist)):
        mylist[i]=round(mylist[i],2)

    return render(request, "home.html", {"result":mylist})