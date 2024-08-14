from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics


def home(request):

    return render(request, 'home.html')


def predict(request):

    return render(request, 'predict.html')

def result(request):
    dataset = pd.read_csv("USA_Housing.csv")
    dataset = dataset.drop(['Address'], axis=1)
    
    # split dataset
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    var1 = float(request.GET.get('n1', 0))
    var2 = float(request.GET.get('n2', 0))
    var3 = float(request.GET.get('n3', 0))
    var4 = float(request.GET.get('n4', 0))
    var5 = float(request.GET.get('n5', 0))
    
    pred = lr.predict(np.array([[var1, var2, var3, var4, var5]]))
    pred = round(pred[0], 2)  # Round to 2 decimal places if needed

    # Format the price with commas and decimals
    price = f"The predicted price is ${pred:,.2f}"
    
    return render(request, 'predict.html', {"price": price})