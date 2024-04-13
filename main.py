from setup import *
from stockData import *
from LSTMmodel import *
from prediction import *
from visualization import *
import globalVariable

def showResult():
    initialDataFrame = getStockData(globalVariable.STOCK)
    initialDataFrame, scaler = Scaler(initialDataFrame)
    initialDataFrame, Xtrain, model = prediction(initialDataFrame, scaler)
    chart = result(model, initialDataFrame, Xtrain, scaler)
    return chart

