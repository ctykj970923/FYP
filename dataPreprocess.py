from setup import *
import globalVariable

#Scaler
def Scaler(initialDataFrame):
    desired_min = 0
    desired_max = 1
    scaler = MinMaxScaler(feature_range=(desired_min, desired_max))
    initialDataFrame["close"] = scaler.fit_transform(
        np.expand_dims(initialDataFrame["close"].values, axis=1)
    )
    return initialDataFrame, scaler


def dataPreprocess(dataframe):
    sequence_dataset = []
    Xtrain, Ytrain = [], []
    dataframe["future"] = dataframe["close"].shift(-1)
    newDataFrame = dataframe.copy()
    sequences = deque(maxlen=globalVariable.timeFrame)
    last_sequence = np.array([dataframe[["close"]].tail(globalVariable.timeFrame)])
    dataframe.dropna(inplace=True)
    close_and_date = dataframe[["close"] + ["date"]].values
    future_price = dataframe["future"].values
    
    #3 day price map to next day price
    for price, futurePrice in zip(close_and_date, future_price):
        sequences.append(price)
        if len(sequences) == globalVariable.timeFrame:
            sequence_dataset.append([np.array(sequences), futurePrice])
            
    #Split into Xtrain, Ytrain       
    for x, y in sequence_dataset:
        x = x[:, : len(["close"])].astype(np.float32)
        Xtrain.append(x)
        Ytrain.append(y)
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    
    ori_Xtrain = Xtrain
   
    return newDataFrame, Xtrain, Ytrain, last_sequence, ori_Xtrain
    
