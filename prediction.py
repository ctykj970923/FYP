from setup import *
from dataPreprocess import *
from LSTMmodel import *
from LSTM_Trans_model import *
import globalVariable


def prediction(initialDataFrame, scaler):
    predictions = []
    initialDataFrame, Xtrain, Ytrain, last_sequence,ori_Xtrain = dataPreprocess(initialDataFrame)
    print(len(Xtrain))
    print(len(Ytrain))
    print("********************************************")
    if(globalVariable.default_mode=='LSTM'):
        print("Model is LSTM")
        model = GetLSTMModel(Xtrain, Ytrain)
    else:  
        print("Model is Transformer")      
        model = GetTransModel(Xtrain, Ytrain)
        
    ori_Xtrain = np.append(ori_Xtrain,np.expand_dims(last_sequence[0][-globalVariable.timeFrame :], axis=0),axis=0,)
    initialDataFrame.drop("future", axis=1, inplace=True)
    
    for day in globalVariable.futureDate:
        prediction = model.predict(
            np.expand_dims(last_sequence[0][-globalVariable.timeFrame :], axis=0)
        )
        last_sequence = np.append(
            last_sequence, np.expand_dims(prediction, axis=1), axis=1
        )
   
        ori_Xtrain = np.append(ori_Xtrain,np.expand_dims(last_sequence[0][-globalVariable.timeFrame :], axis=0),axis=0,)
        predicted_future_price = scaler.inverse_transform(prediction)[0][0]
        predictions.append(round(float(predicted_future_price), 2))
        date_next = dt.date.today() + dt.timedelta(days=int(day))
        initialDataFrame.loc[date_next] = [prediction.item(), date_next]

    return initialDataFrame, ori_Xtrain, model
