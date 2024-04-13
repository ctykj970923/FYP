from setup import *
import globalVariable

def result(model, initialDataFrame, Xtrain, scaler):
    copyDataFrame = initialDataFrame.copy()
    
    if(globalVariable.default_mode=='LSTM'):
        model.load_weights("model/lstm.h5")  
    else:
        model.load_weights("model/lstm_transformer.h5")  

    predicted_future_price = model.predict(Xtrain[:-1])
    predicted_future_price = np.squeeze(scaler.inverse_transform(predicted_future_price))
    print(predicted_future_price[-30:])
    copyDataFrame["close"] = scaler.inverse_transform(
        np.expand_dims(copyDataFrame["close"], axis=1)
    )
    copyDataFrame = copyDataFrame[-(len(copyDataFrame) - globalVariable.timeFrame) :]
    copyDataFrame[f"predicted_close"] = predicted_future_price
    
    #plot graph
    plt.style.use(style="ggplot")
    plt.figure(figsize=(16, 10))
    plt.plot(copyDataFrame["close"][-160:].head(160 - globalVariable.length))
    plt.plot(copyDataFrame["predicted_close"][-160:], linewidth=1, linestyle="dashed")
    plt.xlabel("days")
    plt.ylabel("price")
    plt.legend(
        [
            f"Actual price for {globalVariable.STOCK}",
            f"Predicted price for {globalVariable.STOCK}",
        ]
    )
    
    save_path = "stock/chart/"+globalVariable.STOCK+".png"  
    plt.savefig(save_path)
    
    image_path = "chart/"+globalVariable.STOCK+".png"
    
    return image_path