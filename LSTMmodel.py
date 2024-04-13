from setup import *
import globalVariable


#model LSTM
def GetLSTMModel(x, y):
    model = Sequential()
    model.add(Bidirectional(LSTM(1024,return_sequences=True,input_shape=(globalVariable.timeFrame, len(["close"])))))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(1024, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(1024, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(1024, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(1024, return_sequences=False)))
    model.add(Dense(1))

    BATCH_SIZE = 128
    EPOCHS = 30

    model.compile(loss="mean_squared_error", optimizer="adam")
    
    kf = KFold(n_splits=7, shuffle=True, random_state=42)  # 7-fold cross-validation
    
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
    checkpoint_callback = ModelCheckpoint(
        "model/lstm.h5",
        save_weight_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback],
        verbose=1,
    )

    model.summary()

    return model
