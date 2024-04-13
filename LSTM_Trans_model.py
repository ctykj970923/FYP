from setup import *
import globalVariable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#model transformer
def GetTransModel(x, y):
    input_shape = (3, 1)
    
    def transformer_bidirectional_lstm_model(input_shape):
        inputs = layers.Input(shape=input_shape)

        lstm_output = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(inputs)
        lstm_output = layers.Dropout(0.1)(lstm_output)
        lstm_output = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(lstm_output)
        lstm_output = layers.Dropout(0.1)(lstm_output)
        lstm_output = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(lstm_output)
        lstm_output = layers.Dropout(0.1)(lstm_output)
        lstm_output = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(lstm_output)
        lstm_output = layers.Dropout(0.1)(lstm_output)

        position_embed = layers.Embedding(input_shape[0], input_shape[-1])(tf.range(input_shape[0]))

        x = lstm_output + position_embed
        
        #Block 1 - encoder (hide)
        # x = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        # x = layers.Dropout(0.1)(x)
        # x = layers.LayerNormalization(epsilon=1e-6)(x)
        # x = layers.Dense(units=32, activation="relu")(x) 
        # x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        #Block 2 - encoder
        x = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = layers.Dropout(0.1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(units=16, activation="relu")(x) 
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.GlobalAveragePooling1D()(x)

        outputs = layers.Dense(units=1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    model = transformer_bidirectional_lstm_model(input_shape)
    # model.summary()

    BATCH_SIZE = 128
    EPOCHS = 30
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    
    kf = KFold(n_splits=7, shuffle=True, random_state=42) 
    
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'model/lstm_transformer.h5',
        save_weight_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(x_val, y_val),
              callbacks=[checkpoint_callback],
              verbose=1)

    return model