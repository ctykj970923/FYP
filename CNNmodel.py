from setup import *
from image_generator import *

def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),#32 #64
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),#512
    tf.keras.layers.Dropout(0.1),  
    tf.keras.layers.Dense(5, activation='softmax')
])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)#0.00001

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = create_model()

    checkpoint_callback = ModelCheckpoint(
        'model/cnn.h5',
        save_weight_only=True,
        save_best_only=True,
        monitor='val_accuracy',  
        mode='max',  
        verbose=1
    )

    TRAINING_DIR = train_dir
    VALIDATION_DIR = valid_dir
    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

    history = model.fit(train_generator,
                        epochs=50,
                        verbose=1,
                        callbacks=[checkpoint_callback],
                        validation_data=validation_generator)