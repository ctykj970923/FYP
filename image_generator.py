from setup import *
import globalVariable

train_dir = os.path.join(globalVariable.base_dir, 'training')
valid_dir = os.path.join(globalVariable.base_dir, 'validation')

for category in globalVariable.categories:
    train_category_dir = os.path.join(train_dir, category)
    valid_category_dir = os.path.join(valid_dir, category)
    
    if not os.path.exists(train_category_dir):
        os.makedirs(train_category_dir)
        print(f"Created training directory: {train_category_dir}")
    else:
        print(f"Training directory already exists: {train_category_dir}")
    
    if not os.path.exists(valid_category_dir):
        os.makedirs(valid_category_dir)
        print(f"Created validation directory: {valid_category_dir}")
    else:
        print(f"Validation directory already exists: {valid_category_dir}")
        
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_DIR,
        batch_size=8,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        target_size=(150, 150)
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=VALIDATION_DIR,
        batch_size=8,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        target_size=(150, 150)
    )

    return train_generator, validation_generator


