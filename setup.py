#LSTM
import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
from yahoo_fin import stock_info as yff
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
#CNN
import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import VGG16
import numpy as np
import yfinance as yf
import mplfinance as mpf
from datetime import datetime
from PIL import Image
import json
import matplotlib
matplotlib.use('Agg')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPU Available:", tf.config.list_physical_devices('GPU'))