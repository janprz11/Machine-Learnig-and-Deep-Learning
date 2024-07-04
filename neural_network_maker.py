from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

def make_model_ANN(input_shape):
        model = Sequential([

        Dense(units=64, activation='relu', input_dim=input_shape),
        
        Dense(units=32, activation='relu'),

        Dropout(0.8),

        Dense(units=16, activation='relu'),

        Dense(units=1, activation='sigmoid')
        ])
        return model

def make_model_CNN(input_shape_1, input_shape_2):
        model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape_1, input_shape_2, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
        ])
        return model