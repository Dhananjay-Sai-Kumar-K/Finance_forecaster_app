from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape):
    """
    Builds and compiles a Bidirectional LSTM model.
    """
    model = Sequential()
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    
    # Second Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1))
    
    # Use Adam optimizer with a slightly lower learning rate for stability
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=25, batch_size=32):
    """
    Trains the LSTM model.
    """
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return model

def predict_lstm(model, data):
    """
    Makes predictions using the trained LSTM model.
    """
    predictions = model.predict(data)
    return predictions
