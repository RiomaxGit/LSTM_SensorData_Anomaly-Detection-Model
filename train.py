import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load rocket sensor data
df = pd.read_csv('rocket_sensor_readings.csv')

# Define features and target
X = df[['sensor_id', 'sensor_value']].values  # Exclude 'timestamp' from features - not a valid parameter for calc

# Define the threshold for anomaly detection
threshold = 20.0  # Adjust as needed

# Encode the target variable based on the threshold
y = np.where(df['sensor_value'] > threshold, 1, 0)  # 1 for anomaly, 0 for normal

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape input data to 3D tensor [samples, timesteps, features]
timesteps = 1  # Since each row represents one timestamp
features = X_train.shape[1]

# Define the LSTM model
model = Sequential()
model.add(Input(shape=(timesteps, features)))  # Input layer
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train.reshape(-1, timesteps, features), y_train.reshape(-1, 1), epochs=10, batch_size=32, validation_data=(X_val.reshape(-1, timesteps, features), y_val.reshape(-1, 1)))

# Evaluate the model
loss, accuracy = model.evaluate(X_test.reshape(-1, timesteps, features), y_test.reshape(-1, 1))
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
