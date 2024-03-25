from faker import Faker
import pandas as pd
import random
import datetime
from sklearn.model_selection import train_test_split

fake = Faker()

def generate_sensor_readings(num_readings):
    readings = []
    for _ in range(num_readings):
        timestamp = fake.date_time_this_year()  # Generate a timestamp within the current year
        sensor_id = random.randint(1, 10)  # Assume 10 sensors

        # Generate sensor values for different types of sensors
        sensor_values = {
            'pressure': random.uniform(800.0, 1200.0),  # Example range for pressure in hPa
            'temperature': random.uniform(-20.0, 50.0),  # Example range for temperature in Celsius
            'humidity': random.uniform(0.0, 100.0),  # Example range for humidity in percentage
            'acceleration_x': random.uniform(-10.0, 10.0),  # Example range for acceleration in m/s^2
            'acceleration_y': random.uniform(-10.0, 10.0),  # Example range for acceleration in m/s^2
            'acceleration_z': random.uniform(-10.0, 10.0)  # Example range for acceleration in m/s^2
            # add more if needed. 
        }

        # Append sensor readings to the list
        for sensor_type, value in sensor_values.items():
            readings.append({'timestamp': timestamp, 'sensor_id': sensor_id, 'sensor_type': sensor_type, 'sensor_value': value})
    return readings

# Generate synthetic sensor readings
sensor_readings = generate_sensor_readings(10000)

# Convert to pandas DataFrame
df = pd.DataFrame(sensor_readings)

# Save the sensor readings to a CSV file
df.to_csv('rocket_sensor_readings.csv', index=False)

# Display the first few rows of the DataFrame
print(df.head())

# Split the data into training, validation, and test sets
train_ratio = 0.7  # 70% of the data for training
validation_ratio = 0.15  # 15% of the data for validation
test_ratio = 0.15  # 15% of the data for testing

# Split the data into training and the rest
train_data, rest_data = train_test_split(df, test_size=1 - train_ratio, random_state=42)

# Split the rest into validation and test sets
validation_data, test_data = train_test_split(rest_data, test_size=test_ratio / (validation_ratio + test_ratio), random_state=42)

# Display the sizes of the resulting datasets
print("Training data size:", len(train_data))
print("Validation data size:", len(validation_data))
print("Test data size:", len(test_data))

# Save the data splits to files
train_data.to_csv('train_sensor_readings.csv', index=False)
validation_data.to_csv('validation_sensor_readings.csv', index=False)
test_data.to_csv('test_sensor_readings.csv', index=False)

# Save the data splits to Excel files
train_data.to_excel('train_sensor_readings.xlsx', index=False)
validation_data.to_excel('validation_sensor_readings.xlsx', index=False)
test_data.to_excel('test_sensor_readings.xlsx', index=False)
