
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Make the experiment reproducible
np.random.seed(42)

# 2. Simulate sensor readings from a digital twin
number_of_samples = 2000

temperature = np.random.normal(25, 2, number_of_samples)
vibration = np.random.normal(0.5, 0.15, number_of_samples)
network_delay = np.random.normal(30, 5, number_of_samples)

# 3. Create labels: 0 = normal, 1 = abnormal
abnormal = (
    (temperature > 29)
    | (vibration > 0.85)
    | (network_delay > 42)
).astype(int)

# 4. Build the dataset
data = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "network_delay": network_delay,
    "abnormal": abnormal
})

print("First five sensor readings:")
print(data.head())

print("\nNormal and abnormal sample counts:")
print(data["abnormal"].value_counts().rename(
    index={0: "Normal", 1: "Abnormal"}
))

# 5. Prepare inputs and labels
X = data[["temperature", "vibration", "network_delay"]]
y = data["abnormal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train a baseline machine-learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# 7. Evaluate the model on unseen test samples
predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["Normal", "Abnormal"],
    zero_division=0
))

# 8. Plot the sensor readings
plt.figure(figsize=(9, 5))
plt.scatter(
    data["temperature"],
    data["vibration"],
    c=data["abnormal"],
    alpha=0.6
)
plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Simulated Digital Twin Sensor Readings")
plt.tight_layout()
plt.savefig("sensor_readings.png", dpi=150)
plt.show()

# 9. Save the simulated dataset
data.to_csv("digital_twin_sensor_data.csv", index=False)

print("\nSaved digital_twin_sensor_data.csv")
print("Saved sensor_readings.png")
