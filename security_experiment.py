
import json
import hmac
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# EXPERIMENT 2: Secure Communication for a Digital Twin
# ============================================================

np.random.seed(42)

# Demo key only. Real systems need securely provisioned keys.
SECRET_KEY = b"demo-key-replace-this-in-real-systems"


# 1. Create a digital-twin sensor reading
def create_payload(message_id, temperature, vibration, network_delay):
    return {
        "message_id": message_id,
        "temperature": round(float(temperature), 3),
        "vibration": round(float(vibration), 3),
        "network_delay": round(float(network_delay), 3),
    }


# 2. Convert the message to consistent bytes
def canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


# 3. Sign a message with HMAC
def sign_message(payload):
    return hmac.new(
        SECRET_KEY,
        canonical_bytes(payload),
        hashlib.sha256
    ).hexdigest()


# 4. Verify message integrity at the receiver
def verify_message(payload, received_signature):
    expected_signature = sign_message(payload)
    return hmac.compare_digest(expected_signature, received_signature)


# 5. Generate normal sensor readings
number_of_training_samples = 1000

training_data = pd.DataFrame({
    "temperature": np.random.normal(25, 1.5, number_of_training_samples),
    "vibration": np.random.normal(0.5, 0.10, number_of_training_samples),
    "network_delay": np.random.normal(30, 4, number_of_training_samples),
})

# Train only on normal readings
detector = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
detector.fit(training_data)

print("Normal-only anomaly detector trained.")


# 6. Simulate messages altered while in transit
print("\n--- Test A: In-transit message tampering ---")

original = create_payload(1001, 25.0, 0.5, 30.0)
original_signature = sign_message(original)

# An attacker changes the temperature but cannot update the signature.
tampered = original.copy()
tampered["temperature"] = 90.0

print("Original message:", original)
print("Tampered message:", tampered)
print("Original signature:", original_signature[:20] + "...")

if verify_message(tampered, original_signature):
    print("INTEGRITY CHECK PASSED")
else:
    print("ALERT: Message modified! Integrity check FAILED.")


# 7. Simulate a compromised sensor that signs abnormal readings
# In this simulation, the compromised sensor can access the key.
print("\n--- Test B: Anomalous readings with valid signatures ---")

test_rows = []

# 100 ordinary readings
for i in range(100):
    payload = create_payload(
        i,
        np.random.normal(25, 1.5),
        np.random.normal(0.5, 0.10),
        np.random.normal(30, 4),
    )
    test_rows.append((payload, 0))

# 20 unusual readings from a simulated compromised sensor
for i in range(100, 120):
    payload = create_payload(
        i,
        np.random.normal(40, 2),
        np.random.normal(1.4, 0.15),
        np.random.normal(65, 5),
    )
    test_rows.append((payload, 1))


# 8. Authenticate messages, then run anomaly detection
results = []

for payload, true_label in test_rows:
    signature = sign_message(payload)

    # This simulated sender signs every message.
    # A real compromised sensor might possess a valid key.
    integrity_ok = verify_message(payload, signature)

    features = pd.DataFrame([{
        "temperature": payload["temperature"],
        "vibration": payload["vibration"],
        "network_delay": payload["network_delay"],
    }])

    # IsolationForest: 1 = normal, -1 = anomaly
    prediction = detector.predict(features)[0]
    anomaly_detected = int(prediction == -1)

    results.append({
        "message_id": payload["message_id"],
        "true_compromised": true_label,
        "integrity_ok": integrity_ok,
        "anomaly_detected": anomaly_detected,
        "temperature": payload["temperature"],
        "vibration": payload["vibration"],
        "network_delay": payload["network_delay"],
    })


# 9. Evaluate anomaly detection
results_df = pd.DataFrame(results)

print("\nMessage integrity results:")
print(results_df["integrity_ok"].value_counts())

print("\nAnomaly detection report:")
print(classification_report(
    results_df["true_compromised"],
    results_df["anomaly_detected"],
    labels=[0, 1],
    target_names=["Normal", "Compromised"],
    zero_division=0
))

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    results_df["true_compromised"],
    results_df["anomaly_detected"],
    labels=[0, 1]
))

# 10. Save results for later analysis
results_df.to_csv("security_experiment_results.csv", index=False)

print("\nResults saved to security_experiment_results.csv")
print("Experiment completed.")
