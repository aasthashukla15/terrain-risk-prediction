import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# STEP 1: Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset preview:")
print(data.head())

print("\nColumn names:")
print(data.columns)


# STEP 2: Convert target to numbers
data['Landslide Risk Prediction'] = data['Landslide Risk Prediction'].map({
    'Low': 0,
    'Moderate': 1,
    'High': 2
})


# STEP 3: Remove rows with missing values FROM WHOLE DATASET
data = data.dropna()


# STEP 4: Select features and target

X = data[['Temperature (°C)',
          'Humidity (%)',
          'Precipitation (mm)',
          'Soil Moisture (%)',
          'Elevation (m)']]

y = data['Landslide Risk Prediction']


# STEP 5: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# STEP 6: Train model

model = RandomForestClassifier()

model.fit(X_train, y_train)


# STEP 7: Test model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# STEP 8: Predict custom terrain

sample = pd.DataFrame([[25, 80, 200, 40, 1500]],
columns=[
    'Temperature (°C)',
    'Humidity (%)',
    'Precipitation (mm)',
    'Soil Moisture (%)',
    'Elevation (m)'
])

prediction = model.predict(sample)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("Low Risk Terrain")

elif prediction[0] == 1:
    print("Moderate Risk Terrain")

else:
    print("High Risk Terrain")

import joblib

# save trained model
joblib.dump(model, "terrain_risk_model.pkl")

print("Model saved successfully!")