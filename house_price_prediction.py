# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Housing.csv")

# Display First Five Rows
print("===== FIRST 5 ROWS =====")
print(df.head())

# Dataset Information
print("\n===== DATASET INFORMATION =====")
df.info()

# Statistical Summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True)

print("\n===== DATA AFTER ENCODING =====")
print(df.head())

# Select Features (X) and Target (y)

X = df.drop("price", axis=1)
y = df["price"]

print("\n===== FEATURES (X) =====")
print(X.head())

print("\n===== TARGET (y) =====")
print(y.head())

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== TRAINING DATA =====")
print(X_train.shape)

print("\n===== TESTING DATA =====")
print(X_test.shape)

# Select features (X) and target (y)
X = df.drop("price", axis=1)
y = df["price"]

print("\n===== FEATURES (X) =====")
print(X.head())

print("\n===== TARGET (y) =====")
print(y.head())

from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\n===== MODEL TRAINED SUCCESSFULLY =====")

# Predict house prices
y_pred = model.predict(X_test)

print("\n===== PREDICTED HOUSE PRICES =====")
print(y_pred[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Plot Actual vs Predicted Prices

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.savefig("actual_vs_predicted.png")
plt.show()