import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error

# Load model and data
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('salary_data.csv')
X = df[['YearsExperience']]
y = df['Salary']

# Make predictions
predictions = model.predict(X)

# Calculate metrics
r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)

print(f"R² Score: {r2:.3f}")
print(f"MAE: ${mae:.2f}")