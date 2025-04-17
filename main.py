# COVID-19 ML Analysis and Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("C:/Users/Hp/Downloads/archive (1)/synthetic_covid19_data.csv")  # Adjust path if needed
print("Shape:", df.shape)
print(df.head())

# Check for nulls
print(df.isnull().sum())

# Drop irrelevant columns
df.drop(['Unnamed: 0'], axis=1, errors='ignore', inplace=True)

# Drop nulls
df = df.dropna()

# Label encode object columns
label_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Describe dataset
print(df.describe())

# Improved Correlation Heatmap (Cleaner and More Readable)

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate full correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Pick a target variable to base correlation on (you can change this)
target = 'total_cases'

# Get top 15 features most correlated with the target
top_features = correlation_matrix[target].abs().sort_values(ascending=False).head(15).index

# Subset correlation matrix for top features
top_corr = df[top_features].corr()

# Plot
plt.figure(figsize=(14, 10))
sns.heatmap(
    top_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    annot_kws={"size": 9}
)
plt.title(f"Top Correlations with '{target}'", fontsize=16, pad=30)  # Increased padding to ensure title visibility
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout(pad=5.0)  # Adjusted padding around the plot to avoid cuts
plt.show()

# Vaccinations vs New Cases
sns.scatterplot(data=df, x='people_vaccinated', y='new_cases', hue='continent')
plt.title("Vaccination vs New Cases")
plt.tight_layout()
plt.savefig("vaccination_vs_newcases.png")
plt.show()
plt.close()

# Create Fatality Rate & Category
df['fatality_rate'] = df['total_deaths'] / (df['total_cases'] + 1)
df['fatality_rate_category'] = pd.cut(df['fatality_rate'],
                                      bins=[-1, 0.01, 0.03, 1],
                                      labels=['Low', 'Medium', 'High'])

# Encode target
df['fatality_rate_category'] = le.fit_transform(df['fatality_rate_category'])

# Feature Matrix
X = df.drop(['fatality_rate', 'fatality_rate_category'], axis=1)
y = df['fatality_rate_category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))

# Future Prediction — Linear Regression on New Cases
features = ['people_vaccinated', 'population_density', 'total_tests', 'total_deaths']
df = df.dropna(subset=features + ['new_cases'])

X_reg = df[features]
y_reg = df['new_cases']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_pred_reg = lr.predict(X_test_reg)

# Evaluation
print("Linear Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("Linear Regression R2 Score:", r2_score(y_test_reg, y_pred_reg))

# Plot Actual vs Predicted New Cases
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.xlabel("Actual New Cases")
plt.ylabel("Predicted New Cases")
plt.title("Linear Regression: New Case Predictions")
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_actual_vs_predicted.png")
plt.show()
plt.close()

# Save cleaned data
df.to_csv("cleaned_covid_data.csv", index=False)

print("Analysis complete. Visualizations and cleaned data saved.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("C:/Users/Hp/Downloads/archive (1)/synthetic_covid19_data.csv")  # Make sure to replace with the actual file path

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# --- Time Series Analysis for Total Cases, Deaths, and Tests Over Time ---
plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['total_cases'], label='Total Cases', color='blue')
plt.plot(df['date'], df['total_deaths'], label='Total Deaths', color='red')
plt.plot(df['date'], df['total_tests'], label='Total Tests', color='green')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Time Series: Total Cases, Deaths, and Tests Over Time', fontsize=16)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Vaccination Progress Over Time ---
plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['total_vaccinations'], label='Total Vaccinations', color='purple')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Vaccinations', fontsize=12)
plt.title('Vaccination Progress Over Time', fontsize=16)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Policy Measures vs. Total Cases/Deaths ---
policy_columns = ['stringency_index', 'government_response_index', 'lockdown', 'travel_restrictions']

plt.figure(figsize=(12, 8))
for policy in policy_columns:
    plt.plot(df['date'], df[policy], label=policy)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Policy Measure Value', fontsize=12)
plt.title('Policy Measures Over Time', fontsize=16)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['total_cases'], label='Total Cases', color='blue', linestyle='--')
plt.plot(df['date'], df['total_deaths'], label='Total Deaths', color='red', linestyle='--')
plt.plot(df['date'], df['stringency_index'], label='Stringency Index', color='purple')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Counts and Policy Measures', fontsize=12)
plt.title('Total Cases/Deaths vs. Policy Measures', fontsize=16)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Heatmap for Missing Values ---
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# --- Correlation Heatmap ---
corr_matrix = df.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# --- Pairplot for Feature Relationships ---
sns.pairplot(df[['total_cases', 'total_deaths', 'total_tests', 'total_vaccinations', 'stringency_index']])
plt.suptitle('Pairplot for Feature Relationships', fontsize=16)
plt.tight_layout()
plt.show()

# --- Predictive Modeling: Linear Regression for Total Cases ---
# Prepare features and target
X = df[['total_tests', 'total_vaccinations', 'stringency_index', 'total_deaths']]
y = df['total_cases']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, color='blue', alpha=0.7, label='Actual Total Cases')
plt.scatter(y_test.index, y_pred, color='red', alpha=0.7, label='Predicted Total Cases')
plt.xlabel('Index', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.title('Actual vs Predicted Total Cases', fontsize=16)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# --- Prediction Analysis: Predicted vs Actual Total Cases ---
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Total Cases', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Total Cases', color='red')
plt.xlabel('Index', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.title('Actual vs Predicted Total Cases (Time Series)', fontsize=16)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# --- Time Series Forecasting (ARIMA or other advanced models can be used for predictions) ---
# Note: Implementing forecasting here is optional and depends on the data's time series nature.