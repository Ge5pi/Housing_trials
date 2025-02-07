import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the data
housing = pd.read_csv("train.csv")

# Select important features based on correlation and domain knowledge
features = [
    'OverallQual',    # Overall material and finish quality
    'GrLivArea',      # Above ground living area
    'GarageCars',     # Size of garage in car capacity
    'TotalBsmtSF',    # Total basement square footage
    'FullBath',       # Number of full bathrooms
    'YearBuilt',      # Original construction date
    'YearRemodAdd',   # Remodel date
    'GarageArea',     # Size of garage in square feet
    'MSZoning',       # General zoning classification
    'ExterQual',      # Exterior material quality
    'KitchenQual',    # Kitchen quality
    'BsmtQual',       # Height of basement
    'GarageFinish',   # Interior finish of garage
    'FireplaceQu',    # Fireplace quality
    'Foundation',     # Type of foundation
]

# Create a copy of the dataframe with selected features
df = housing[features + ['SalePrice']]

# Handle missing values
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
df['GarageFinish'] = df['GarageFinish'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('None')

# Convert categorical variables to numeric using LabelEncoder
categorical_features = ['MSZoning', 'ExterQual', 'KitchenQual', 'BsmtQual',
                       'GarageFinish', 'FireplaceQu', 'Foundation']

label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    df[feature] = label_encoders[feature].fit_transform(df[feature])

# Split features and target
X = df[features]
y = df['SalePrice']

# Split the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the numerical features
scaler = StandardScaler()
numerical_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                     'FullBath', 'YearBuilt', 'YearRemodAdd', 'GarageArea']
train_X[numerical_features] = scaler.fit_transform(train_X[numerical_features])
test_X[numerical_features] = scaler.transform(test_X[numerical_features])

# Try different models and print their MAE scores
models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1, max_depth=20),
    'RandomForestRegressor': RandomForestRegressor(random_state=1, n_estimators=100, max_depth=20)
}

for name, model in models.items():
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    mae = mean_absolute_error(test_y, predictions)
    print(f"{name} MAE: ${mae:,.2f}")

# Print feature importance for RandomForestRegressor
rf_model = models['RandomForestRegressor']
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))
