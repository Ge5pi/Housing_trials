import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

#Housing data
df = pd.read_csv("train.csv")

# 2. Basic data exploration
print("Data frame shape:", df.shape)
print("\nData types and non-null counts:\n", df.info())
print("\nBasic stats:\n", df.describe())

# 3. Basic cleaning

# 3.1 Drop columns with too many missing values (example: Alley, PoolQC, Fence, MiscFeature often missing)
cols_with_many_nans = ["Alley", "PoolQC", "Fence", "MiscFeature"]
df.drop(columns=cols_with_many_nans, inplace=True)


# 3.2 Fill numeric missing values with median
num_features = df.select_dtypes(include=[np.number]).columns
num_imputer = SimpleImputer(strategy="median")
df[num_features] = num_imputer.fit_transform(df[num_features])


# 3.3 Fill categorical missing values with "None"
cat_features = df.select_dtypes(include=["object"]).columns
for cat_col in cat_features:
    df[cat_col] = df[cat_col].fillna("None")


# 3.4 Remove a small number of extreme outliers for SalePrice (optional)
#      E.g., Suppose any price over 700000 might be an outlier in this dataset
df = df[df["SalePrice"] < 700000]

# 4. Exploratory analysis
#    Quick correlation look at the top correlated features with SalePrice
corr_matrix = df.corr()
top_corr = corr_matrix["SalePrice"].abs().sort_values(ascending=False).head(15)
print("\nTop correlated features with SalePrice:\n", top_corr)


# 5. correlation ratio
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "YearBuilt",
    "YearRemodAdd",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd"
]

X = df[features]
y = df["SalePrice"]

# 6. Split into train/test sets
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# 7. Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
preds = model.predict(test_X)

mae = mean_absolute_error(test_y, preds)

print("\nRandomForestRegressor MAE:", mae)

scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
print("Cross-validated MAE:", -1 * np.mean(scores))