import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
housing = pd.read_csv("train.csv")
features = ["MSSubClass", "MSZoning_RL", "MSZoning_FV", "MSZoning_RH", "MSZoning_RM",
            "LotFrontage", "LotArea", "Utilities_AllPub", "Utilities_NoSeWa", "OverallQual", "OverallCond",
            "YearBuilt", "ExterQual_Ex", "ExterQual_Gd", "ExterQual_TA", "ExterQual_Fa", "ExterCond_Fa",
            "ExterCond_Po", "ExterCond_Ex", "ExterCond_Gd", "ExterCond_TA", "BsmtCond_Gd", "BsmtCond_TA", "BsmtCond_Fa",
            "BsmtCond_Po", "BsmtFinType2_GLQ", "BsmtFinType2_ALQ", "BsmtFinType2_BLQ", "BsmtFinType2_Rec", "BsmtFinType2_LwQ",
            "BsmtFinType2_Unf", "HeatingQC_Ex", "HeatingQC_Gd", "HeatingQC_TA", "HeatingQC_Fa", "HeatingQC_Po",
            "Electrical_SBrkr", "Electrical_FuseA", "Electrical_FuseF", "Electrical_FuseP", "Electrical_Mix", "GrLivArea", "BedroomAbvGr", "PoolArea", "YrSold"]
housing = pd.get_dummies(housing)
housing = housing.dropna(axis=0, how='any')
print(housing)
y = housing.SalePrice
X = housing[features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)


LR = LogisticRegression(random_state=1, max_iter=5000)
LR.fit(train_X, train_y)
data_y = LR.predict(test_X)
print(mean_absolute_error(data_y, test_y))


DTC = DecisionTreeClassifier(max_depth=38, random_state=1)
DTC.fit(train_X, train_y)
data_y_dtc = DTC.predict(test_X)
print(mean_absolute_error(data_y_dtc, test_y))

DTR = DecisionTreeRegressor(max_depth=38, random_state=1)
DTR.fit(train_X, train_y)
data_y_dtr = DTR.predict(test_X)
print(mean_absolute_error(data_y_dtr, test_y))