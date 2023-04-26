from json import load
from string import digits
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
x_data_all = digits.data
y_data_all = digits.target
print(x_data_all.shape)
print(y_data_all.shape)

x_train_all, x_test, y_train_all, y_test = train_test_split(
    x_data_all,y_data_all,random_state=7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all,y_train_all,random_state=11
)
print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)

scaler = StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)