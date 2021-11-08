import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


dataTrain = pd.read_csv("train.csv")
X_train, X_test = train_test_split(
    dataTrain, test_size=0.5, random_state=int(time.time()))
gnb = GaussianNB()

used_features = [
    "ChidreenId"
    "Age",
    "Happy",
    "Sad"
]

gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)

y_pred = gnb.predict(X_test[used_features])

print("test finish")
