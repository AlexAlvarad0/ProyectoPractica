from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

data = pd.DataFrame({'X': [1,2,3,4], 'Y': [2,4,6,8]})
X = data [['X']]
Y = data [['Y']]

model=LinearRegression()
model.fit(X,Y)
with open('modelo.pk1', 'wb') as f:
    pickle.dump(model, f)