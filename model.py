import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('dataset.csv')
X = np.array(data.iloc[:,0:3])
y = np.array(data.iloc[:,3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lm = LinearRegression()
lm.fit(X_train, y_train)

filename = 'FlaskAPI/model.pkl'
pickle.dump(lm, open(filename, 'wb'))