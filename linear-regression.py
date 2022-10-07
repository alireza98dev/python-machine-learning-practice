import pickle

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import style, pyplot

data = pd.read_csv("./data/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "traveltime", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]]

predict = "G3"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# Recalculate the model ====>
# bestScore = 0
# while bestScore < 0.96:
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     score = linear.score(x_test, y_test)
#     if score > bestScore:
#         bestScore = score
#         print("Best Score : ", bestScore)
#         with open("./data/student.pickle", "wb") as f:
#             pickle.dump(linear, f)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
pickle_in = open("./data/student.pickle", "rb")
linear = pickle.load(pickle_in)
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

style.use("ggplot")
pyplot.scatter(data["failures"], data["G3"])
pyplot.xlabel("G1")
pyplot.ylabel("Final grade")
pyplot.show()
