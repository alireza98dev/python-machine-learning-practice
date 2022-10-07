import sklearn
from sklearn import datasets, svm, metrics

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["malignant", "benign"]

clf = svm.SVC(kernel="linear")
model = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
for i in range(len(y_pred)):
    print("prdicted : ", y_pred[i], "actual value : ", y_test[i])
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ", acc)