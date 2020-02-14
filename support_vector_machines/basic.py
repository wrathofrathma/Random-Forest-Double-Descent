from sklearn import svm
from tensorflow.keras.datasets import mnist

# import graphviz

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

clf = svm.SVC()


clf = clf.fit(x_train, y_train)


def evaluate(clf, x, y):
    total = len(y)
    count = 0.0
    for v, k in zip(x, y):
        if clf.predict([v]) == k:
            count += 1.0
    print("Accuracy: " + str(count / total))


evaluate(clf, x_test, y_test)
