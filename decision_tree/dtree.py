from sklearn import tree

from tensorflow.keras.datasets import mnist

# import graphviz

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf2 = tree.DecisionTreeClassifier()


clf = clf.fit(x_train, y_train)
clf2 = clf2.fit(x_train, y_train)

# graph_data = tree.export_graphviz(
# clf, out_file=None, special_characters=True, filled=True
# )
# graph = graphviz.Source(graph_data)
# graph.render("mnist_test")


def evaluate(clf, x, y):
    total = len(y)
    count = 0.0
    for v, k in zip(x, y):
        if clf.predict([v]) == k:
            count += 1.0
    print("Accuracy: " + str(count / total))


evaluate(clf, x_test, y_test)
evaluate(clf2, x_test, y_test)
