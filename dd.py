from dataprep import prep_mnist
from sklearn.ensemble import RandomForestClassifier


def bulk_train_estimators(start, end, stride):
    accuracies = []
    (xtrain, ytrain), (xtest, ytest) = prep_mnist()
    for i in range(start, end, stride):
        clf = RandomForestClassifier(n_estimators=i)
        clf = clf.fit(xtrain, ytrain)
        accuracies += [clf.score(xtest, ytest)]
    print(accuracies)


if __name__ == "__main__":
    bulk_train_estimators(1, 100, 2)
    # print("Accuracy:" + str(clf.score(xtest, ytest)))

