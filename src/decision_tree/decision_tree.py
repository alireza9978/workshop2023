import graphviz
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # load and splitting the dataset
    dataset = load_wine()
    x, y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Training a simple tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    # plotting the trained tree
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=dataset.feature_names,
                                    class_names=dataset.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view("decision_tree/tree")

    # Evaluating the trained model on the test dataset
    # Using the trained model to predict test data
    y_hat = clf.predict(x_test)

    # Calculating and Saving the confusion Matrix
    cm = confusion_matrix(y_test, y_hat, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.savefig("decision_tree/confusion_matrix.jpeg")
    # Printing accuracy
    print("Accuracy: ", accuracy_score(y_test, y_hat))


if __name__ == '__main__':
    main()
