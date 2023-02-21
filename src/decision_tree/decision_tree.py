import graphviz
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def main():
    dataset = load_wine()
    x, y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=dataset.feature_names,
                                    class_names=dataset.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view("tree")
    y_hat = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_hat, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.savefig("confusion_matrix.jpeg")


if __name__ == '__main__':
    main()
