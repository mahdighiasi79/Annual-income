import helper_functions as hf
import numpy as np
import pandas as pd

df = pd.read_csv("preprocessed_dataset.csv")


class Node:

    def __init__(self, attribute, records, parent):
        self.attribute = attribute
        self.records = records
        self.parent = parent
        self.children = []


def GINI(node):
    income = df["income"]
    labels = []
    for element in node.records:
        labels.append(income[element])
    labels = np.array(labels)
    class1 = (labels == "<=50K")
    class2 = (labels == ">50K")
    num_class1 = np.sum(class1, axis=0, keepdims=False)
    num_class2 = np.sum(class2, axis=0, keepdims=False)
    num_records = len(node.records)
    p1 = num_class1 / num_records
    p2 = num_class2 / num_records
    gini = 1 - (pow(p1, 2) + pow(p2, 2))
    return gini


def GINI_spilt(parent):
    attribute = parent.attribute
    feature = df[attribute]
    values = hf.ExtractValues(attribute)

    children = []
    for value in values.keys():
        child = []
        for element in parent.records:
            if feature[element] == value:
                child.append(element)
        children.append(Node("", child, parent))

    gini_split = 0
    for child_node in children:
        gini_split += len(child_node.records) * GINI(child_node)
    gini_split /= len(parent.records)
    return gini_split


if __name__ == "__main__":
    print()
