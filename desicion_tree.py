import copy
import helper_functions as hf
import numpy as np
import pandas as pd

df = pd.read_csv("preprocessed_dataset.csv")
features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"]

# hyperparameters
depth_threshold = 10


class Node:

    def __init__(self, records, parent, branch):
        self.attribute = None
        self.records = records
        self.parent = parent
        self.branch = branch
        self.children = []
        self.label = None


def Split(parent):
    attribute = parent.attribute
    feature = df[attribute]
    values = hf.ExtractValues(feature)

    children = []
    for value in values.keys():
        child_records = []

        for element in parent.records:
            if feature[element] == value:
                child_records.append(element)

        if len(child_records) != 0:
            children.append(Node(child_records, parent, value))

    return children


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
    children = Split(parent)
    gini_split = 0
    for child_node in children:
        gini_split += len(child_node.records) * GINI(child_node)
    gini_split /= len(parent.records)
    return gini_split


def Build_DT(root, unexpanded_features, depth):
    labels = df["income"]
    class1 = 0
    class2 = 0
    for element in root.records:
        if labels[element] == "<=50K":
            class1 += 1
        else:
            class2 += 1

    if depth >= depth_threshold or len(unexpanded_features) == 0 or class1 == 0 or class2 == 0:
        root.children = None
        if class1 > class2:
            root.label = "<=50K"
        else:
            root.label = ">50K"
        return

    biggest_gini = -np.inf
    feature_to_expand = ""
    for feature in unexpanded_features:
        root.attribute = feature
        gini_split = GINI_spilt(root)
        if gini_split >= biggest_gini:
            biggest_gini = gini_split
            feature_to_expand = feature

    root.attribute = feature_to_expand
    root.children = Split(root)

    if len(root.children) == 0:
        root.children = None
        if class1 > class2:
            root.label = "<=50K"
        else:
            root.label = ">50K"
        return

    unexpanded_features_copy = copy.deepcopy(unexpanded_features)
    unexpanded_features_copy.remove(feature_to_expand)

    for child in root.children:
        Build_DT(child, unexpanded_features_copy, depth + 1)


if __name__ == "__main__":
    root_records = []
    for i in range(len(df)):
        root_records.append(i)
    root_node = Node(root_records, None, None)
    Build_DT(root_node, features, 0)
