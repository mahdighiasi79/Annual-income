import helper_functions as hf
import pandas as pd
import numpy as np
import pickle
import copy
import math

df = pd.read_csv("preprocessed_dataset.csv")
features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"]

# hyperparameters
depth_threshold = 4


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

    lowest_gini = np.inf
    feature_to_expand = ""
    for feature in unexpanded_features:
        root.attribute = feature
        gini_split = GINI_spilt(root)
        if gini_split <= lowest_gini:
            lowest_gini = gini_split
            feature_to_expand = feature

    root.attribute = feature_to_expand
    root.children = Split(root)

    if len(root.children) == 0:
        root.children = None
        root.attribute = None
        if class1 > class2:
            root.label = "<=50K"
        else:
            root.label = ">50K"
        return

    unexpanded_features_copy = copy.deepcopy(unexpanded_features)
    unexpanded_features_copy.remove(feature_to_expand)

    for child in root.children:
        Build_DT(child, unexpanded_features_copy, depth + 1)


def Predict(node, record):
    if node.label is not None:
        return node.label

    attribute = node.attribute
    value = record[attribute]
    for child in node.children:
        if child.branch == value:
            return Predict(child, record)

    labels = df["income"]
    class1 = 0
    class2 = 0
    for element in node.records:
        if labels[element] == "<=50K":
            class1 += 1
        else:
            class2 += 1
    if class1 > class2:
        return "<=50K"
    else:
        return ">50K"


def PrintDT(root):
    if root.label is not None:
        print(root.label)
        return

    print(root.attribute)
    for child in root.children:
        PrintDT(child)


if __name__ == "__main__":
    # # build tree
    # train_set = []
    # for i in range(math.floor(0.8 * len(df))):
    #     train_set.append(i)
    # root_node = Node(train_set, None, None)
    # Build_DT(root_node, features, 0)
    #
    # # saving the built model into a file
    # with open("decision_tree.pkl", "wb") as file:
    #     pickle.dump(root_node, file)
    #     file.close()

    # loading the model from file
    with open("decision_tree.pkl", "rb") as file:
        decision_tree = pickle.load(file)
        file.close()

    # cross validation testing and hyperparameter setting
    # cross_validation_set = []
    # for i in range(math.floor(0.8 * len(df)), math.floor(0.9 * len(df))):
    #     cross_validation_set.append(i)
    #
    # true_positives = 0
    # false_positives = 0
    # false_negatives = 0
    # for record_id in cross_validation_set:
    #     label = df["income"][record_id]
    #     predicted_label = Predict(decision_tree, df.iloc(0)[record_id])
    #     if label == ">50K":
    #         if predicted_label == ">50K":
    #             true_positives += 1
    #         else:
    #             false_negatives += 1
    #     else:
    #         if predicted_label == ">50K":
    #             false_positives += 1
    # precision = true_positives / (true_positives + false_positives)
    # recall = true_positives / (true_positives + false_negatives)
    # f1_score = (precision * recall) / (precision + recall)
    # print("precision:", precision)
    # print("recall:", recall)
    # print("f1 score:", f1_score)

    # test set evaluation
    test_set = []
    for i in range(math.floor(0.9 * len(df)), len(df)):
        test_set.append(i)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for record_id in test_set:
        label = df["income"][record_id]
        predicted_label = Predict(decision_tree, df.iloc(0)[record_id])
        if label == ">50K":
            if predicted_label == ">50K":
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_label == ">50K":
                false_positives += 1
    precision = 100 * true_positives / (true_positives + false_positives)
    recall = 100 * true_positives / (true_positives + false_negatives)
    f1_score = (precision * recall) / (precision + recall)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 score:", f1_score)
