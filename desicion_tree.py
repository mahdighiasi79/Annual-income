import helper_functions as hf
import pandas as pd
import numpy as np
import pickle
import copy
import math

df = pd.read_csv("preprocessed_dataset.csv")
features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
            "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"]


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


def Build_DT(root, unexpanded_features, depth, threshold):
    labels = df["income"]
    class1 = 0
    class2 = 0
    for element in root.records:
        if labels[element] == "<=50K":
            class1 += 1
        else:
            class2 += 1

    if depth >= threshold or len(unexpanded_features) == 0 or class1 == 0 or class2 == 0:
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
        Build_DT(child, unexpanded_features_copy, depth + 1, threshold)


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


def F1_score(records, dt):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for record_id in records:
        label = df["income"][record_id]
        predicted_label = Predict(dt, df.iloc(0)[record_id])
        if label == ">50K":
            if predicted_label == ">50K":
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_label == ">50K":
                false_positives += 1

    if (true_positives + false_positives) == 0:
        p = 0
    else:
        p = true_positives / (true_positives + false_positives)
    if (true_positives + false_negatives) == 0:
        r = 0
    else:
        r = true_positives / (true_positives + false_negatives)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = (p * r) / (p + r)
    return [p, r, f1]


if __name__ == "__main__":
    # # build tree
    # train_set = np.arange(math.floor(0.8 * len(df)))
    # root_node = Node(train_set, None, None)
    # depth_threshold = 5
    # Build_DT(root_node, features, 0, depth_threshold)
    #
    # # saving the built model into a file
    # with open("decision_tree.pkl", "wb") as file:
    #     pickle.dump(root_node, file)
    #     file.close()

    # loading the model from file
    with open("decision_tree.pkl", "rb") as file:
        decision_tree = pickle.load(file)
        file.close()

    test_set = np.arange(math.floor(0.8 * len(df)), len(df))
    results = F1_score(test_set, decision_tree)

    print("precision:", results[0])
    print("recall", results[1])
    print("f1 score", results[2])

    # 10-fold cross validation testing
    # cross_validation_set_size = math.floor(0.1 * len(df))
    # ultimate_depth_threshold = -1
    # average_precision = 0
    # average_recall = 0
    # average_f1_score = 0
    #
    # for i in range(9):
    #     train_set = np.append(np.arange(i * cross_validation_set_size),
    #                           np.arange((i + 2) * cross_validation_set_size, len(df)), axis=0)
    #     cross_validation_set = np.arange(i * cross_validation_set_size, (i + 1) * cross_validation_set_size)
    #     test_set = np.arange((i + 1) * cross_validation_set_size, (i + 2) * cross_validation_set_size)
    #     biggest_f1_score = -np.inf
    #
    #     for j in range(len(features)):
    #         depth_threshold = j
    #         root_node = Node(train_set, None, None)
    #         Build_DT(root_node, features, 0, depth_threshold)
    #
    #         results = F1_score(cross_validation_set, root_node)
    #         precision = results[0]
    #         recall = results[1]
    #         f1_score = results[2]
    #
    #         if f1_score >= biggest_f1_score:
    #             biggest_f1_score = f1_score
    #             ultimate_depth_threshold = depth_threshold
    #         else:
    #             break
    #
    #         print("round:", i)
    #         print("depth threshold:", j)
    #         print("precision:", precision)
    #         print("recall:", recall)
    #         print("f1 score:", f1_score)
    #         print("/////////////////////////////////////////////////////////////")
    #
    #     depth_threshold = ultimate_depth_threshold
    #     root_node = Node(train_set, None, None)
    #     Build_DT(root_node, features, 0, depth_threshold)
    #     results = F1_score(test_set, root_node)
    #     average_precision += results[0]
    #     average_recall += results[1]
    #     average_f1_score += results[2]
    #
    #     print("end of round:", i)
    #     print("depth:", depth_threshold)
    #     print("precision:", results[0])
    #     print("recall:", results[1])
    #     print("f1 score:", results[2])
    #     print("////////////////////////////////////////////////////////////////")
    #
    # average_precision /= 9
    # average_recall /= 9
    # average_f1_score /= 9
    # print("average precision:", average_precision)
    # print("average recall:", average_recall)
    # print("average f1 score", average_f1_score)
