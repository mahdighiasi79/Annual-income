import math
import numpy as np
import pandas as pd

outlier_threshold = 0.0001


def ExtractValues(feature):
    values = {}
    for value in feature:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1
    return values


def HasMissingValue(attribute):
    df = pd.read_csv("Dataset1.csv")
    feature = df[attribute]
    for element in feature:
        if element == '?':
            return True
    return False


def InformativeValues(attribute):
    df = pd.read_csv("preprocessed_data.csv")
    records = len(df)
    diag_1 = np.array(df[attribute])
    labels = np.array(df["readmitted"])
    l1 = (labels == ">30")
    l2 = (labels == "<30")
    l = l1 + l2

    values = ExtractValues(attribute)
    informative_values = {}
    for key in values.keys():
        value = (diag_1 == key)
        n_276 = np.sum(value, axis=0, keepdims=False)
        p = value * l
        p = np.sum(p, axis=0, keepdims=False)
        p /= n_276
        p *= 100
        informative_values[key] = p
    print(values)
    print(informative_values)

    information = {}
    for value in informative_values.keys():
        informativeness = (values[value] / records) * 100 * pow(abs(informative_values[value] - 50), 2)
        information[value] = informativeness
        print(value, informativeness)
    sorted_informativeness = sorted(information.values())
    print(sorted_informativeness)

    # while True:
    #     query = input()
    #     if query == "informativeness":
    #         print(informative_values[input()])
    #     elif query == "value":
    #         i = int(input())
    #         for key in information.keys():
    #             if math.floor(information[key]) == i:
    #                 print(key)


def MeanVariance(feature):
    records = len(feature)
    mean = np.sum(feature, axis=0, keepdims=False) / records
    variance = np.power(feature - mean, 2)
    variance = np.sum(variance, axis=0, keepdims=False) / (records - 1)
    return [mean, variance]


def Entropy(feature):
    records = len(feature)
    values = {}

    for value in feature:
        if values.get(value) is None:
            values[value] = 1
        else:
            values[value] += 1

    entropy = 0
    for key in values.keys():
        probability = values[key] / records
        entropy += probability * math.log2(probability)
    entropy *= -1
    return entropy


def MutualInformation(feature1, feature2):
    feature1_np = np.array(feature1)
    feature2_np = np.array(feature2)
    records = len(feature1)
    outcomes1 = np.array([])
    outcomes2 = np.array([])

    for value in feature1:
        if value not in outcomes1:
            outcomes1 = np.append(outcomes1, value)

    for value in feature2:
        if value not in outcomes2:
            outcomes2 = np.append(outcomes2, value)

    n1 = len(outcomes1)
    n2 = len(outcomes2)
    probability_matrix = np.zeros((n1, n2))

    for i in range(n1):
        f1 = (feature1_np == outcomes1[i])
        for j in range(n2):
            f2 = (feature2_np == outcomes2[j])
            f3 = f1 * f2
            count = np.sum(f3, axis=0)
            if count == 0:
                count = records
            probability_matrix[i][j] = count
    probability_matrix /= records

    joint_entropy = probability_matrix * np.log2(probability_matrix)
    joint_entropy = np.sum(joint_entropy, axis=0)
    joint_entropy = np.sum(joint_entropy, axis=0)
    joint_entropy *= -1

    mutual_information = Entropy(feature1) + Entropy(feature2) - joint_entropy
    return mutual_information


def DetectOutliers(feature):
    records = len(feature)

    categories = {}
    for category in feature:
        if categories.get(category) is None:
            categories[category] = 1
        else:
            categories[category] += 1

    noise_categories = []
    for key in categories.keys():
        if categories[key] < (records * outlier_threshold):
            noise_categories.append(key)

    result = []
    for i in range(records):
        if feature[i] in noise_categories:
            result.append(i)
    return result
