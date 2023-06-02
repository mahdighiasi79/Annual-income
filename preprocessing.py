import math
import pandas as pd


def PrepareAttributes(file_name):
    df = pd.read_csv(file_name)

    # age
    age = df["age"]
    for i in range(len(age)):
        if age[i] < 10:
            df.at[i, "age"] = "child"
        elif age[i] < 20:
            df.at[i, "age"] = "teenager"
        elif age[i] < 30:
            df.at[i, "age"] = "adult"
        elif age[i] < 40:
            df.at[i, "age"] = "young"
        elif age[i] < 50:
            df.at[i, "age"] = "middle_aged"
        elif age[i] < 60:
            df.at[i, "age"] = "middle_aged2"
        elif age[i] < 70:
            df.at[i, "age"] = "old"
        elif age[i] < 80:
            df.at[i, "age"] = "very_old"
        else:
            df.at[i, "age"] = "last_years"

    # work class
    work_class = df["workclass"]
    for i in range(len(work_class)):
        if work_class[i] == '?':
            df.at[i, "workclass"] = "Private"

    # fnlwgt
    fnlwgt = df["fnlwgt"]
    sorted_fnlwgt = sorted(fnlwgt)
    p25 = sorted_fnlwgt[math.floor(0.25 * len(fnlwgt))]
    p50 = sorted_fnlwgt[math.floor(0.5 * len(fnlwgt))]
    p75 = sorted_fnlwgt[math.floor(0.75 * len(fnlwgt))]

    for i in range(len(fnlwgt)):
        if fnlwgt[i] <= p25:
            df.at[i, "fnlwgt"] = "p25"
        elif fnlwgt[i] <= p50:
            df.at[i, "fnlwgt"] = "p50"
        elif fnlwgt[i] <= p75:
            df.at[i, "fnlwgt"] = "p75"
        else:
            df.at[i, "fnlwgt"] = "p100"

    # education-num
    education_num = df["education-num"]
    for i in range(len(education_num)):
        if education_num[i] <= 8:
            df.at[i, "education-num"] = "elementary"
        elif education_num[i] <= 14:
            df.at[i, "education-num"] = "average"
        else:
            df.at[i, "education-num"] = "high_education"

    # occupation
    occupation = df["occupation"]
    for i in range(len(occupation)):
        if occupation[i] == '?':
            df.at[i, "occupation"] = "Other-service"

    # capital-gain
    capital_gain = df["capital-gain"]
    for i in range(len(capital_gain)):
        if capital_gain[i] == 0:
            df.at[i, "capital-gain"] = "zero"
        else:
            df.at[i, "capital-gain"] = "nonzero"

    # capital-loss
    capital_loss = df["capital-loss"]
    for i in range(len(capital_loss)):
        if capital_loss[i] == 0:
            df.at[i, "capital-loss"] = "zero"
        else:
            df.at[i, "capital-loss"] = "nonzero"

    # hours-per-week
    hours_per_week = df["hours-per-week"]
    for i in range(len(hours_per_week)):
        if hours_per_week[i] < 25:
            df.at[i, "hours-per-week"] = "low"
        elif hours_per_week[i] < 50:
            df.at[i, "hours-per-week"] = "normal"
        elif hours_per_week[i] < 75:
            df.at[i, "hours-per-week"] = "high"
        else:
            df.at[i, "hours-per-week"] = "very_high"

    df.to_csv("preprocessed_dataset.csv")


if __name__ == "__main__":
    PrepareAttributes("Dataset1.csv")
    PrepareAttributes("Dataset1_Unknown.csv")
