# Annual-income
In this project, we aim to predict an approximation of people's annual income based on some information about them.
We have 14 features that we should make prediction based on. Those features include information like age, work class, education, native country, work hours per week, etc.
We have predict the "income" column for records. This column can have two values: <=50K or >50K.
"<=50K" means that the respective person have had an annual income less than or equal to 50 thousand dollars.
Respectively, ">50K" means that the respective person have had an annual income more than 50 thousand dollars.
Therefore, we have to classify data into two classes.

The project is devided into three main phases:
1- preprocessing
2- model deployment
3- evaluation

1- preprocessing: 
In this phase, we eliminate missing values and prepare attributes to be used in a decision tree.
Based on the each feature's property, we have treated each one differently.
For the "age" attribute, we separate values into ten categories. Each category involves all values in an interval of length 10.
For instance, people younger than 10 years old are named "child", people whose age is between 10 and 20 are named "teenager", and so on.
For the "workclass" attribute, we replaced all missing values with "Private" which is the less informative value.
For the "fnlwgt" attribute, we separated values into four categories. 
In order to this, we have used percentiles. Such that, the first category covers the values less than the 25th percentile, the second category covers the values between the 25th and 50th percentile, and so on.
For the "education-num" attribute, we separated values into three categories. 
The first category covers the values less than 8, the second one covers the values between 8 and 14, and the third one contains the values bigger than 14.
These separations are proposed based on the distribution of data.
For the "occupation" attribute, we replaced all missing values with "Other-service" which is the less informative value.
For the "capital-gain" and "capital-loss" attributes, based on the data distribution, we have put values into two categories: zero and nonzero.
For the "hours-per-week" attribute, we have divided the domain of values into four partitions and we have put the values into four categories based on them.

2- model deployment:
Decision Tree (DT) is the model we have used for this classification task.
We use a greedy approach for building our tree.
In each node, we expand the feature which gives us the lowest gini_split value.
The number of children of each node is equal to the number of categories its expanded feature has. 
Consequently, our tree is not binary and nodes may have different number of children.

3- evaluation:
10-fold-cross-validation is the approach we have used for evaluation.
The number of people who have an annual income more than 50 thousand dollars are much less than the number of people who have an annual income less than 50 thousand dollars.
Because of that, our data are not evenly distributed among classes. So, we used precision, recall, and f1-score as evaluation metrics.
In general, with a shalow tree, we will have a high precision (around 80%), and with a deep tree, we will have a high recall (around 78%).
But in the case that the f1-score is prioritized, a tree with depth 5 is ideal.
Based on the 10-fold-cross-validation testing, our model has gives an average precision of 64.14%, an average recall of 58.25%, and an average f1-score of 30.49.
The evaluation metrics may be low but keep in mind that if you use Random Forest (RF), you will defenitally get higher metrics.
We didn't use RF beacause it's a university homework and we are obligated to report the DT accuracy. :)
