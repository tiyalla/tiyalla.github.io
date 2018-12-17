+++
date = "2018-12-12"
title = "Evaluating Machine Learning Algorithms' performance in predicting presence of heart disease"
math = "true"

+++

### Introduction
Heart disease is one of the deadliest disease patients face and early detection/prediction is key to achieving timely diagnosis and decision. The key to managing and detecting cardiovascular disease is to evaluate large scores of datasets, compare and mine for information that can be used to predict, prevent, manage and treat chronic diseases such as heart attacks.

Currently, there are a variety of machine learning algorithms that are popular and in use for predicting and performing various tasks. The aim is to compare 4 of the most popular and efficient algorithms on the same data set and evaluate how well they can make heart disease predictions or early detections.

The main goal of this project was to evaluate 4 machine learning algorithms by creating a learning model and evaluating how well they perform on the same data set.

### Machine Learning
Machine learning is a technology that gives computers the ability to learn from past experiences (or data), just like humans. From self-driving cars, to discovering planets — the applications of ML are immense. It uses computational methods to “learn” information directly from data without relying on a predetermined equation as a model. The algorithms adaptively improve their performance as the number of samples available for learning increases.

Machine learning uses a few types of techiques - Supervised Learning, Unsupervised Learning, Semi-supervised, Clustering etc. 
<div align="center">
![](images/machinelearning.PNG)
</div>
<p align="center"> **Figure 1: Machine learning techniques. [via: Matlab]** </p>

##### Supervised learning
As the name suggests needs a human to supervise and tell the computer what to be trained for. We feed the computer with training data containing various features, and we also tell it the right answer. For my project, I used Supervised learning to build my learning model.

##### Unsupervised learning
We let the computer discover patterns on its own, and then choose the one that makes most sense.

##### Supervised learning can be used to solve two problems: Classification and regression
**Classification:** It is used when you need to categorize a certain observation into a group.For example — to predict if a given email is spam or not spam. Each learning example is associated with a qualitative target value, which corresponds to a class (e.g., cancer, healthy). There can be two classes (binary classification) or more (multiclass classification).

**Regression:** It is used for predicting and forecasting, but for continuous values. For example, consider you’re a real estate agent who wants to sell a house that is 2000 sq.feet, has 3 bedrooms, has 5 schools in the area. What price should you sell the house for? 
Each learning example is associated with a quantitative target value (e.g., survial time). The goal of the model is to estimate the correct output, given a feature vector.


![](images/classandregress.png)
<p align="center"> **Figure 2: Classification and Regression. [via: Microbiome]** </p>

### Mechanism of Machine Learning
<div align="center">
![](images/mlmechanism.png)
</div>
<p align="center"> **Figure 3: Machine learning pipeline. [via: Datanami]** </p>

For my project, I built a classification model using Supervised learning. Supervised learning is done is several stages, here they are:

**1.Data preparation & transformation** <br>
The first step is to analyze the data set and prepare it to be used for training purposes. Once data is understood, it is preprocessed and feature transformations is applied to it.

Next, the data is split into two sets — a bigger chunk for training, and the other smaller chunk for testing. The classifier uses the training data-set to “learn”. We need a separate chunk of data for testing and validation, so that we can see how well the model works on data that it hasn’t seen before.<br>

**2.Feature extraction**<br>
The initial set of raw variables is reduced to more manageable groups (features) for processing, while still accurately and completely describing the original data set.

**3.Training**<br>
To train a model, a function is created that internally uses the algorithms of choice, and use the data to train itself and understand patterns, or learn.<br>

**4.Testing and Validation**<br>
Once the model is trained, the nest step is to give it new unseen data, and it’ll should give an output or a prediction.

### About the Dataset
The data set used for this project was gotten from a Kaggle-  web-based data-science environment. The data set contains patients information gotten from 4 hospital databases - Cleveland, Hungary, Switzerland, Virginia. The data contains 76 attributes about the patient and all attributes are numerical, but for my project a subset of 14 of them was used.

Experiments with the data concentrates on simply attempting to distinguish presence (value 1) from absence (value 0).

**Attributes**<br>
1. age,<br>
2. sex,<br>
3. chest pain (cp), <br>
4. resting blood pressure (trestbps), <br>
5. cholesterol level (chol), <br>
6. fasting blood sugar (fbs) , <br>
7. resting electrocardiographic results (restecg), <br>
8. maximum heart rate achieved  (thalach), <br>
9. exercise induced angina (exang), <br>
10. ST depression induced by exercise relative to rest(oldpeak),<br>
11. the slope of the peak exercise ST segment (slope), <br>
12. number of major vessels (0-3) colored by flourosopy (ca),<br> 
13. thallium heart scan  3 = normal; 6 = fixed defect; 7 = reversable defect  (thal), <br>
14. target number (the predicted attribute) - angiographic disease status - presence (1) from absence (0).<br>


**Sample of the unprocessed data**<br>
<br>
<div align="center">
![](images/unprocessed.PNG)
</div>
<p align="center"> **Figure 4: Unprocessed data** </p>

**Sample of the processed data**<br>
<br>
<div align="center">
![](images/processed.PNG)
</div>
<p align="center"> **Figure 5: Processed data** </p>
##### About the machine learning algorithms used
**Gaussian Naive Bayes**: it belongs to a family of algorithms called probabilistic classifiers or conditional probability, where it also assumes independence between features.

**Decision Trees**: a classification algorithm that uses tree-like data structures to model decisions and their possible outcomes.

**Random Forests**: works by using multiple decision trees — using a multitude of different decision trees with different predictions, then combines the results of those individual trees to give the final outcomes. It helps to correct possible overfitting that could occur from decision trees.

**Logistic Regression**:  in statistics used for prediction of outcome of a categorical dependent variable from a set of predictor or independent variables. In logistic regression the dependent variable is always binary.



##### Data Distribution<br>
The dataset comprises of 361 rows of data, average patient age is  55, total numer of women is 120, total number of men is 241, all between ages 28 to 80 years old.
<br>
<div align="center">
![](images/distribution1.PNG)
![](images/distribution2.PNG)

<br>
<br>
There are 176 with no heart disease and 185 at risk of heart disease.
![](images/targetDistribution.PNG)

</div>
<p align="center"> **Figure 6: Output of data distribution** </p>

##### Learning model<br>
I begin by importing all the necessary libraries and then reading the processed heart patient dataset from the csv file and displaying the first 5 rows to make sure the data was read in correctly. 
<div align="center">
![](images/readCSV.PNG)
</div>
<p align="center"> **Figure 7: First five rows of data** </p>

Then I split the data into feature and target set. The first 13 columns are the feature set and the last column has the predicted attribute, so it's the target.

**Features**: These are individual independent variables that act as the input in your system. Prediction models use features to make predictions.

**Target**:  This is the predictor data that can predict with 100% accuracy on future data.<br>

<div align="center">
![](images/feature_target.PNG)
</div>
<p align="center"> **Figure 8: Splitting dataset into feature and target set** </p>

**Classification** -
I use python's sklearn library's train_test_split method to split a bigger chunk of the data into training and test data.

<div align="center">
![](images/train_test.PNG)
</div>
<p align="center"> **Figure 9: Training and Testing data** </p>

Then using a  function [via: Medium] that accepts the Machine Learning algorithm, and the training and testing datasets, I run training and  evaluate the performance of the algorithms using some performance metrics.
The function takes two classification metrics from sklearn library - fbeta_score and accuracy_score, these methods are used to evaluate performance metrics.

``` python
# Import two classification metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict_evaluate(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: quality training set
       - X_test: features testing set
       - y_test: quality testing set
    '''
    
    results = {}
    
    """
    Fit/train the learner to the training data using slicing with 'sample_size' 
    using .fit(training_features[:], training_labels[:])
    """
    start = time() # Get start time of training
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size]) #Train the model
    end = time() # Get end time of training
    
    # Calculate the training time
    results['train_time'] = end - start
    
    """
    Get the predictions on the first 250 training samples(X_train), 
    and also predictions on the test set(X_test) using .predict()
    """
    start = time() # Get start time
    predictions_train = learner.predict(X_train[:288])
    predictions_test = learner.predict(X_test)
    
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 288 training samples which is y_train[:288]
    results['acc_train'] = accuracy_score(y_train[:288], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F1-score on the the first 288 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:288], predictions_train, beta=0.5, average='micro')
        
    # Compute F1-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5, average='micro')
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
```
<br>
Next, in a new function, I initialize all 4 chosen algorithms, and run the training on each of them using the previous function. Then I aggregate and visualize all the results.
<br>

``` python
# Import any three supervised learning classification models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(max_depth=None, random_state=None)
clf_C = RandomForestClassifier(max_depth=None, random_state=None)
clf_D = LogisticRegression()
# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100
samples_100 = len(y_train)
samples_10 = int(len(y_train)*10/100)
samples_1 = int(len(y_train)*1/100)
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C,clf_D]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict_evaluate(clf, samples, X_train, y_train, X_test, y_test)
#print(results)
# Run metrics visualization for the three supervised learning models chosen
vs.visualize_classification_performance(results)
```

##### The evaluation results

<div align="center">
![](images/classifiers.PNG)
![](images/performanceMetrics.PNG)
![](images/Testmetrics.PNG)
</div>
<p align="center">**Figure 9: Performance Evaluation results** </p>

##### Performance metrics<br>

**Accuracy** is by far the simplest and most commonly used performance metric. It is simply the ratio of correct predictions divided by the total data points. 

**Precision** tells us what proportion of messages classified as spam were actually were spam. It is a ratio of true positives (emails classified as spam, and which are actually spam) to all positives (all emails classified as spam, irrespective of whether that was the correct classification). In other words, it is the ratio of True Positives/(True Positives + False Positives).

**Recall** or sensitivity tells us what proportion of messages that actually were spam were classified by us as spam. It is a ratio of true positives (words classified as spam, and which are actually spam) to all the words that were actually spam (irrespective of whether we classified them correctly). It is given by the formula — True Positives/(True Positives + False Negatives)

**F1 Score** is the harmonic average of precision and recall. It’s given by the formula:
![](images/f1score.PNG)

#### Feature importance


The result shows how well the algorithms performed in predictions. The first row shows the performance metrics on the training data, and the second row shows the metrics for the testing data (data which hasn’t been seen before).
From these results we can see that RandomForest & DecisionTree Classifier performed better than GaussianNB and Logistic Regression.

Using RandomForest classifier's .feature_importance attribute I view the importance of each feature by its relative ranks when making predictions.
The graph shows the five most important features that determine is a patient is at risk for heart disease.

``` python
# Import a supervised learning model that has 'feature_importances_'
model = RandomForestClassifier(max_depth=None, random_state=None)
# Train the supervised model on the training set using .fit(X_train, y_train)
model = model.fit(X_train, y_train)
# Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_
print(X_train.columns)
print(importances)
# Plot
vs.feature_plot(importances, X_train, y_train)
```
<br>

<div align="center">

**Feature Importance Result**

![](images/feature_importance.PNG)
</div>
<p align="center">**Figure 10: Feature importance in patients data** </p> 

#### confusion matrix

An error matrix, also known as a confusion matrix, is a specific table layout that allows visualization of the performance of an algorithm. So I imported confusion matrix mretrics from sklearn library, and ran this metric on all 4 algorithms.

**Attributes**

**True positive (TP)**
eqv. with hit

**True negative (TN)**
eqv. with correct rejection

**False positive (FP)**
eqv. with false alarm, Type I error

**False negative (FN)**
eqv. with miss, Type II error

##### Here are the results

**Logistic Regression**:

The confusion matrix shows 24(true positives) +28(true negatives) = 52 correct predictions and 13(false positives)+8(false negatives) = 21 incorrect ones from the total 73 test data.
<div align="center">
![](images/logic_cm.PNG)
</div>
<p></p>

**Gaussian Naive Bayes**

The confusion matrix shows 20(true positives)+32(true negatives) = 52 correct predictions and 17(false positives)+4(false negatives) = 21 incorrect ones from the total 73 test data.
<div align="center">
![](images/gaussian_cm.PNG)
</div>
<p></p>

**Decision Tree**

The confusion matrix shows 32(true positives)+27(true negatives) = 59 correct predictions and 9(false positives)+5(false negatives) = 14 incorrect ones from the total 73 test data.

<div align="center">
![](images/decision_cm.PNG)
</div>
<p></p>

**RandomForest**

The confusion matrix shows 31(true positives)+25(true negatives) = 56 correct predictions and 11(false positives)+6(false negatives) = 17 incorrect ones from the total 73 test data.

<div align="center">
![](images/random_cm.PNG)
</div>
<p></p>

Once again, it shows that the Decision Tree and Random Forest classifiers performed better; making higher correct predictions that the Gaussian Naives Bayes and Logistic Regression classifiers.

##### hyperparameters
 In machine learning, a hyperparameter is a special type of configuration variable whose value cannot be directly calculated using the data-set. So in order to figure out the optimal values of these hyperparameters, this process is called hyperparameter tuning.
 Luckily, sklearn's GridSearchCV API would train and run cross validation on all possible combinations of these hyperparameters and give us the optimum configuration. Function code [via: Medium]
###### Hyperparamter tuning on Random Forest Classifier
 <br>


 ``` python
 # TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = RandomForestClassifier(max_depth=None, random_state=None)

# Create the parameters or base_estimators list you wish to tune, using a dictionary if needed.
# Example: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

"""
n_estimators: Number of trees in the forest
max_features: The number of features to consider when looking for the best split
max_depth: The maximum depth of the tree
"""
parameters = {'n_estimators': [10, 20, 30], 'max_features':[3,4,5, None], 'max_depth': [5,6,7, None]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5, average="micro")

# TODO: Perform grid search on the claszsifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average="micro")))
print("\nOptimized Model\n------")
print(best_clf)
print("\nFinal accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5,  average="micro")))
```

###### hyperparameter output
![](images/hyper_output.PNG)

As you can see, after the hyperparamter tuning, the score accuracy score improved from 79.4% to 80.8%

##### Finally, testing the learning model for prediction

I gave it a bunch of values for various features, using unseen data from the larger data set and it predicted 6/8 correctly.
<br>

![](images/testing_output.PNG)

#### Conclusion

Overall model could be improved with more data and more machine learning algorithms can be compared to improve this project.The model predicted with 0.80 accuracy. The model is more specific than sensitive.

#### References
1. Turaga, Deepak S., and Michael Schmidt. “Mining of Sensor Data in Healthcare: A Survey.” SpringerLink, Springer, 1 Jan. 1970, link.springer.com/chapter/10.1007/978-1-4614-6309-2_14.
2. Hindawi. “Mobile Information Systems.” Advances in Decision Sciences, Hindawi, www.hindawi.com/journals/misy/si/287385/cfp/. 
3. An Introduction to Biometric Recognition - IEEE Journals & Magazine, Wiley-IEEE Press, ieeexplore.ieee.org/document/6864376.
4. UCI Machine Learning Repository: Flags Data Set, archive.ics.uci.edu/ml/datasets/Heart Disease.
5. “How to Make Predictions with Scikit-Learn.” Machine Learning Mastery, 5 Apr. 2018, machinelearningmastery.com/make-predictions-scikit-learn/.
6. “Microbiome Summer School 2017.” Microbiome Summer School 2017 by aldro61, aldro61.github.io/microbiome-summer-school-2017/sections/basics/.
7. Sifium. “Types of Classification Algorithms in Machine Learning.” Medium.com, Medium, 28 Feb. 2017, medium.com/@sifium/machine-learning-types-of-classification-9497bd4f2e14.
8. “How to Build a Better Machine Learning Pipeline.” Datanami, 5 Sept. 2018, www.datanami.com/2018/09/05/how-to-build-a-better-machine-learning-pipeline/.
9. 10201378713505824. “How to Use Machine Learning to Predict the Quality of Wines.” FreeCodeCamp.org, FreeCodeCamp.org, 7 Feb. 2018 medium.freecodecamp.org/using-machine-learning-to-predict-the-quality-of-wines-9e2e13d7480d.
10. “What Is Machine Learning? | How It Works, Techniques & Applications.” Reconstructing an Image from Projection Data - MATLAB & Simulink Example, www.mathworks.com/discovery/machine-learning.html.
11. RSNA Pneumonia Detection Challenge | Kaggle, www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression.
12. “Heart Disease Prediction System in Python Using Support Vector Machine and PCA | Machine Learning.” Artificial Intelligence Videos, 2 Nov. 2016, www.artificial-intelligence.video/heart-disease-prediction-system-in-python-using-support-vector-machine-and-pca-machine-learning.
13. Ronit. “Heart Disease UCI.” RSNA Pneumonia Detection Challenge | Kaggle, 25 June 2018, www.kaggle.com/ronitf/heart-disease-uci.