This is part of my Learning Journey as an aspiring Machine Learning Engineering.
Here, I did a regression analysis of the Life Expectancy Dataset on Kaggle from the
World Health Organization (WHO).

I tried to predict Life Expectancy in years using Regression models such as Linear Regression,
Decision Trees, and Random Forest Regression.

Initially the models where split with 80-20% for training and testing sets.
The models are then trained and evaluated using the Test Set. With Decision Treees achieving 
a 0.0 Root Mean Squared Error. Firther investigation was done on the models by implementing cross
validation with K-Fold Cross Validation with 10 splits. Results have shown that Decision Trees 
show overfitting with a large training-to-k-fold gap (0.00 vs. 2.75, std 0.24). Linear Regression remains
stable (3.82 vs. 3.87, difference 0.05, std 0.26), though its higher std suggests more variability. Random 
Forest maintains a moderate gap (0.70 vs. 1.91, difference 1.21, std 0.23), with the lowest std, 
indicating good stability and low error. Therefore Random Forest was the chosen model.

After model selection, Random Forest Regressor underwent hyperparameter tuning using GridSearch to optimize the
model, where a parameter grid was defined with two sets of hyperparameters: the first set included n_estimators 
values of 3, 10, and 30, along with max_features values of 2, 4, 6, 8, 12, 14, 16, 18, 20, and 22, while the second
set set bootstrap to False, n_estimators to 3 and 10, and max_features to 2, 3, and 4. The GridSearchCV was 
configured with the Random Forest estimator, the parameter grid to test all combinations, 10-fold cross-validation 
to assess performance, scoring based on negative mean squared error where lower values are better, and n_jobs=-1 to
utilize all available CPU cores for parallel processing. Finally, the grid search was fitted to the transformed 
training data and target variable to evaluate all parameter combinations and select the best-performing 
model based on the cross-validation score. Results show that the parameters that optimizes performance well were
'max_features' =14 and  'n_estimators' =  30. With these parameters, the training RMSE were reduced to 1.90 from
1.91 pre-tuning. A small improvement.

Finally the post-tuned Random Forest Regressor was then evaluated on the test with RMSE and R-squared value of
1.85 and 0.963. Showing that the model was able to predict Life Expectancy with a relatively small error of 1.85 and
capturing most of the variations in the test set.

**FURTHER IMPROVEMENT**
As I progress into my journey of studying Machine Learning and AI, i have encountered concepts such as Learning Curves.
These surves helps assess how well a model learns and generalizes, typically indicating whether it suffers from underfitting 
(poor performance even with more data) or overfitting (excellent training performance but poor generalization). 
The curve is plotted with training data size or epochs on the x-axis and the performance metric (e.g., error or score) 
on the y-axis, often comparing training and validation performance to diagnose model behavior. If I were to improve this model,
I would implement Learning Curves to test if the models were overfitting or underfitting rather than just plain comparison
of the test error vs cross-validation error. I also would have tested SGDregressor and Support Vector Regressors and even used
Polynomial Features to capture non-linearities in the Data.

