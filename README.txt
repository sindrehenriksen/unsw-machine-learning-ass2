main.py contains code for running grid-searches over learning 
algorithm parameters and running the learning algorithms on the
training and test data provided in the files with corresponding
names, obtained May 2018 from 
https://www.kaggle.com/c/bioresponse#Evaluation. Different
sections of the code are placed in different if True/False blocks
for easy commenting in/out. Random state is set everywhere for
reproducibility.

classifier_functions.py contains shell functions around scikit-
learn functions implementing different learning algorithms and
around scikit-learn's GridSearchCV(). This is done for easy
running of code and for storing of prefered parameter values and 
search grids. A function to print top performing parameters and 
saving variables used for later inspection/plotting is also
included.

plotting.py contains plotting functions and code to make the
plots, using the same style of if True/False blocks as in
main.py.

perceptron.py contains code for experimentation with a
perceptron.

Directories: grid_search_results contains saved results from
grid-searches and plots. learning_curves contains plots.
predictions contains the predictions from models ran on the
test set, in the format required to upload to the Kaggle and
obtain results (https://www.kaggle.com/c/bioresponse/data).
It also contains a svm benchmark file provided in the
competition.
