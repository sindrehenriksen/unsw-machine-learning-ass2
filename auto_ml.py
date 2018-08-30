import numpy as np
from scipy.sparse import csr_matrix
import autosklearn.classification
from autosklearn.metrics import log_loss

# Read training data
f = open('train.csv')
f.readline()    # skip column names
train_data = np.loadtxt(f, delimiter=',')
X = train_data[:, 1:]
X_sparse = csr_matrix(X)
y = train_data[:, 0]
f.close()

# Read test data
f = open('test.csv')
f.readline()    # skip column names
X_test = np.loadtxt(f, delimiter=',')
f.close()

# Learn model
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=36000, per_run_time_limit=600)
cls.fit(X_sparse, y, metric=log_loss)

# Predict and save
y_pred = cls.predict_proba(X_test)[:, 1]
np.savetxt('predictions/auto_clf.csv', list(enumerate(y_pred, 1)), fmt=['%i ', '%.10f'], delimiter=',',
           header='MoleculeId,PredictedProbability', comments='')
