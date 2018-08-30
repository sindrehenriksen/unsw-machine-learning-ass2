import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, StratifiedKFold


def perceptron_cross_validation(train_x, train_y, max_it=50, folds=5):
    # Create models
    clf = (
        Perceptron(max_iter=max_it, penalty=None),
        Perceptron(max_iter=max_it, penalty='l2'),
        Perceptron(max_iter=max_it, penalty='l1'),
        Perceptron(max_iter=max_it, penalty='elasticnet')
    )

    # Run cross-validation
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=10)   # random_state for reproducible results
    scores = np.empty((4, folds))
    for i in range(len(clf)):
        scores[i] = cross_val_score(clf[i], train_x, train_y, cv=kf)

    # Print cross-validation scores
    print(np.round(np.mean(scores, axis=1), 4))

    return clf
