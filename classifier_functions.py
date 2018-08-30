import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle


def mlp_grid_search_cv(X_train, y_train, max_it=1000, folds=5, param_grid=None, random_state=None,
                       save_filename=None):
    # Default param grid
    if param_grid is None:
        param_grid = {'hidden_layer_sizes': ((5, 5, 5), (5, 5), (10, 10), (20, 20), (10,), (20,), (50,), (100,)),
                      'activation': ('tanh', 'relu'),
                      'alpha': np.logspace(-1, -5, 5),
                      'learning_rate_init': np.logspace(-3, -5, 3)}

    # Create model
    clf = MLPClassifier(max_iter=max_it, random_state=random_state)

    # Run grid-search
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf_grid = GridSearchCV(clf, param_grid=param_grid, scoring='neg_log_loss', cv=kf, verbose=2)
    clf_grid = clf_grid.fit(X_train, y_train)

    # Print best parameters and scores
    print_best_params(clf_grid, save_filename)

    return clf_grid


def mlp(X_train, y_train, hidden_layer_sizes=(50,), alpha=1e-1, learning_rate_init=1e-5, max_it=2000,
        random_state=None):
    # Create and fit model
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, learning_rate_init=learning_rate_init,
                        max_iter=max_it, random_state=random_state)
    clf.fit(X_train, y_train)

    return clf


def svm_grid_search_cv(X_train, y_train, max_it=-1, folds=5, cache_size=1000, param_grid=None, random_state=None,
                       save_filename=None):
    # Default param grid
    if param_grid is None:
        param_grid = {'C': np.logspace(-1, 3, 5),
                      'gamma': np.logspace(0, -3, 4)}

    # Create model
    clf = SVC(max_iter=max_it, random_state=random_state, cache_size=cache_size)

    # Grid-search
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=kf, verbose=2)
    clf_grid = clf_grid.fit(X_train, y_train)

    # Print best parameters and scores
    print_best_params(clf_grid, save_filename)

    return clf_grid


def svm(X_train, y_train, kernel='rbf', degree=3, coef0=1, C=10, gamma=1e-2, max_it=-1,
        probability=False, cache_size=1000, random_state=None):
    # Create and fit model
    clf = SVC(kernel=kernel, degree=degree, coef0=coef0, C=C, gamma=gamma, max_iter=max_it,
              probability=probability, cache_size=cache_size, random_state=random_state)
    clf.fit(X_train, y_train)

    return clf


def dt_grid_search_cv(X_train, y_train, folds=5, param_grid=None, random_state=None, save_filename=None):
    # Default param grid
    if param_grid is None:
        param_grid = {'max_depth': np.arange(2, 6, 1),
                      'min_samples_leaf': np.arange(10, 31, 10),
                      'max_features': np.arange(80, 121, 20)}

    # Create model
    clf = DecisionTreeClassifier(random_state=random_state)

    # Grid-search
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf_grid = GridSearchCV(clf, param_grid=param_grid, scoring='neg_log_loss', cv=kf, verbose=2)
    clf_grid = clf_grid.fit(X_train, y_train)

    # Print best parameters and scores
    print_best_params(clf_grid, save_filename)

    return clf_grid


def dt(X_train, y_train, max_depth=3, min_samples_leaf=20, max_features=100, random_state=None):
    # Create and fit model
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features,
    random_state=random_state)
    clf.fit(X_train, y_train)

    return clf


def rf_grid_search_cv(X_train, y_train, folds=5, param_grid=None, random_state=None, min_samples_split=15,
                      max_features=100, bootstrap=False, save_filename=None):
    # Default param grid
    if param_grid is None:
        param_grid = {'max_depth': np.arange(13, 20, 3),
                      'min_samples_leaf': np.arange(2, 10, 3),
                      'n_estimators': [550, 580, 600]}

    # Create model
    clf = RandomForestClassifier(min_samples_split=min_samples_split, max_features=max_features,
                                 bootstrap=bootstrap, random_state=random_state)

    # Grid-search
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf_grid = GridSearchCV(clf, param_grid=param_grid, scoring='neg_log_loss', cv=kf, verbose=2)
    clf_grid = clf_grid.fit(X_train, y_train)

    # Print best parameters and scores
    print_best_params(clf_grid, save_filename)

    return clf_grid


def rf(X_train, y_train, min_samples_split=15, max_depth=13, min_samples_leaf=2, n_estimators=550, max_features=100,
       bootstrap=False, random_state=None):
    # Create and fit model
    clf = RandomForestClassifier(min_samples_split=min_samples_split, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, n_estimators=n_estimators,
                                 max_features=max_features, bootstrap=bootstrap, random_state=random_state)
    clf.fit(X_train, y_train)

    return clf


def gb_grid_search_cv(X_train, y_train, folds=5, param_grid=None, n_estimators=300, max_depth=5, random_state=None,
                      save_filename=None):
    # Default param grid
    if param_grid is None:
        param_grid = {
           'min_samples_leaf': np.arange(40, 61, 10),
           'max_features': np.arange(20, 61, 20),
           'subsample': np.arange(0.8, 1.05, 0.1),
          }

    # Create model
    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Grid-search
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf_grid = GridSearchCV(clf, param_grid=param_grid, scoring='neg_log_loss', cv=kf, verbose=2)
    clf_grid = clf_grid.fit(X_train, y_train)

    # Print best parameters and scores
    print_best_params(clf_grid, save_filename)

    return clf_grid


def gb(X_train, y_train, n_estimators=300, max_depth=5, subsample=0.9, max_features=40, min_samples_leaf=50,
       random_state=None):
    # Create and fit model
    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, subsample=subsample,
                                     max_features=max_features, min_samples_leaf=min_samples_leaf,
                                     random_state=random_state)
    clf.fit(X_train, y_train)

    return clf


def logreg_grid_search_cv(X_train, y_train, max_it=100, folds=5, param_grid=None, random_state=None,
                          save_filename=None):
    # Default param grid
    if param_grid is None:
        param_grid = {'penalty': ('l2', 'l1'),
                      'C': np.logspace(-3, 3, 7),
                      'tol': np.logspace(-2, -6, 5)}

    # Create model
    log_reg = LogisticRegression(max_iter=max_it, random_state=random_state)

    # Grid-search
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf_grid = GridSearchCV(log_reg, param_grid=param_grid, scoring='neg_log_loss', cv=kf, verbose=2)
    clf_grid = clf_grid.fit(X_train, y_train)

    # Print best parameters and scores
    print_best_params(clf_grid, save_filename)

    return clf_grid


def logreg(X_train, y_train, penalty='l2', C=1e3, tol=10e-4, max_it=500, random_state=None):
    # Create model
    clf = LogisticRegression(penalty=penalty, C=C, tol=tol, max_iter=max_it, random_state=random_state, solver='lbfgs')
    clf.fit(X_train, y_train)

    return clf


def print_best_params(clf_grid, save_filename):
    print('Mean test score, mean train score, rank, mean fit time, parameters')
    params = clf_grid.cv_results_['params']
    ranks = clf_grid.cv_results_['rank_test_score']
    test_scores = clf_grid.cv_results_['mean_test_score']
    train_scores = clf_grid.cv_results_['mean_train_score']
    times = clf_grid.cv_results_['mean_fit_time']
    results = []
    for tup in [(test_scores[i], train_scores[i], ranks[i], times[i], params[i]) for i in range(len(params))]:
        results.append(tup)
    results.sort(key=lambda tup: tup[2])
    for i in range(min(len(results), 10)):
        print(results[i])

    if save_filename is not None:
        f = open(save_filename, 'wb')
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        f.close()
