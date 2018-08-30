from classifier_functions import mlp_grid_search_cv, mlp, svm_grid_search_cv, svm, rf_grid_search_cv, rf,\
    gb_grid_search_cv, gb, logreg_grid_search_cv, logreg
from perceptron_cv import perceptron_cross_validation
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

# Read training data
f = open('train.csv')
f.readline()    # skip column names
train_data = np.loadtxt(f, delimiter=',')
X = train_data[:, 1:]
X_sparse = csr_matrix(X)
y = train_data[:, 0]
f.close()

# Run grid-searches (takes time running)
if 0:
    perceptron_clf = perceptron_cross_validation(X, y)
    mlp_clf_grid = mlp_grid_search_cv(X_sparse, y, save_filename='grid_search_results/mlp.pkl', random_state=0,
                                      param_grid={'hidden_layer_sizes': ((20,), (50,), (100,)),
                                                  'alpha': np.logspace(-1, -5, 5),
                                                  'learning_rate_init': np.logspace(-3, -5, 3)})
    svm_clf_grid = svm_grid_search_cv(X_sparse, y, save_filename='grid_search_results/svm.pkl', random_state=0)
    rf_clf_grid = rf_grid_search_cv(X_sparse, y, save_filename='grid_search_results/rf.pkl', random_state=0)
    gb_clf_grid = gb_grid_search_cv(X_sparse, y, save_filename='grid_search_results/gb.pkl', random_state=0)

# Logreg bagging with validation
if 0:
    models = 2
    folds = 10

    seeds = list(range(folds))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=seeds[0])
    X_train_sparse = csr_matrix(X_train)
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seeds[0])

    bag_X_train = np.empty((len(X_train), models))
    bag_X_val = np.zeros((len(X_val), models))

    for i, (fold_train, fold_test) in enumerate(kf.split(X_train, y_train)):
        print('Fold', i + 1)

        # Fold training set
        X_fold_train = X_train_sparse[fold_train]
        y_fold_train = y_train[fold_train]
        X_fold_test = X_train[fold_test]
        y_fold_test = y_train[fold_test]

        # Classifiers
        rf_clf = rf(X_fold_train, y_fold_train, random_state=seeds[i])
        gb_clf = gb(X_fold_train, y_fold_train, random_state=seeds[i])

        # Array of predictions from classifiers
        bag_X_train[fold_test] = np.stack((rf_clf.predict_proba(X_fold_test)[:, 1],
                                           gb_clf.predict_proba(X_fold_test)[:, 1],
                                           ), 1)

        # Array of predictions for validation set
        bag_X_val += np.stack((rf_clf.predict_proba(X_val)[:, 1],
                               gb_clf.predict_proba(X_val)[:, 1],
                               ), 1)

        print('Log-loss model predictions:', log_loss(y_val, rf_clf.predict_proba(X_val)[:, 1]),
              log_loss(y_val, gb_clf.predict_proba(X_val)[:, 1]))

    # Mean of predictions
    bag_X_val /= folds
    for i in range(models):
        print('Log-loss average model', i,  'predictions:', log_loss(y_val, bag_X_val[:, i]))

    # Run grid-search on logistic regression
    bagging_clf_grid = logreg_grid_search_cv(bag_X_train, y_train, save_filename='grid_search_results/bagging.pkl',
                                             random_state=0)

    # Run logistic regression on classifier predictions
    bagging_clf = logreg(bag_X_train, y_train, random_state=seeds[0])
    y_val_pred = bagging_clf.predict_proba(bag_X_val)
    ll = log_loss(y_val, y_val_pred)
    print('Log-loss after bagging using logistic regression:', ll)

# Run learners on entire dataset
f = open('test.csv')
f.readline()    # skip column names
X_test = np.loadtxt(f, delimiter=',')

# "Simple" learners
if 0:
    y_pred_mlp = mlp(X_sparse, y, random_state=0).predict_proba(X_test)[:, 1]
    np.savetxt('predictions/predictions_mlp.csv', list(enumerate(y_pred_mlp, 1)), fmt=['%i ', '%.10f'], delimiter=',',
               header='MoleculeId,PredictedProbability', comments='')
    y_pred_svm = svm(X_sparse, y, probability=True, random_state=0).predict_proba(X_test)[:, 1]
    np.savetxt('predictions/predictions_svm.csv', list(enumerate(y_pred_svm, 1)), fmt=['%i ', '%.10f'], delimiter=',',
               header='MoleculeId,PredictedProbability', comments='')
    y_pred_rf = rf(X_sparse, y, random_state=0).predict_proba(X_test)[:, 1]
    np.savetxt('predictions/predictions_rf.csv', list(enumerate(y_pred_rf, 1)), fmt=['%i ', '%.10f'], delimiter=',',
               header='MoleculeId,PredictedProbability', comments='')
    y_pred_gb = gb(X_sparse, y, random_state=0).predict_proba(X_test)[:, 1]
    np.savetxt('predictions/predictions_gb.csv', list(enumerate(y_pred_gb, 1)), fmt=['%i ', '%.10f'], delimiter=',',
               header='MoleculeId,PredictedProbability', comments='')
    rf_benchmark = RandomForestClassifier(n_estimators=100, min_samples_split=2)
    rf_benchmark.fit(X_sparse, y)
    y_pred_rf_benchmark = rf_benchmark.predict_proba(X_test)[:, 1]
    np.savetxt('predictions/predictions_rf_benchmark.csv', list(enumerate(y_pred_rf_benchmark, 1)), fmt=['%i ', '%.10f'],
               delimiter=',', header='MoleculeId,PredictedProbability', comments='')

# Logreg bagging (stacking is the right word)
if 1:
    models = 2
    folds = 10

    seeds = list(range(folds))  # [None]*folds
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seeds[0])

    bag_X = np.empty((len(X), models))
    bag_X_test = np.zeros((len(X_test), models))

    for i, (fold_train, fold_test) in enumerate(kf.split(X, y)):
        print('Fold', i + 1)

        # Fold training set
        X_fold_train = X_sparse[fold_train]
        y_fold_train = y[fold_train]
        X_fold_test = X[fold_test]

        # Classifiers
        rf_clf = rf(X_fold_train, y_fold_train, random_state=seeds[i])
        gb_clf = gb(X_fold_train, y_fold_train, random_state=seeds[i])

        # Array of predictions from classifiers
        bag_X[fold_test] = np.stack((rf_clf.predict_proba(X_fold_test)[:, 1],
                                           gb_clf.predict_proba(X_fold_test)[:, 1],
                                           ), 1)

        # Array of predictions for test set
        bag_X_test += np.stack((rf_clf.predict_proba(X_test)[:, 1],
                                gb_clf.predict_proba(X_test)[:, 1],
                                ), 1)

        # Try cross-blending to avoid data leakage

    # Mean of predictions
    bag_X_test /= folds

    # Run logistic regression on classifier predictions
    bagging_clf = logreg(bag_X, y, random_state=seeds[0])

    # Predict and save output
    y_pred = bagging_clf.predict_proba(bag_X_test)[:, 1]
    np.savetxt('predictions/predictions_bagging.csv', list(enumerate(y_pred, 1)), fmt=['%i ', '%.10f'], delimiter=',',
               header='MoleculeId,PredictedProbability', comments='')
