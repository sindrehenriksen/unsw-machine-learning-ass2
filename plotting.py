import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from classifier_functions import mlp, svm, rf, gb
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss


def load_list(filename):
    f = open(filename, 'rb')
    d = pickle.load(f)
    f.close()
    return d


def plot_param_grid_scores(learning_algorithm, save=True):
    results = load_list('grid_search_results/' + learning_algorithm + '.pkl')
    score = [tup[0] for tup in results]
    train_score = [tup[1] for tup in results]
    time = [tup[3] for tup in results]
    params = [tup[4] for tup in results]
    param_vals_grid = [list(d.values()) for d in params]
    param_keys = list(params[0].keys())

    for i in range(len(params[0])):
        param = [param_vals[i] for param_vals in param_vals_grid]
        fig = plt.figure(i)
        ax = fig.gca()
        not_min = [i for i, x in enumerate(param) if param[i] != min(param)]
        if type(param[i]) is np.float64 and min([param[i] for i in not_min])/min(param) == 10:
            ax.set_xscale('log')

        # Time as axis and no training score
        # ax.scatter(param, score, label='Score', c=time)
        # ax.set_xlabel(param_keys[i])
        # ax.set_ylabel('Mean test score')
        # ax.legend(loc='lower left', bbox_to_anchor=(0, 1))
        # ax_time = ax.twinx()
        # ax_time.scatter(param, time, label='Time (s)', color='y', marker='v')
        # ax_time.set_ylabel('Mean fit time (s)')
        # ax_time.legend(loc='lower right', bbox_to_anchor=(1, 1))

        plt.scatter(param, score, label='Test', c=time, cmap='coolwarm', alpha=0.8)
        plt.scatter(param, train_score, label='Training', c=time, cmap='coolwarm', alpha=0.8, marker='+')
        plt.xlabel(param_keys[i])
        plt.ylabel('Mean score')
        plt.legend()
        cbar = plt.colorbar()
        cbar.set_label('Time (s)')
        ax.yaxis.grid(True)
        plt.title(learning_algorithm)
        plt.show()

        if save:
            fig.savefig('grid_search_results/' + learning_algorithm + '_' + param_keys[i] + '.pdf', bbox_inches='tight')


# Code for the following function based on (May 2018):
# http://scikit-learn.orgc/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='neg_log_loss'):
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()
    return fig


if __name__ == '__main__':
    # Plot parameter scores
    if 0:
        # Multi-layer perceptron
        plot_param_grid_scores('mlp')
        mlp_results = load_list('grid_search_results/mlp.pkl')
        for tup in mlp_results:
            print(tup)

        # Support vector machines
        plot_param_grid_scores('svm')
        svm_results = load_list('grid_search_results/svm.pkl')
        for tup in svm_results:
            print(tup)

        # Random forest
        plot_param_grid_scores('rf')
        rf_results = load_list('grid_search_results/rf.pkl')
        for tup in rf_results:
            print(tup)

        # Gradient boosting
        plot_param_grid_scores('gb')
        gb_results = load_list('grid_search_results/gb.pkl')
        for tup in gb_results:
            print(tup)

        # Logreg bagging
        plot_param_grid_scores('bagging')
        bagging_results = load_list('grid_search_results/bagging.pkl')
        for tup in bagging_results:
            print(tup)

    data = False

    # Learning curves (takes time running)
    if 0:
        data = True
        f = open('train.csv')
        f.readline()  # skip column names
        train_data = np.loadtxt(f, delimiter=',')
        X = train_data[:, 1:]
        X_sparse = csr_matrix(X)
        y = train_data[:, 0]
        f.close()

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        clf_mlp = mlp(X_sparse, y, random_state=0)
        fig = plot_learning_curve(clf_mlp, 'MLP learning curves', X, y, ylim=(-0.7, -0.09), cv=kf)
        fig.savefig('learning_curves/mlp.pdf', bbox_inches='tight')

        clf_svm = svm(X_sparse, y, probability=True, random_state=0)
        fig = plot_learning_curve(clf_svm, 'SVM learning curves', X, y, ylim=(0.7, 1.01), cv=kf, scoring=None)
        fig.savefig('learning_curves/svm.pdf', bbox_inches='tight')

        clf_rf = rf(X_sparse, y, random_state=0)
        fig = plot_learning_curve(clf_rf, 'Random forest learning curves', X, y, ylim=(-0.7, -0.09), cv=kf)
        fig.savefig('learning_curves/rf.pdf', bbox_inches='tight')

        clf_gb = gb(X_sparse, y, random_state=0)
        fig = plot_learning_curve(clf_gb, 'Gradient boosting learning curves', X, y, ylim=(-0.7, -0.09), cv=kf)
        fig.savefig('learning_curves/gb.pdf', bbox_inches='tight')

    # MLP error vs weight updates
    if 0:
        if not data:
            data = True
            f = open('train.csv')
            f.readline()  # skip column names
            train_data = np.loadtxt(f, delimiter=',')
            X = train_data[:, 1:]
            y = train_data[:, 0]
            f.close()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
        X_train_sparse = csr_matrix(X_train)
        points = 100
        it = 600
        small_network = True

        if small_network:
            hidden_layer_sizes = (50,)
        else:
            hidden_layer_sizes = (20, 20)

        clf_mlp_convergence = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=1e-1, learning_rate_init=1e-5,
                                            max_iter=1, tol=1e-8, warm_start=True, random_state=0)
        ll_train, ll_test = (np.empty(points), np.empty(points))
        for i in range(points):
            for _ in range(it//points):
                clf_mlp_convergence.fit(X_train_sparse, y_train)
            ll_train[i] = log_loss(y_train, clf_mlp_convergence.predict_proba(X_train))
            ll_test[i] = log_loss(y_val, clf_mlp_convergence.predict_proba(X_val))
        assert(clf_mlp_convergence.n_iter_==it)
        fig = plt.figure()
        plt.scatter(np.linspace(it//points, it, points), ll_train, label='Training set', marker='+', color='r')
        plt.scatter(np.linspace(it//points, it, points), ll_test, label='Test set',  marker='+', color='g')
        plt.grid()
        plt.xlabel('Number of weight updates')
        plt.ylabel('Log-loss')
        plt.legend()
        plt.ylim([0.33, 0.75])
        plt.show()

        if small_network:
            fig.savefig('learning_curves/mlp_convergence_small.pdf', bbox_inches='tight')
        else:
            fig.savefig('learning_curves/mlp_convergence_big.pdf', bbox_inches='tight')
