import matplotlib.pyplot as plt
import numpy as np

MEDIUM_SIZE = 40
BIGGER_SIZE = 50

plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rc('text', color='green')

plt.rc('xtick', color='green')
plt.rc('ytick', color='green')

plt.rc('axes', labelcolor='green')
plt.rc('axes', labelcolor='green')


def plot_results(cv_s, labels, folds):
    f, axs = plt.subplots(1, len(cv_s), figsize=(50, 15))
    for i, ax in enumerate(axs):
        ax.plot(cv_s[0][list(cv_s[0].keys())[i]], label=labels[0])
        ax.plot(cv_s[1][list(cv_s[1].keys())[i]], label=labels[1])
        ax.plot(cv_s[2][list(cv_s[2].keys())[i]], label=labels[2])
        ax.plot(cv_s[3][list(cv_s[3].keys())[i]], label=labels[3])
        ax.set_title('{} on {}-Fold cv'.format(list(cv_s[0].keys())[i], str(len(folds))))
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.legend()
        ax.grid(True)
    plt.setp(axs, xticks=range(len(folds)), xticklabels=folds)
    plt.tight_layout()
    plt.show()


def plot_mean_results(cv_s, labels):
    keys = list(cv_s[0].keys())
    f, axs = plt.subplots(1, len(keys), figsize=(50, 15))
    for i, ax in enumerate(axs):
        ax.errorbar(range(len(cv_s)), [np.mean(cv[keys[i]]) for cv in cv_s], [np.std(cv[keys[i]]) for cv in cv_s], linestyle='None', marker='o')
        ax.set_title('Mean {} with std'.format(keys[i]))
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    plt.setp(axs, xticks=range(len(labels)), xticklabels=labels)
    plt.tight_layout()
    plt.show()