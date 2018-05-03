from surprise.model_selection import KFold
from source.code.metrics import precision_recall_at_k
from datetime import datetime
from six import iteritems

import numpy as np


def my_cross_validation(algo, data, k=5, threshold=7, n_splits=5, verbose=False):
    kf = KFold(n_splits=n_splits)
    cv_map = {'map@{}'.format(k): [], 'mar@{}'.format(k): []}
    time_map = {'Fit time': [], 'Test time': []}
    for trainset, testset in kf.split(data):
        step_one = datetime.now()
        algo.fit(trainset)
        step_two = datetime.now()
        predictions = algo.test(testset)
        step_three = datetime.now()
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
        cv_map['map@{}'.format(k)].append(sum(precisions.values()) / len(precisions))
        cv_map['mar@{}'.format(k)].append(sum(recalls.values()) / len(recalls))
        time_map['Fit time'].append((step_two - step_one).total_seconds())
        time_map['Test time'].append((step_three - step_two).total_seconds())
    if verbose:
        print_summary(
            algo,
            ['map@{}'.format(k), 'mar@{}'.format(k)],
            cv_map,
            time_map['Fit time'],
            time_map['Test time'],
            n_splits
        )
    cv_map.update(time_map)
    return cv_map


def print_summary(algo, measures, test_measures, fit_times, test_times, n_splits):
    print('Evaluating {0} of algorithm {1} on {2} split(s).'.format(', '.join((m.upper() for m in measures)), algo.__class__.__name__, n_splits))
    print()
    row_format = '{:<18}' + '{:<8}' * (n_splits + 2)
    s = row_format.format('', *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] + ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(key.upper() + ' (testset)', *['{:1.4f}'.format(v) for v in vals] + ['{:1.4f}'.format(np.mean(vals))] + ['{:1.4f}'.format(np.std(vals))]) for (key, vals) in iteritems(test_measures))
    s += '\n'
    s += row_format.format('Fit time', *['{:.2f}'.format(t) for t in fit_times] + ['{:.2f}'.format(np.mean(fit_times))] + ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time', *['{:.2f}'.format(t) for t in test_times] + ['{:.2f}'.format(np.mean(test_times))] + ['{:.2f}'.format(np.std(test_times))])
    print(s)
