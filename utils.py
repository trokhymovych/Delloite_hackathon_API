import random

import numpy as np


def get_kfold_with_seacrh_criteria(df, n_splits=5, shuffle=True, silent=False):
    df['full_criterion'] = df['accepted_function'] + df['rejected_function'] + df['accepted_product'] + \
                           df['rejected_product']
    criterions = df['full_criterion'].unique()
    n_criterions = len(criterions)
    crit_in_split = n_criterions // n_splits

    split_idx = []
    for i in range(crit_in_split, n_criterions, crit_in_split):
        split_idx.append(i)
    if split_idx[len(split_idx) - 1] != n_criterions:
        split_idx[len(split_idx) - 1] = n_criterions
    split_idx = np.array(split_idx)

    df['fold'] = None
    criterions = list(criterions)
    for i in range(len(df['full_criterion'])):
        df['fold'][i] = sum(split_idx < criterions.index(df['full_criterion'][i]))

    splt_idx = []
    for i in range(n_splits):
        test_idx = df.index[df['fold'] == i].tolist()
        train_idx = df.index[df['fold'] != i].tolist()

        if shuffle:
            test_idx = random.sample(test_idx, len(test_idx))
            train_idx = random.sample(train_idx, len(train_idx))

        if not silent:
            print('\nSplit : ' + str(i))
            print('Len train : ' + str(len(train_idx)))
            print('Len test : ' + str(len(test_idx)))
            print('=======')

        splt_idx.append((train_idx, test_idx))

    return splt_idx
