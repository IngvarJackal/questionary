#!/usr/bin/env python

from constants import *
import numpy as np

def replace(l, a, b):
    new_l = []
    for x in l:
        changed = False
        for a_i, b_i in zip(a, b):
            if x == a_i:
                new_l.append(b_i)
                changed = True
        if not changed:
            new_l.append(x)
    return new_l

def classToBinary(df, variable, varname):
    unique = set()
    for x in variable:
        if type(x) is float:
            if not np.isnan(x):
                unique.add(x)
        else:
            unique.add(x)
    unique = list(unique)
    columns = [[] for x in range(len(unique))]
    for x in variable:
        for index, column in enumerate(columns):
            if unique[index] == x:
                column.append(1)
            else:
                column.append(0)
        if x not in unique:
            for column in columns:
                column[-1] = NA
    for index, column in enumerate(columns):
        df[varname + "_" + str(unique[index])] = column

def multiclassToBinary(df, variable, varname):
    unique = set()
    for x in variable:
        if type(x) is float:
            if not np.isnan(x):
                unique.add(tuple(x))
        else:
            for y in x:
                unique.add(y)
    unique = list(unique)
    columns = [[] for x in range(len(unique))]
    for x in variable:
        for index, column in enumerate(columns):
            if type(x) is float:
                if np.isnan(x):
                    column.append(NA)
                else:
                    if unique[index] in x:
                        column.append(1)
                    else:
                        column.append(0)
            else:
                if unique[index] in x:
                    column.append(1)
                else:
                    column.append(0)
    for index, column in enumerate(columns):
        df[varname + "_" + "".join([str(x) for x in tuple(unique[index])])] = column

def test_classifier(classifier, data, train_set_percent=0.85, subsets=15, runs=15, balanced=True, verbose=False):
    import numpy as np
    from random import shuffle
    import sys
        
    MEAN = []
    PRESS = 0.0
    SS = 0.0
    TRUE_POSITIVES = []
    FALSE_POSITIVES = []
    TRUE_NEGATIVES = []
    FALSE_NEGATIVES = []

    for subset in range(1, subsets+1):
        if balanced:
            illed = data["tb_intensivity"] > 0
            illed_indices = [i for i, x in enumerate(illed) if x == True]
            healthy_indices = [i for i, x in enumerate(illed) if x == False]
            maxpossible = min(len(illed_indices), len(healthy_indices))
            shuffle(healthy_indices)
            shuffle(illed_indices)
            all_indeces = healthy_indices[:maxpossible] + illed_indices[:maxpossible]
        else:
            all_indeces = data.index.tolist()

        n_train = int(len(all_indeces)*train_set_percent)

        for run in range(1, runs+1):
            indeces = all_indeces[:]
            shuffle(indeces)
            X_train = data.ix[indeces[:n_train], 1:(-1)]
            X_test = data.ix[indeces[n_train:], 1:(-1)]
            Y_train = np.ravel(np.array(data.ix[indeces[:n_train], (-1):]))
            Y_test = np.ravel(np.array(data.ix[indeces[n_train:], (-1):]))

            clf = classifier.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            
            binvector = (Y_pred>0) == (Y_test.transpose()>0)
            testvector = Y_test.transpose() > 0
            TRUE_POSITIVES.append((testvector * binvector).sum()*1.0/len(binvector))
            FALSE_NEGATIVES.append((testvector * -binvector).sum()*1.0/len(binvector))
            FALSE_POSITIVES.append((-testvector * binvector).sum()*1.0/len(binvector))
            TRUE_NEGATIVES.append((-testvector * -binvector).sum()*1.0/len(binvector))  
            matches = round(binvector.sum()*100.0/len(Y_test), 2)
            if verbose:
                print "Subset", subset, "run", run, ":", matches, "% correctly classified"
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
            
            MEAN.append(matches)
            PRESS += ((Y_pred - Y_test)**2).sum()
            SS += ((Y_pred - Y_pred.mean())**2).sum()

    MEAN = np.array(MEAN)
    TRUE_POSITIVES = np.array(TRUE_POSITIVES)
    FALSE_POSITIVES = np.array(FALSE_POSITIVES)
    TRUE_NEGATIVES = np.array(TRUE_NEGATIVES)
    FALSE_NEGATIVES = np.array(FALSE_NEGATIVES)

    print
    print "Mean:", round(MEAN.sum()/runs/subsets, 2), "% correctly classified; std =", round(MEAN.std(), 2)
    print "Q squared:", 1 - PRESS/SS
    print "\tfor %s subsets and %s runs"%(subsets, runs)
    print "True positives: %.3f; False positives: %.3f (std %.3f)\nTrue negatives: %.3f; False negatives: %.3f (std %.3f)"%(float(TRUE_POSITIVES.sum())/(TRUE_POSITIVES.sum()+FALSE_POSITIVES.sum()),
                                                                                                      float(FALSE_POSITIVES.sum())/(TRUE_POSITIVES.sum()+FALSE_POSITIVES.sum()),
                                                                                                      max(TRUE_POSITIVES.std(),FALSE_POSITIVES.std()),
                                                                                                      float(TRUE_NEGATIVES.sum())/(TRUE_NEGATIVES.sum()+FALSE_NEGATIVES.sum()),
                                                                                                      float(FALSE_NEGATIVES.sum())/(TRUE_NEGATIVES.sum()+FALSE_NEGATIVES.sum()),
                                                                                                      max(TRUE_NEGATIVES.std(),FALSE_NEGATIVES.std()),)


