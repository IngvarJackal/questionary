#!/usr/bin/env python

from constants import *
from sklearn import tree
import pandas
import numpy as np
from random import shuffle
import operator
import sys

TRAIN_SET_PERCENT = 0.85
SUBSETS = 15
RUNS = 15
MAX_DEPTH = 12 # default is None
PLOT = False
SHOW_IMPORTANCE = False
VERBOSE = False
BALANCED = True

filled_data = pandas.read_csv(PATH_TO_FILLED_DATA)
red_dis_data = pandas.read_csv(PATH_TO_RED_DIS_DATA)

data = red_dis_data

del data[data.columns[-1]]
    
MEAN = []
PRESS = 0.0
SS = 0.0
IMPORTANCES = 0.0
TRUE_POSITIVES = []
FALSE_POSITIVES = []
TRUE_NEGATIVES = []
FALSE_NEGATIVES = []

for subset in range(1, SUBSETS+1):
    if BALANCED:
        illed = data["tb_intensivity"] > 0
        illed_indices = [i for i, x in enumerate(illed) if x == True]
        healthy_indices = [i for i, x in enumerate(illed) if x == False]
        maxpossible = min(len(illed_indices), len(healthy_indices))
        shuffle(healthy_indices)
        shuffle(illed_indices)
        all_indeces = healthy_indices[:maxpossible] + illed_indices[:maxpossible]
    else:
        all_indeces = data.index.tolist()

    n_train = int(len(all_indeces)*TRAIN_SET_PERCENT)

    for run in range(1, RUNS+1):
        indeces = all_indeces[:]
        shuffle(indeces)
        X_train = data.ix[indeces[:n_train], 1:(-1)]
        X_test = data.ix[indeces[n_train:], 1:(-1)]
        Y_train = data.ix[indeces[:n_train], (-1):]
        Y_test = np.array(data.ix[indeces[n_train:], (-1):])
           
        clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
        clf = clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        binvector = (Y_pred>0) == (Y_test.transpose()>0)
        testvector = Y_test.transpose() > 0
        TRUE_POSITIVES.append((testvector * binvector).sum()*1.0/len(binvector))
        FALSE_NEGATIVES.append((testvector * -binvector).sum()*1.0/len(binvector))
        FALSE_POSITIVES.append((-testvector * binvector).sum()*1.0/len(binvector))
        TRUE_NEGATIVES.append((-testvector * -binvector).sum()*1.0/len(binvector))
        matches = round(binvector.sum()*100.0/len(Y_test), 2)
        if VERBOSE:
            print "Run", run, ":", matches, "% correctly classified"
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
        
        MEAN.append(matches)
        PRESS += ((Y_pred - Y_test)**2).sum()
        SS += ((Y_pred - Y_pred.mean())**2).sum()
        IMPORTANCES += clf.feature_importances_
        
        if PLOT:
            from sklearn.externals.six import StringIO  
            import pydot
            dot_data = StringIO() 
            tree.export_graphviz(clf, out_file=dot_data) 
            graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
            graph.write_png("model-%s.png" % run)

MEAN = np.array(MEAN)
TRUE_POSITIVES = np.array(TRUE_POSITIVES)
FALSE_POSITIVES = np.array(FALSE_POSITIVES)
TRUE_NEGATIVES = np.array(TRUE_NEGATIVES)
FALSE_NEGATIVES = np.array(FALSE_NEGATIVES)

print
print "Mean:", round(MEAN.sum()/RUNS/SUBSETS, 2), "% correctly classified; std =", round(MEAN.std(), 2)
print "Q squared:", 1 - PRESS/SS
print "\tfor %s subsets and %s runs"%(SUBSETS, RUNS)
print "True positives: %.3f; False positives: %.3f (std %.3f)\nTrue negatives: %.3f; False negatives: %.3f (std %.3f)"%(float(TRUE_POSITIVES.sum())/(TRUE_POSITIVES.sum()+FALSE_POSITIVES.sum()),
                                                                                                  float(FALSE_POSITIVES.sum())/(TRUE_POSITIVES.sum()+FALSE_POSITIVES.sum()),
                                                                                                  max(TRUE_POSITIVES.std(),FALSE_POSITIVES.std()),
                                                                                                  float(TRUE_NEGATIVES.sum())/(TRUE_NEGATIVES.sum()+FALSE_NEGATIVES.sum()),
                                                                                                  float(FALSE_NEGATIVES.sum())/(TRUE_NEGATIVES.sum()+FALSE_NEGATIVES.sum()),
                                                                                                  max(TRUE_NEGATIVES.std(),FALSE_NEGATIVES.std()),)

if SHOW_IMPORTANCE:
    importances = {}
    for name, imp in zip(data.ix[indeces[:n_train], 1:-1].columns, IMPORTANCES/RUNS):
        importances[name] = imp
    sorted_importances = sorted(importances.items(), key=operator.itemgetter(1))
    for name, imp in sorted_importances:
        print name, imp
