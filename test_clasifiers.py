from constants import *
from utils import *
import pandas
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVR

red_dis_data = pandas.read_csv(PATH_TO_RED_DIS_DATA)
data = red_dis_data

del data[data.columns[-1]]

N_RUNS = 20
N_SUBSETS = 80

print "DecisionTreeClassifier"
test_classifier(DecisionTreeClassifier(), data, subsets=N_SUBSETS, runs=N_RUNS)
print "-"*79

print "RandomForestClassifier"
test_classifier(RandomForestClassifier(n_estimators=100), data, subsets=N_SUBSETS, runs=N_RUNS)
print "-"*79

print "ExtraTreesClassifier"
test_classifier(ExtraTreesClassifier(n_estimators=100), data, subsets=N_SUBSETS, runs=N_RUNS)
print "-"*79

print "LDA"
test_classifier(LDA(), data, subsets=N_SUBSETS, runs=N_RUNS)
print "-"*79

print "SVM"
test_classifier(SVR(), data, subsets=N_SUBSETS, runs=N_RUNS)
print "-"*79

print "DecisionTreeClassifier + AdaBoostClassifier"
test_classifier(AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=100), data, subsets=N_SUBSETS, runs=N_RUNS)
print "-"*79
