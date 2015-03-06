#!/usr/bin/env python

import pandas
from constants import *
import numpy as np
np.set_printoptions(threshold=50000000)

def print_full(x):
    pandas.set_option("display.max_rows", x.shape[0])
    pandas.set_option("display.max_columns", x.shape[1])
    print(x)
    pandas.reset_option("display.max_rows")
    pandas.reset_option("display.max_columns", x.shape[1])

A_MISSED = False
A_PEARSON = True; A_PEARSON_SIGNIFICANT_ONLY = False
A_PLOT = False
A_STDS = False
A_MAD = False
A_FREQ = False
A_AUTOCORR = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
STAT_PDF = "descriptive.pdf"
pp = PdfPages(STAT_PDF)

SIGNIFICANCE_MULTIPLIER = 2

tidy_data = pandas.read_csv(PATH_TO_TIDY_DATA)
filled_data = pandas.read_csv(PATH_TO_FILLED_DATA)
red_dis_data = pandas.read_csv(PATH_TO_RED_DIS_DATA)

data = filled_data

if A_MISSED:
    print "Missed values in tidy"
    variables_missed = data.count(axis=0); variables_missed.sort(axis=1)
    observations_missed = data.count(axis=1); observations_missed.sort(axis=1)

    print variables_missed[:25], "\n\tOF", variables_missed.max()
    print observations_missed[:25], "\n\tOF", observations_missed.max()

if A_STDS:
    print "Stds values in tidy"
    stds = data.std(axis=0); stds.sort(axis=1)

    print stds

if A_PLOT:
    for name in data.columns[1:-2]:
        plt.scatter(data[name], data.ix[:, -2], marker=".", linewidths=0, s=1000, alpha=0.1)
        plt.suptitle(name + " vs " + data.columns[-2])
        plt.savefig(pp, format='pdf')
        plt.clf()
        #plt.show()
    plt.scatter(data["tb_resistance"], data["tb_intensivity"], marker=".", linewidths=0, s=1000, alpha=0.1)
    plt.suptitle("tb_resistance vs tb_intensivity")
    plt.savefig(pp, format='pdf')
    plt.clf()
        
if A_PEARSON:
    from scipy.stats.stats import pearsonr
    print "Pearson correlation test\nSignificance level", SIG_LEVEL*SIGNIFICANCE_MULTIPLIER
    for name in data.columns[1:-2]:
        pear = pearsonr(data[name], data.ix[:, -2])
        if A_PEARSON_SIGNIFICANT_ONLY:
            if SIG_LEVEL*SIGNIFICANCE_MULTIPLIER > pear[1]:
                print name, round(pear[0]*100, 4), "% ,", round(pear[1], 6)
        else:
            print name, round(pear[0]*100, 4), "% ,", round(pear[1], 6), ";", "is significant:", SIG_LEVEL*SIGNIFICANCE_MULTIPLIER > pear[1]

if A_MAD:
    print "MAD Mahalanobis"
    for name in data.columns[1:-2]:
        median = data[name].median()
        distance = (data[name] - median)/(median + 1)
        distance.sort(axis=1)
        print distance
        raw_input()
        
if A_FREQ:
    import matplotlib.pyplot as plt
    print "Computing of freqs"
    print_full(data.describe())
    for name in set(["age", "visdoc", "relatives", "satisf", "sleep"]).intersection(data.columns):
        data.boxplot(name)
        plt.savefig(pp, format='pdf')
        plt.clf()
        #plt.show()
    for name in set(["sex", "stress", "xray", "treat_bef", "city_type", "moving", "train", "ctran", "car", "occup", "expense", "finstate", "sport", "smoking", "alcohol", "prison", "diabet", "hiv", "chron_dis", "reason_send","reason_unwell","tb_intensivity", "tb_resistance"]).intersection(data.columns):
        print name
        print data[name].value_counts()

if A_AUTOCORR:
    print "Correlation between variables"
    coeffs = np.corrcoef(data.ix[:, 1:-2])
    print coeffs
    print "Below 85%", round((abs(coeffs)<0.75).sum()*1.0/coeffs.shape[0]/coeffs.shape[1]*100), "%"

pp.close()
