#!/usr/bin/env python

from constants import *
from utils import *
import numpy as np
import pandas

######################################################################
# id, tb_intensivity and tb_resistance are mandatory
######################################################################

MUCH_PERCENTILE = 80

raw_data = pandas.read_csv(PATH_TO_RAW_DATA, sep=";")
tidy_data = pandas.DataFrame()
red_dis_data = pandas.DataFrame()

tidy_data["id"] = raw_data["ID"]
tidy_data["age"] = raw_data["age"]
tidy_data["sex"] = raw_data["sex"].map({"M":0, "F":1})
def procDish(x):
    try:
        return list(set([a for a in x.lower()]))
    except Exception:
        return NA
multiclassToBinary(tidy_data, raw_data["Dishes"].map(procDish), "dish")
tidy_data["stress"] = raw_data["Stress"].map({"Yes":1, "No":0})
def procSympt(x):
    try:
        return list(set(x.lower().split(",")))
    except Exception:
        return NA
multiclassToBinary(tidy_data, raw_data["Sympt"].map(procSympt), "sympt")
def procVisdoc(col):
    col_clear = [float(x) for x in replace(col, ["NA", "much"], [0, 0])]
    muchval = np.percentile(col_clear, MUCH_PERCENTILE)
    return replace(col, ["NA", "much"], [NA, muchval])
tidy_data["visdoc"] = procVisdoc(raw_data["Visdoc"])
tidy_data["xray"] = raw_data["Xrays"].map({"Yes":1, "No":0})
tidy_data["relatives"] = (((raw_data["Famdis"].map({"Yes":1.5, "No":0}) + raw_data["Famtreat"].map({"Yes":1.5, "No":0}))/2).round())
tidy_data["treat_bef"] = raw_data["TreatTB"].map({"Yes":1, "No":0})
tidy_data["city_type"] = raw_data["Home"].map({"City":1, "Village":0})
tidy_data["moving"] = raw_data["Leaving"]
tidy_data["train"] = raw_data["Train"].map({"No":0, "twice":2, "often":10})
tidy_data["ctran"] = raw_data["CityTran"].map({"No":0, "twice":2, "often":10})
tidy_data["car"] = raw_data["Car"].map({"No":0, "rare":2, "often":10})
tidy_data["satisf"] = raw_data["Satisf"]
classToBinary(tidy_data, raw_data["LifeCond"], "condit")
tidy_data["sleep"] = raw_data["SlWith"]
def procOccup(col):
    result = []
    for x in col:
        if x == "unempl":
            result.append(0)
        elif x == "NA":
            result.append(NA)
        else:
            result.append(2)
    return result
tidy_data["occup"] = procOccup(raw_data["Occup"])
tidy_data["expense"] = raw_data["Expense"].map({"little":0, "average":3, "much":6})
tidy_data["finstate"] = raw_data["FinanSit"].map({"worse":-1, "same":0, "better":1})
tidy_data["sport"] = raw_data["Sport"].map({"Yes":1, "No":0})
tidy_data["smoking"] = raw_data["Smoking"].map({"Yes":2, "No":0, "P":1})
classToBinary(tidy_data, replace(raw_data["TBis"], ["C,E"], ["C"]), "tb_attit")
tidy_data["alcohol"] = raw_data["Alcohol"]
tidy_data["prison"] = raw_data["Prison"].map({"Yes":1, "No":0})
tidy_data["diabet"] = raw_data["Diabet"].map({"Yes":1, "No":0})
tidy_data["hiv"] = raw_data["HIV"].map({"Yes":1, "No":0})
tidy_data["chron_dis"] = raw_data["ChronDis"].map({"Yes":1, "No":0})
multiclassToBinary(tidy_data, replace(raw_data["ReasonHosp"], ["unwell", "Send", "both", "NA"], [["unwell"], ["send"], ["send", "unwell"], NA]), "reason")
tidy_data["tb_intensivity"] = (raw_data["MTBn"]*(-3) +
                                   (raw_data["MTBh"] + raw_data["MTBhRR"])*4 +
                                   (raw_data["MTBm"] + raw_data["MTBmRR"])*3 +
                                   (raw_data["MTBl"] + raw_data["MTBlRR"])*2 +
                                   (raw_data["MTBvl"] + raw_data["MTBvlRR"])*1) 
tidy_data["tb_resistance"] =  raw_data["MTBhRR"] + raw_data["MTBmRR"] + raw_data["MTBlRR"] + raw_data["MTBvlRR"]

tidy_data = tidy_data.drop(tidy_data.index[[5, 21, 61, 13]]) # many missed values
tidy_data.to_csv(PATH_TO_TIDY_DATA, encoding="utf-8", na_rep="NA", index=False)

filled_data = tidy_data.interpolate(method="barycentric").interpolate(method="values") # no NAs
filled_data.fillna(filled_data.median(), inplace=True)

filled_data.to_csv(PATH_TO_FILLED_DATA, encoding="utf-8", na_rep="NA", index=False)

red_dis_data["id"] = filled_data["id"]
red_dis_data["age"] = filled_data["age"]
red_dis_data["dish_g"] = filled_data["dish_g"]
red_dis_data["stress"] = filled_data["stress"]
red_dis_data["sympt_a"] = filled_data["sympt_a"]
red_dis_data["xray"] = filled_data["xray"]
red_dis_data["expense"] = filled_data["expense"]
red_dis_data["tb_attit_C"] = filled_data["tb_attit_C"]
red_dis_data["diabet"] = filled_data["diabet"]
red_dis_data["hiv"] = filled_data["hiv"]
red_dis_data["chron_dis"] = filled_data["chron_dis"]
red_dis_data["reason_send"] = filled_data["reason_send"]
red_dis_data["reason_unwell"] = filled_data["reason_unwell"]
red_dis_data["tb_intensivity"] = filled_data["tb_intensivity"]
red_dis_data["tb_resistance"] = filled_data["tb_resistance"]

red_dis_data.to_csv(PATH_TO_RED_DIS_DATA, encoding="utf-8", na_rep="NA", index=False)
