# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:58:19 2018

@author: RUI WANG
"""
import numpy as np
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = np.float(labelCounts[key])/numEntries
        shannonEnt -= prob*np.log2(prob) 
    return shannonEnt
            
        