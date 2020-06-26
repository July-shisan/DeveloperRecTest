#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers
import json
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.DNNModel import DNNCLassifier
from ML_Models.EnsembleModel import EnsembleClassifier
from DataPrepare.TopcoderDataSet import TopcoderWin
from DataPrepare.TaskContent import *
from DataPrepare.TaskUserInstances import *
class PolicyModel:
    def initData(self):
        self.mymetric = TopKMetrics(tasktype=self.datatype,testMode=True)
        self.subExpr = self.mymetric.subRank
        self.userIndex = getUsers(self.datatype+"-test", mode=2)

    def __init__(self, tasktype=None, datatype=None):
        self.availableModels = {
            0: EnsembleClassifier,
            1: XGBoostClassifier,
            2: DNNCLassifier
        }
        self.metaReg = 1
        self.metaSub = 1
        self.metaWin = 1
        #aux info
        self.tasktype = tasktype
        self.datatype = self.tasktype
        if datatype is not None:
            self.datatype = datatype
        self.initData()
        self.name = tasktype+"rulePredictor"

    def predictUsers(self, topReg, topSub, topWin, top_R, top_S):
        userNum = len(self.userIndex)
        taskNum = len(topReg)
        topRN = int(top_R * len(self.userIndex))
        topSN = int(top_S * len(self.userIndex))
        for i in range(taskNum):
            predictY = copy.deepcopy(topWin[i])
            topRY = topReg[i, :topRN]
            topSY = topSub[i, :topSN]
            for pos in range(userNum):
                if pos not in topRY:
                    predictY[pos] = 0
                    continue
                if pos not in topSY:
                    predictY[pos] = 0
            predictY, _ = self.mymetric.getTopKonPossibility(predictY, 10)
            name = []
            for i in predictY:
                name.append(self.userIndex[i])
            return name


    def TuneTempResults(self, X):
        regModels = []
        subModels = []
        winModels = []

        tasktype = self.tasktype
        for i in self.availableModels.keys():
            regModel = self.availableModels[i]()
            subModel = self.availableModels[i]()
            winModel = self.availableModels[i]()
            winModel.name=tasktype+"-classifierWin"
            if "#" in tasktype:
                pos=tasktype.find("#")
                regModel.name=tasktype[:pos]+"-classifierReg"
                subModel.name=tasktype[:pos]+"-classifierSub"
            else:
                regModel.name=tasktype+"-classifierReg"
                subModel.name=tasktype+"-classifierSub"

            regModel.loadModel()
            subModel.loadModel()
            winModel.loadModel()
            regModels.append(regModel)
            subModels.append(subModel)
            winModels.append(winModel)
        regYs=[]
        subYs=[]
        winYs=[]

        taskNum=len(X)//len(self.userIndex) # 219
        userNum=len(self.userIndex) # 212
        for i in self.availableModels.keys():
            regModel, subModel, winModel = regModels[i], subModels[i], winModels[i]
            regY = regModel.predict(X)
            subY = subModel.predict(X)
            winY = winModel.predict(X)

            regY = np.reshape(regY, newshape=(taskNum, userNum))
            subY = np.reshape(subY, newshape=(taskNum, userNum))
            winY = np.reshape(winY, newshape=(taskNum, userNum))

            regYs.append(regY)
            subYs.append(subY)
            winYs.append(winY)

        for i in range(len(regYs)):
            regY, subY = regYs[i], subYs[i]
            for j in range(taskNum):
                topReg, _ = self.mymetric.getTopKonPossibility(regY[j], 10000)
                topSub, _ = self.mymetric.getTopKonPossibility(subY[j], 10000)
                regY[j] = topReg
                subY[j] = topSub
            regYs[i], subYs[i] = regY, subY
        return regYs, subYs, winYs

def getBestPerformance(metadata, tasktype):
    bestacc10=0
    feature = []
    a1 = 0
    a2 = 0
    a3 = 0
    top_r = 0
    top_s = 0
    for md in metadata:
        if md[-2] > bestacc10:
            bestacc10=md[-2]
            print(bestacc10)
            a1 = md[0]
            a2 = md[1]
            a3 = md[2]
            top_r = md[3]
            top_s = md[4]
    feature.append(a1)
    feature.append(a2)
    feature.append(a3)
    feature.append(top_r)
    feature.append(top_s)
    with open("../data/FeatureData/" + tasktype + ".pkl", "wb") as f:
         pickle.dump(feature, f)
    return a1, a2, a3, top_r, top_s

def generateSearchData(saveData=True, tasktype='', regYs=[], subYs=[], winYs=[]):
    predictUsers = []
    if saveData:
        with open("../data/MetaData/" + tasktype + ".pkl", "rb") as f:
            metafeatures = pickle.load(f)
        feature = np.array(metafeatures, dtype=np.float32)
        getBestPerformance(feature, tasktype)
    else:
        with open("../data/FeatureData/" + tasktype + ".pkl", "rb") as f:
            metafeatures = pickle.load(f)
        feature = np.array(metafeatures, dtype=np.float32)
        a1, a2, a3, top_r, top_s = feature
        regY = regYs[int(a1)]
        subY = subYs[int(a2)]
        winY = winYs[int(a3)]
        name = model.predictUsers(regY, subY, winY, top_r, top_s)
        return name
    return predictUsers

def recommend(challenge):
    data = []
    # tashid, title,detail, duration, tec, lan, prize, startdate, diffdeg, tasktype
    data.append(challenge.id)
    data.append(challenge.title)
    data.append(challenge.requirment)
    data.append(10)
    data.append(challenge.technology)
    data.append(challenge.technology)
    data.append(challenge.award)
    data.append(2996)
    data.append(0.01600)
    data.append(challenge.chtype)

    # modelType = data[9]
    dataType = data[9]
    dataSet = initDataSet(data, tasktype=dataType)
    dataSet.encodingFeature(1)
    saveTaskData(dataSet)

    genDataSet(dataType, mode=2, testInst=True)
    data = TopcoderWin(dataType, testratio=1, validateratio=0)
    data.setParameter(dataType, 2, True)
    data.loadData()
    data.WinClassificationData()
    # search
    model = PolicyModel(dataType)
    regYs, subYs, winYs = model.TuneTempResults(data.testX)  # 测试数据
    username = generateSearchData(False, dataType, regYs, subYs, winYs)
    print(username)
    return username
