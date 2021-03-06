#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers
import json
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.DNNModel import DNNCLassifier
from ML_Models.EnsembleModel import EnsembleClassifier
import warnings
warnings.filterwarnings("ignore")
# Policy Model main code
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

    def TuneTempResults(self, X, savedata=False):
        if savedata:
            with open("../data/MetaData/" + self.tasktype + ".pkl", "rb") as f:
                metafeatures = pickle.load(f)
            feature = np.array(metafeatures, dtype=np.float32)
            a1, a2, a3, top_r, top_s = getBestPerformance(feature, self.tasktype)
        else:
            with open("../data/FeatureData/" + self.tasktype + ".pkl", "rb") as f:
                metafeatures = pickle.load(f)
            feature = np.array(metafeatures, dtype=np.float32)
            a1, a2, a3, top_r, top_s = feature

        print(a1)
        print(a2)
        print(a3)
        print(top_r)
        print(top_s)
        regModels = []
        subModels = []
        winModels = []

        tasktype = self.tasktype

        regModel = self.availableModels[int(a1)]()
        subModel = self.availableModels[int(a2)]()
        winModel = self.availableModels[int(a3)]()
        winModel.name = tasktype + "-classifierWin"
        if "#" in tasktype:
            pos = tasktype.find("#")
            regModel.name = tasktype[:pos] + "-classifierReg"
            subModel.name = tasktype[:pos] + "-classifierSub"
        else:
            regModel.name = tasktype + "-classifierReg"
            subModel.name = tasktype + "-classifierSub"

        regModel.loadModel()
        subModel.loadModel()
        winModel.loadModel()
        regModels.append(regModel)
        subModels.append(subModel)
        winModels.append(winModel)
        # print("all meta models are loaded")
        regYs=[]
        subYs=[]
        winYs=[]

        taskNum = len(X)//len(self.userIndex) # 219
        userNum = len(self.userIndex) # 212

        regModel, subModel, winModel = regModels[int(a1)], subModels[int(a2)], winModels[int(a3)]
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
        return regYs, subYs, winYs, top_r, top_s

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
            bestacc10 = md[-2]
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

def generateSearchData(saveData=True, tasktype=''):
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

if __name__ == '__main__':
    from DataPrepare.TopcoderDataSet import TopcoderWin
    from DataPrepare.TaskContent import *
    from DataPrepare.TaskUserInstances import *
    data = []
    # tashid, title,detail, duration, tec, lan, prize, startdate, diffdeg, tasktype
    data.append(300267)
    data.append('BCMS Web Bug Hunt 3 - Firefox on PC')
    detail = 'The client is looking for a solution for key teams within our company to keep critical services working in the event of a major disaster (ranging from major technical infrastructure failures to fires and other local environmental disasters). The Business Continuity Mobility Solution should allow them to manage at least three key functions:This bug hunt will attempt to identify as many issues as possible in a short amount of time, across all aspects of the application.The scope of this contest is only Firefox on PC. See below for how to log the bugs correctly so they count for the bug hunt.'
    data.append(detail)
    data.append(2)
    data.append('HTML,JavaScript')
    data.append('HTML,JavaScript')
    data.append(5002)
    data.append(2996)
    data.append(0.01600)
    data.append('Bug Hunt')

    dataType = data[9]
    dataSet = initDataSet(data, tasktype=dataType)
    dataSet.encodingFeature(1)
    data = getTaskData(dataSet)

    genDataSet(dataType, mode=2, testInst=True)
    data = TopcoderWin(dataType, testratio=1, validateratio=0)
    data.setParameter(dataType, 2, True)
    data.loadData()
    data.WinClassificationData()

    model = PolicyModel(dataType)
    regYs, subYs, winYs, top_r, top_s = model.TuneTempResults(data.testX, False)  # 测试数据
    username = model.predictUsers(regYs[0], subYs[0], winYs[0], top_r, top_s)
    print(username)