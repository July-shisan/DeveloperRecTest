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

    def testResults(self,trueLabel,topReg,topSub,topWin,top_R,top_S):
        '''
        :param topReg: 2 dim (taskNum,userNum)
        :param topSub: 2 dim (taskNum,userNum)
        :param topWin: 2 dim (taskNum,userNum)
        :param top_R:  0~1
        :param top_S: 0~1
        :return: predicts
        '''
        taskNum=len(topReg)
        userNum=len(self.userIndex)
        trueLabel=np.reshape(trueLabel,newshape=(taskNum,userNum))
        topRN=int(top_R*len(self.userIndex))
        topSN=int(top_S*len(self.userIndex))

        mrr=np.zeros(shape=taskNum,dtype=np.float32)
        acc3=np.zeros(shape=taskNum,dtype=np.int)
        acc5=np.zeros(shape=taskNum,dtype=np.int)
        acc10=np.zeros(shape=taskNum,dtype=np.int)

        for i in range(taskNum):
            predictY=copy.deepcopy(topWin[i])
            topRY=topReg[i,:topRN]
            topSY=topSub[i,:topSN]
            for pos in range(userNum):
                if pos not in topRY:
                    predictY[pos]=0
                    continue
                if pos not in topSY:
                    predictY[pos]=0

            predictY,_=self.mymetric.getTopKonPossibility(predictY, 10)

            trueY = trueLabel[i]
            trueY = np.where(trueY == 1)[0]
            if len(trueY) == 0:
                continue
            trueY = set(trueY)

            com=trueY.intersection(predictY[:3])
            if len(com)==0:
                acc3[i]=0
            else:
                acc3[i]=1
            com=trueY.intersection(predictY[:5])
            if len(com)==0:
                acc5[i]=0
            else:
                acc5[i]=1
                com=trueY.intersection(predictY[:10])
            if len(com)==0:
                acc10[i]=0
            else:
                acc10[i]=1
            #mrr
            for j in range(len(predictY)):
                if predictY[j] in trueY:
                    mrr[i]=1.0/(1.0+j)
                break
        MRR=np.mean(mrr)
        ACC3=np.mean(acc3)
        ACC5=np.mean(acc5)
        ACC10=np.mean(acc10)
        #print(self.tasktype,"top 3 5 and 10 acc=",ACC3,ACC5,ACC10,"mrr=",MRR)
        return ACC3,ACC5,ACC10,MRR

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
        # print("all meta models are loaded")
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

def generateSearchData(saveData=True, tasktype=''):
    predictUsers = []
    if saveData:
        bestfeature = {
            "a1": 0,
            "a2": 0,
            "a3": 0,
            "regt": 0.1,
            "subt": 0.1,
        }
        curfeature = copy.deepcopy(bestfeature)
        featureAll = []
        for a1 in featureSpace["a1"]:
            for a2 in featureSpace["a2"]:
                for a3 in featureSpace["a3"]:
                    for top_r in featureSpace["regt"]:
                        for top_s in featureSpace["subt"]:
                            featureAll.append((a1, a2, a3, top_r, top_s))
        # 27
        for f in featureAll:
            a1, a2, a3, top_r, top_s = f
            regY = regYs[a1]
            subY = subYs[a2]
            winY = winYs[a3]
            name = model.predictUsers(regY, subY, winY, top_r, top_s)
            predictUsers.append(name)
        # print("save Meta Data for", tasktype)
        # with open("../data/MetaData/" + tasktype + ".pkl", "wb") as f:
        #     pickle.dump(featureData, f)
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

    # modelType = data[9]
    dataType = data[9]
    dataSet = initDataSet(data, tasktype=dataType)
    dataSet.encodingFeature(1)
    saveTaskData(dataSet)

    genDataSet(dataType, mode=2, testInst=True)
    # mode = 2
    featureSpace = {
                "a1":[0,1,2],
                "a2":[0,1,2],
                "a3":[0,1,2],
                "regt":[i/10.0 for i in (5, )],
                "subt":[i/10.0 for i in (3, )],
            }
    data = TopcoderWin(dataType, testratio=1, validateratio=0)
    data.setParameter(dataType, 2, True)
    data.loadData()
    data.WinClassificationData()
    # search
    model = PolicyModel(dataType)
    regYs, subYs, winYs = model.TuneTempResults(data.testX)  # 测试数据
    username = generateSearchData(False, dataType)
    print(username)

    # names = []
    # dict = {}
    # for l in username:
    #     names += l
    # for key in names:
    #     dict[key] = dict.get(key, 0) + 1
    #
    # d = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    # print(d)
    # for i in range(0, 10):
    #     print(d[i][0])