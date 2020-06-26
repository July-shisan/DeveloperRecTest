from Utility.FeatureEncoder import onehotFeatures
import numpy as np
import matplotlib.pyplot as plt
from DataPrepare.DataContainer import TaskDataContainer
import multiprocessing,_pickle as pickle
from ML_Models.DocTopicsModel import LDAFlow

def initDataSet(data, tasktype):
        ids=[]
        docs=[]
        techs=[]
        lans=[]
        startdates=[]
        durations=[]
        prizes=[]
        diffdegs=[]
        tasktypes=[]
        ids.append(data[0])
        docs.append(data[2] + " " + data[1])
        if data[3] > 50:
            durations.append(50)
        elif data[3] < 1:
            durations.append(1)
        else:
            durations.append(data[3])
        techs.append(data[4])
        lans.append(data[5])
        if data[6] != '':
            prize = data[6]
            if prize > 6000:
                prize = 6000
            if prize < 1:
                prize = 1
            prizes.append(prize)
        else:
            prizes.append(1.)
        if data[7] < 1:
            startdates.append(1)
        else:
            startdates.append(data[7])
        if data[8] > 0.6:
            diffdegs.append(0.6)
        else:
            diffdegs.append(data[8])
        tasktypes.append(data[9].replace("/", "_"))

        container = TaskDataContainer(typename=tasktype)
        container.ids = ids
        container.docs = docs
        container.techs = techs
        container.lans = lans
        container.startdates = startdates
        container.durations = durations
        container.prizes = prizes
        container.diffdegs = diffdegs
        return container
#save data content as a vector
def saveTaskData(taskdata):
    data={}
    docX=np.array(taskdata.docs)
    data["docX"]=docX
    lans=[None for i in range(len(taskdata.ids))]
    techs=[None for i in range(len(taskdata.ids))]
    for i in range(len(taskdata.ids)):
        lans[i]=taskdata.lans[i].split(",")
        techs[i]=taskdata.techs[i].split(",")

    data["lans"]=lans
    # print(lans)
    data["techs"]=techs
    data["diffdegs"]=taskdata.diffdegs
    data["startdates"]=taskdata.startdates
    data["durations"]=taskdata.durations
    data["prizes"]=taskdata.prizes
    data["ids"]=taskdata.ids
    with open("../data/TaskInstances/taskDataSet/"+taskdata.taskType+"-taskData.data","wb") as f:
        pickle.dump(data, f, True)

def getTaskData(taskdata):
    data={}
    docX=np.array(taskdata.docs)
    data["docX"]=docX
    lans=[None for i in range(len(taskdata.ids))]
    techs=[None for i in range(len(taskdata.ids))]
    for i in range(len(taskdata.ids)):
        lans[i]=taskdata.lans[i].split(",")
        techs[i]=taskdata.techs[i].split(",")

    data["lans"]=lans
    # print(lans)
    data["techs"]=techs
    data["diffdegs"]=taskdata.diffdegs
    data["startdates"]=taskdata.startdates
    data["durations"]=taskdata.durations
    data["prizes"]=taskdata.prizes
    data["ids"]=taskdata.ids
    return data

if __name__ == '__main__':
    data = (30006961, 'Ferguson.com Omniture Implementation Assembly',
            'All new pages on ferguson.com should be appropriately tagged for Omniture, to meet the requirements as described below. ',
            5, 'JSON,Java', 'Java', 7003, 4003, 0.02820, 'Assembly Competition')
    tasktype = 'Assembly Competition'
    dataSet=initDataSet(data, tasktype=tasktype)
    dataSet.encodingFeature(1)
    saveTaskData(dataSet)
