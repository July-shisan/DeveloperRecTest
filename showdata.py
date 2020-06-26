import _pickle as pickle,copy

class Tasks:
    def __init__(self,tasktype,begindate=3000):
        filepath="../data/TaskInstances/taskDataSet/"+tasktype+"-taskData.data"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            for k in data.keys():
                data[k]=list(data[k])
                data[k].reverse()

            self.taskIDs = data["ids"]
            self.docX=data["docX"]
            self.lans=data["lans"]
            self.techs=data["techs"]
            self.diffdegs=data["diffdegs"]
            self.postingdate=data["startdates"]
            self.durations=data["durations"]
            self.prizes=data["prizes"]

            print(tasktype,"=>","Task Instances data size=%d"%(len(self.taskIDs)))
            #print("post",self.postingdate[:20])
        taskdata = {}
        for i in range(len(self.taskIDs)):
            taskdata[self.taskIDs[i]]=i

        self.dataIndex=taskdata

        self.tasktype=tasktype
        self.taskdata=taskdata

        self.loadData(begindate)

    def loadData(self,begindate=4200):
        pos=0
        for pos in range(len(self.taskIDs)):
            if self.postingdate[pos]>=begindate:
                break

        self.taskIDs = self.taskIDs[:pos]
        self.docX=self.docX[:pos]
        self.lans=self.lans[:pos]
        self.techs=self.techs[:pos]
        self.diffdegs=self.diffdegs[:pos]
        self.postingdate=self.postingdate[:pos]
        self.durations=self.durations[:pos]
        self.prizes=self.prizes[:pos]

        print(self.tasktype+": task size=%d"%len(self.taskIDs))

    def ClipRatio(self,ratio=0.4):
        clip_pos=int(ratio*len(self.taskIDs))
        self.taskIDs=self.taskIDs[:clip_pos]
        self.docX=self.docX[:clip_pos]
        self.lans=self.lans[:clip_pos]
        self.techs=self.techs[:clip_pos]
        self.diffdegs=self.diffdegs[:clip_pos]
        self.postingdate=self.postingdate[:clip_pos]
        self.durations=self.durations[:clip_pos]
        self.prizes=self.prizes[:clip_pos]

if __name__ == '__main__':
    filepath = "Architecture-taskData.data"
    userpath = 'Architecture-UsersReg.data'

    with open(filepath, "rb") as f:
        data = pickle.load(f)
        for k in data.keys():
            print(k)
            data[k] = list(data[k])
            data[k].reverse()

        # print(data)
        taskIDs = data["ids"]
        print(len(taskIDs))
        docX = data["docX"]
        print(docX)
        lans = data["lans"]
        techs = data["techs"]
        diffdegs = data["diffdegs"]
        postingdate = data["startdates"]
        durations = data["durations"]
        prizes = data["prizes"]

    # with open(userpath, "rb") as f:
    #     user = pickle.load(f)
    #     print(len(user))
    #     print(user)