from ML_Models.DocTopicsModel import LDAFlow,LSAFlow
# from DataPrepare.ConnectDB import ConnectDB
from Utility.FeatureEncoder import *
from Utility.TagsDef import  *
import _pickle as pickle,  copy

class TaskDataContainer:
    def __init__(self, typename):
        self.ids=[]
        self.docs=[]
        self.techs=[]
        self.lans=[]
        self.startdates=[]
        self.durations=[]
        self.prizes=[]
        self.diffdegs=[]
        self.taskType = typename

    def encodingFeature(self, choice):
        self.choice=choice
        # print("loading encoding data",self.taskType)
        path = "../data/TaskInstances/GlobalEncoding.data"
        with open("../data/TaskInstances/GlobalEncoding.data","rb") as f:
            encoder=pickle.load(f)
        techs_dict=encoder["techs"]
        lans_dict=encoder["lans"]
        docsEncoder={1:LDAFlow,2:LSAFlow}
        doc_model=docsEncoder[choice]()
        doc_model.name="global"
        try:
            doc_model.loadModel()
        except :
            print("loading doc model failed, now begin to train the model")
            doc_model.name=self.taskType
            doc_model.train_doctopics(self.docs)

        self.docs=doc_model.transformVec(self.docs)
        self.techs_vec = EncodeByDict(self.techs, techs_dict, TaskTechs)
        self.lans_vec = EncodeByDict(self.lans, lans_dict, TaskLans)
        # print(self.taskType,"docs shape",self.docs.shape)
        # print("encoding techs",self.taskType)
        # print(self.taskType,"techs shape",self.techs_vec.shape)
        # print("encoding lans",self.taskType)
        # print(self.taskType,"lans shape",self.lans_vec.shape)

class RegistrationDataContainer:
    def __init__(self,tasktype,taskids,usernames,regdates):
        self.taskids=np.array(taskids)
        self.names=np.array(usernames)
        self.regdates=np.array(regdates)
        self.tasktype=tasktype

        # print("reg data of",tasktype+",size=%d"%len(self.taskids))

    def getRegUsers(self,taskid):
        indices=np.where(self.taskids==taskid)[0]
        if len(indices)==0:
            return None,None
        #print(indices)
        #print(len(self.username))
        regUsers=self.names[indices]
        regDates=self.regdates[indices]
        return regUsers,regDates

    def getAllUsers(self):
        usernames=set(self.names)
        return usernames

    def getUserHistory(self,username):
        indices=np.where(self.names==username)[0]
        if len(indices)==0:
            return (np.array([]),np.array([]))

        ids=self.taskids[indices]
        dates=self.regdates[indices]
        return [ids,dates]

class SubmissionDataContainer:
    def __init__(self,tasktype,taskids,usernames,subnums,subdates,scores,finalranks):
        self.taskids=np.array(taskids)
        self.names=np.array(usernames)
        self.subnums=np.array(subnums)
        self.subdates=np.array(subdates)
        self.scores=np.array(scores)
        self.finalranks=np.array(finalranks)

        # print("submission data of "+tasktype+", size=%d"%len(self.taskids))

    def getSubUsers(self,taskid):
        indices = np.where(self.taskids == taskid)[0]
        if len(indices) == 0:
            return None,None
        # print(indices)
        # print(len(self.username))
        subUsers=self.names[indices]
        subDates=self.subdates[indices]
        return subUsers,subDates

    def getAllUsers(self):
        usernames=set(self.names)
        return usernames

    def getResultOfSubmit(self,username,taskid):
        indices=np.where(self.names==username)[0]
        #print(indices);exit(10)
        if len(indices)==0:
            return None
        indices1=np.where(self.taskids[indices]==taskid)[0]
        if len(indices1)==0:
            return None

        index=indices[indices1[0]]

        return [self.subnums[index],self.finalranks[index],self.scores[index]]

    def getUserHistory(self,username):
        indices=np.where(self.names==username)[0]
        if len(indices)==0:
            return (np.array([]),np.array([]),np.array([]),np.array([]),np.array([]))

        ids=self.taskids[indices]
        subnum=self.subnums[indices]
        date=self.subdates[indices]
        score=self.scores[indices]
        rank=self.finalranks[indices]
        return (ids,subnum,date,score,rank)

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

            # print(tasktype,"=>","Task Instances data size=%d"%(len(self.taskIDs)))
        taskdata = {}
        for i in range(len(self.taskIDs)):
            taskdata[self.taskIDs[i]]=i
        self.dataIndex=taskdata
        self.tasktype=tasktype
        self.taskdata=taskdata

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

class UserHistoryGenerator:
    def __init__(self, testMode=False):
        self.testMode=testMode
    def genActiveUserHistory(self,userdata,regdata,subdata,mode,tasktype,filepath=None):

        userhistory = {}

        for username in userdata.names:
            #fiter incomplete data information
            tenure,skills,skills_vec=userdata.getUserInfo(username)
            if tenure is None:
                continue

            regids, regdates = regdata.getUserHistory(username)
            if mode==0 and len(regids) <minRegNum:
                #default for those have registered
                continue
            subids, subnum, subdates, score, rank = subdata.getUserHistory(username)
            winindices=np.where(rank==0)[0]
            if mode==1 and np.sum(subnum)<minSubNum:
                #for those ever submitted
                continue
            if mode==2 and len(winindices)<minWinNum:
                #for those ever won
                continue

            regids=list(regids)
            regids.reverse()
            regdates=list(regdates)
            regdates.reverse()

            subids=list(subids)
            subids.reverse()
            subnum=list(subnum)
            subnum.reverse()
            subdates=list(subdates)
            subdates.reverse()
            score=list(score)
            score.reverse()
            rank=list(rank)
            rank.reverse()

            userhistory[username] = {"regtasks": [regids, regdates],
                                  "subtasks": [subids, subnum, subdates, score, rank],
                                  "tenure":tenure,"skills":skills.split(","),"skills_vec":skills_vec}

            #print(regdates)
            #print(subdates)

            #print(username, "sub histroy and reg histrory=", len(userData[username]["subtasks"][0]),
            #      len(userData[username]["regtasks"][0]))
        if self.testMode:
            tasktype=tasktype+"-test"
        if filepath is None:
            filepath="data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data"
        print("saving %s history of %d users"%(ModeTag[mode], len(userhistory)),"type="+tasktype)
        with open(filepath, "wb") as f:
            pickle.dump(userhistory, f,True)

    def loadActiveUserHistory(self,tasktype,mode,filepath=None):
        # print("loading %s history of active users "%ModeTag[mode])
        if self.testMode:
            tasktype=tasktype+"-test"

        if filepath is None:
            filepath="../data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data"
        with open(filepath, "rb") as f:
            userData = pickle.load(f)

        return userData


