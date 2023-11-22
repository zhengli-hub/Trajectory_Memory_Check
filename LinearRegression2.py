
import scipy.optimize
import sklearn.metrics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ACCDataProcess import ACCData
from HistoricDataProcess import HistoricData
from VanderbiltDataProcess import VanderbiltData
from OpenAccDataProcess import OpenAccData
from sklearn.model_selection import KFold
class linearCFModelRegress:

    # CF_Data: all tra info
    # oneTra: one tra info

    def __init__(self, CF_Data):
        self.CF_Data = CF_Data
        self.memory = None  # 后延的阶数：1, 2, 3, 4, ...
        self.maxMemory = 50

        self.memoryStart = 0

        # self.tests = None
        # self.stationaryTests = None
        # self.oscillationTests = None
        self.AVName = None
        self.HVName = None

        self.delta_t = 0.1

        self.numFolds = 5  # 训练集和测试集合数据划分
        self.numVar = 3  # 决策变量的个数：距离、速度差、速度

        self.individualRegressFlag = True

    def addressOneTra(self, oneTra):
        xData = []  # 自变量数据
        yData = []  # 应变量数据
        for t in range(self.memory, oneTra.shape[0]):  # t对应行数
            oneX = []
            for lag in range(self.memory+1):
                oneLag = []
                # print(lag)
                oneLag.append(oneTra['spacing'].iloc[t - lag])
                oneLag.append(oneTra['speed difference'].iloc[t - lag])
                oneLag.append(oneTra['speed'].iloc[t - lag])
                oneX.append(oneLag)
            xData.append(oneX)
            yData.append([oneTra['acceleration'].iloc[t]])
            ddebug = 1

        return xData, yData

    def reorganizeDataAllVeh(self):
        AVData = {"x": [], "y": []}
        HVData = {"x": [], "y": []}
        for testScenario in self.CF_Data.keys():
            for veh in self.CF_Data[testScenario].keys():
                oneTra = self.CF_Data[testScenario][veh]
                x_oneTra, y_oneTra = self.addressOneTra(oneTra)
                if veh in self.AVName:
                    AVData["x"] += x_oneTra
                    AVData["y"] += y_oneTra
                if veh in self.HVName:
                    HVData["x"] += x_oneTra
                    HVData["y"] += y_oneTra
                debug = 1

        return {'AVData': AVData, 'HVData': HVData}

    def reorganizeDataInidividualVeh(self):
        reorganizedData = {}
        for testScenario in self.CF_Data.keys():
            reorganizedData_oneScenario = {}
            for veh in self.CF_Data[testScenario].keys():
                IniData = {"x": [], "y": []}
                oneTra = self.CF_Data[testScenario][veh]
                x_oneTra, y_oneTra = self.addressOneTra(oneTra)
                IniData["x"] += x_oneTra
                IniData["y"] += y_oneTra
                reorganizedData_oneScenario[veh] = IniData
            reorganizedData[testScenario] = reorganizedData_oneScenario
            debug = 1

        return reorganizedData

    def linearRegression(self, xData, yData):
        kf = KFold(n_splits=self.numFolds, shuffle=True, random_state=42)
        evalR2List = []
        xData_np = np.array(xData)
        xData_np_reshape = xData_np.reshape(xData_np.shape[0], -1)
        yData_np = np.array(yData)
        for fold, (train_index, val_index) in enumerate(kf.split(xData_np_reshape)):
            print('-----fold: ', fold)
            # 划分训练集和和测试集合
            train_index = train_index.astype(int)
            val_index = val_index.astype(int)
            x_train, x_val = xData_np_reshape[train_index], xData_np_reshape[val_index]
            y_train, y_val = yData_np[train_index], yData_np[val_index]
            debug = 1
            # 模型对象
            linearModel = LinearRegression()
            # 拟合
            linearModel.fit(x_train, y_train)
            # 评价
            r_squared = linearModel.score(x_val, y_val)
            # 计算adjusted r_square
            n_temp = xData_np_reshape.shape[0]  # number of observations
            k_temp = self.numVar * (self.memory + 1)  # number of predictors
            adjusted_r_squared = 1 - (1 - r_squared) * (n_temp - 1) / (n_temp - k_temp - 1)
            # 保存结果
            evalR2List.append(adjusted_r_squared)
            # print(r_squared)
            # print(adjusted_r_squared)

        return np.average(evalR2List)

    def findOrder(self):
        R2Dict = {}
        R2DictTrans = {}
        if self.individualRegressFlag:
            for memory_count in range(self.memoryStart, self.maxMemory + 1):
                # 遍历每一个阶数
                self.memory = memory_count
                print('-------------------------memory: ', self.memory)
                R2DictOneOrder = {}
                # 获取这一阶数下的数据
                allData = self.reorganizeDataInidividualVeh()
                for testScenario in allData.keys():
                    R2DictOneOrderOneSce = {}
                    for veh in allData[testScenario].keys():
                        print('-------------vehType: ', veh)
                        R2 = self.linearRegression(allData[testScenario][veh]['x'], allData[testScenario][veh]['y'])
                        R2DictOneOrderOneSce[veh] = R2
                    R2DictOneOrder[testScenario] = R2DictOneOrderOneSce
                R2Dict[memory_count] = R2DictOneOrder

            testSce = R2Dict[list(R2Dict.keys())[0]].keys()
            for sce in testSce:
                R2DictTrans[sce] = {}
            for order in R2Dict.keys():
                for sce in testSce:
                    R2DictTrans[sce][order] = R2Dict[order][sce]
            for key in R2DictTrans.keys():
                dfOneSce = pd.DataFrame(R2DictTrans[key])
                dfOneSce.to_excel('Results_LinearRegression2/OpenACCData/' + key + '+' + 'DeltaT0.1sMemory2.xlsx')

        else:
            for memory_count in range(self.memoryStart, self.maxMemory + 1):
                # 遍历每一个阶数
                self.memory = memory_count
                print('-------------------------memory: ', self.memory)
                R2DictOneOrder = {}
                # 获取这一阶数下的数据
                allData = self.reorganizeDataAllVeh()
                for vehType in allData.keys():
                    print('-------------vehType: ', vehType)
                    if len(allData[vehType]['x']) != 0:
                        R2 = self.linearRegression(allData[vehType]['x'], allData[vehType]['y'])
                    else:
                        R2 = -9999
                    R2DictOneOrder[vehType] = R2
                R2Dict[memory_count] = R2DictOneOrder
            keyTemp = R2Dict[list(R2Dict.keys())[0]].keys()
            for veh_type in keyTemp:
                R2DictTrans[veh_type] = {}
            for order in R2Dict.keys():
                for veh_type in keyTemp:
                    R2DictTrans[veh_type][order] = R2Dict[order][veh_type]
            for key in R2DictTrans.keys():
                pd.DataFrame(R2DictTrans[key], index=[0]).to_excel(
                    'Results_LinearRegression2/OpenACCData/' + key + '+' + 'DeltaT0.1sMemory2.xlsx')

        return

if __name__ == '__main__':

    # Open ACC Data
    m_OpenAccData = OpenAccData()
    m_OpenAccData.structureTransfer()
    debug = 1
    m_linearCFModelRegress = linearCFModelRegress(m_OpenAccData.CFData)
    m_linearCFModelRegress.AVName = ['2', '3', '4', '5']
    m_linearCFModelRegress.HVName = []
    m_linearCFModelRegress.findOrder()



