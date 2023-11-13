
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statistics

from ACCDataProcess import ACCData
from HistoricDataProcess import HistoricData
from VanderbiltDataProcess import VanderbiltData

class RNNCFModelRegress2:

    # CF_Data: all tra info
    # oneTra: one tra info

    def __init__(self, CF_Data):
        self.CF_Data = CF_Data
        self.memory = None  # 后延的阶数：1, 2, 3, 4, ...
        self.maxMemory = 20
        self.numVar = 3  # 决策变量的个数：距离、速度差、速度
        self.numFolds = 5  # 训练集和测试集合数据划分

        self.trainEpochs = 100
        self.memoryStart = 0

        self.tests = ['test1118/test1', 'test1118/test2', 'test1118/test3', 'test1118/test4', 'test1118/test5',
                      'test1124/test1', 'test1124/test2', 'test1124/test3', 'test1124/test6', 'test1124/test7',
                      'test1124/test8', 'test1124/test9', 'test1124/test10']
        self.stationaryTests = ['test1118/test1', 'test1118/test2', 'test1124/test1', 'test1124/test2', 'test1124/test3']
        self.oscillationTests = ['test1118/test3', 'test1118/test4', 'test1118/test5','test1124/test6', 'test1124/test7',
                                 'test1124/test8', 'test1124/test9', 'test1124/10']

        self.AVName = ['veh 2', 'veh 3']
        self.HVName = ['veh 4', 'veh 5']

    def addressOneTra(self, oneTra):
        xData = []  # 自变量数据
        yData = []  # 应变量数据
        for t in range(self.memory, int(oneTra['time new (s)'].iloc[-1] - oneTra['time new (s)'].iloc[0])):
            oneX = []
            for lag in range(self.memory+1):
                oneLag = []
                # print(lag)
                # xData: [0]headway, [1]frontLength, [2]frontSpeed, [3]followSpeed
                # oneX.append(oneTra['tra']['SpaceHeadway'].iloc[t - lag])  # headway
                # oneX.append(oneTra['tra']['precedingLength'].iloc[t - lag])  # frontLength
                oneLag.append(oneTra['distance (m)'].iloc[t - lag])
                oneLag.append(oneTra['front speed (m/s)'].iloc[t - lag])  # frontSpeed
                oneLag.append(oneTra['speed (m/s)'].iloc[t - lag])  # followSpeed
                oneX.append(oneLag)
            xData.append(oneX)
            yData.append([oneTra['acceleration (m/s2)'].iloc[t]])
            ddebug = 1

        return xData, yData

    def reorganizeData(self):
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

    def rnnRegression(self, xData, yData):
        kf = KFold(n_splits=self.numFolds, shuffle=True, random_state=42)
        evalAccuracyList = []
        for fold, (train_index, val_index) in enumerate(kf.split(xData)):
            print('-----fold: ', fold)
            # 划分训练集和和测试集合
            train_index = train_index.astype(int)
            val_index = val_index.astype(int)
            xData = np.array(xData)
            yData = np.array(yData)
            x_train, x_val = xData[train_index], xData[val_index]
            y_train, y_val = yData[train_index], yData[val_index]
            debug = 1
            # 模型对象
            model = Sequential()
            # GRU层
            model.add(SimpleRNN(units=30, input_shape=(self.memory + 1, self.numVar), activation='tanh', return_sequences=True))
            model.add(SimpleRNN(units=10, activation='tanh', return_sequences=True))
            model.add(SimpleRNN(units=10, activation='tanh', return_sequences=True))
            # 输出层
            model.add(Dense(units=1))
            # 定义call back，模型精度没有时提前终止训练
            callbacks_set = [EarlyStopping(monitor='loss', min_delta=0.001, patience=50, mode='min', verbose=0)]
            # 编译模型
            model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # 训练模型
            print('训练模型')
            model.fit(x_train, y_train, epochs=self.trainEpochs, callbacks=callbacks_set, verbose=1, validation_data=(x_val, y_val))
            print('训练完了')
            # model.fit(X_train, y_train, epochs=1000, batch_size=8, callbacks=callbacks_set, verbose=0, validation_data=(X_val, y_val))
            # 模型评价
            # yPredict = model.predict(xTest)
            # r2 = r2_score(yTest, yPredict)
            # r2List.append(r2)
            evalLoss, evalAccuracy = model.evaluate(x_val, y_val)
            evalAccuracyList.append(evalAccuracy)

        return np.average(evalAccuracyList)

    def findOrder(self):
        # allData = self.reorganizeData()
        # accuracyDict = {}
        # for vehType in allData.keys():
        #     accuracyList = []
        #     for memory_count in range(0, self.maxMemory + 1):
        #         # 遍历每一个阶数
        #         self.memory = memory_count
        #         print(self.memory)
        #         accuracy = self.gruRegression(allData[vehType]['x'], allData[vehType]['y'])
        #         accuracyList.append(accuracy)
        #     accuracyDict[vehType] = accuracyList

        accuracyDict = {}
        for memory_count in range(self.memoryStart, self.maxMemory + 1):
            # 遍历每一个阶数
            self.memory = memory_count
            print('-------------------------memory: ', self.memory)
            accuracyDictOneOrder = {}
            # 获取这一阶数下所有的数据
            allData = self.reorganizeData()
            for vehType in allData.keys():
                print('-------------vehType: ', vehType)
                accuracy = self.rnnRegression(allData[vehType]['x'], allData[vehType]['y'])
                accuracyDictOneOrder[vehType] = accuracy
            accuracyDict[memory_count] = accuracyDictOneOrder

        keyTemp = accuracyDict[list(accuracyDict.keys())[0]].keys()
        accuracyDictTrans = {}
        for veh_type in keyTemp:
            accuracyDictTrans[veh_type] = {}
        for order in accuracyDict.keys():
            for veh_type in keyTemp:
                accuracyDictTrans[veh_type][order] = accuracyDict[order][veh_type]

        return accuracyDictTrans

if __name__ == '__main__':

    # Cats ACC Data
    m_ACCData = ACCData()
    m_ACCData.allPeriod()
    m_ACCData.CFDataProcess()
    CFDataAll = m_ACCData.CFData
    m_rnnCFModelRegress = RNNCFModelRegress2(CFDataAll)
    accuracyDict = m_rnnCFModelRegress.findOrder()
    for key in accuracyDict.keys():
        pd.DataFrame(accuracyDict[key], index=[0]).to_excel('Results_RNNRegression2/CatsACCData/' + key + '+' + 'Memory20.xlsx')

    # for scenario in scenarioList:
    #     CFData = CFDataAll[scenario]
    #     m_gruCFModelRegress = GRUCFModelRegress(CFData)
    #     allRes = m_gruCFModelRegress.enmurateAllTra()
    #     allResDF = pd.DataFrame(allRes)
    #     allResDF.to_excel('Results_GRURegression/CatsACCData/' + scenario + '+' + 'Memory20.xlsx')

    # # Vanderbilt Data
    # m_VanderbiltData = VanderbiltData()
    # m_VanderbiltData.CFDataProcess()
    # CFDataAll = m_VanderbiltData.CFData
    # scenarioList = CFDataAll.keys()
    # for scenario in scenarioList:
    #     CFData = CFDataAll[scenario]
    #     m_linearCFModelRegress = linearCFModelRegress(CFData)
    #     allRes = m_linearCFModelRegress.enmurateAllTra()
    #     allResDF = pd.DataFrame(allRes)
    #     allResDF.to_excel('Results_LinearRegression/VanderbiltData/' + scenario + '+' + 'Memory80.xlsx')

    # # HistoricData Data
    # m_HistoricData = HistoricData()
    # print('done1')
    # m_HistoricData.allPeriod()
    # print('done2')
    # m_HistoricData.CFDataProcess()
    # print('done3')
    # CFDataAll = m_HistoricData.CFData
    # scenarioList = CFDataAll.keys()
    # for scenario in scenarioList:
    #     CFData = CFDataAll[scenario]
    #     m_linearCFModelRegress = linearCFModelRegress(CFData)
    #     allRes = m_linearCFModelRegress.enmurateAllTra()
    #     allResDF = pd.DataFrame(allRes)
    #     allResDF.to_excel('Results_LinearRegression/HistoricData/' + scenario + '+' + 'Memory80.xlsx')