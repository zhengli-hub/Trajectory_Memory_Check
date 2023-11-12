

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statistics

from ACCDataProcess import ACCData
from HistoricDataProcess import HistoricData
from VanderbiltDataProcess import VanderbiltData

class GRUCFModelRegress:

    # CF_Data: all tra info
    # oneTra: one tra info

    def __init__(self, CF_Data):
        self.CF_Data = CF_Data
        self.memory = None  # 后延的阶数：1, 2, 3, 4, ...
        self.maxMemory = 20
        self.numVar = 3  # 决策变量的个数：距离、速度差、速度
        self.rangeStateList = range(1, 11)  # 训练集和测试集合数据划分

    def reorganizeData(self, oneTra):
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

    def gruRegressionWithSplit(self, xData, yData):
        r2List = []
        for randomState in self.rangeStateList:
            # 划分训练集和和测试集合
            xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=randomState)
            xTrain = np.array(xTrain)
            xTest = np.array(xTest)
            yTrain = np.array(yTrain)
            yTest = np.array(yTest)
            debug = 1

            # 模型对象
            model = Sequential()
            # GRU层
            model.add(GRU(units=32, input_shape=(self.maxMemory, self.numVar), activation='relu', return_sequences=True))
            model.add(GRU(units=32, activation='relu'))
            # 输出层
            model.add(Dense(units=1))
            # 定义call back，模型精度没有时提前终止训练
            callbacks_set = [EarlyStopping(monitor='loss', min_delta=0.0001, patience=60, mode='min', verbose=0)]
            # 编译模型
            model.compile(optimizer='adam', loss='mse')
            # 训练模型
            model.fit(xTrain, yTrain, epochs=1000, batch_size=8, callbacks=callbacks_set, verbose=0)
            # 模型评价
            yPredict = model.predict(xTest)
            r2 = r2_score(yTest, yPredict)
            r2List.append(r2)

        return statistics.mean(r2List)

    def gruRegressionWithoutSplit(self, xData, yData):
        # 模型对象
        model = Sequential()
        # RNN层
        model.add(GRU(units=32, input_shape=(self.maxMemory, self.numVar), activation='relu', return_sequences=True))
        model.add(GRU(units=32, activation='relu'))
        # 输出层
        model.add(Dense(units=1))
        # 定义call back，模型精度没有时提前终止训练
        callbacks_set = [EarlyStopping(monitor='loss', min_delta=0.0001, patience=60, mode='min', verbose=0)]
        # 编译模型
        model.compile(optimizer='adam', loss='mse')
        # 训练模型
        model.fit(xData, yData, epochs=1000, batch_size=8, callbacks=callbacks_set, verbose=0)
        # 模型评价
        yPredict = model.predict(xData)
        r2 = r2_score(yData, yPredict)
        # final_loss, final_accuracy = model.evaluate(xData, yData)

        return r2

    def findOrder(self, oneTra):
        accuracy_list = []
        for memory_count in range(0, self.maxMemory + 1):
            # 遍历每一个阶数
            self.memory = memory_count
            print(self.memory)
            # 准备数据
            xData, yData = self.reorganizeData(oneTra)
            debug = 1
            # 拟合线性回归模型
            accuracy = self.gruRegressionWithoutSplit(xData, yData)
            accuracy_list.append(accuracy)

        return accuracy_list

    def enmurateAllTra(self):
        allRes = []
        for count in self.CF_Data.keys():
            print(count)
            # 提取一条跟驰轨迹
            oneTra = self.CF_Data[count]
            # 评估记忆阶数
            allRes.append(self.findOrder(oneTra))

        return allRes

if __name__ == '__main__':

    # Cats ACC Data
    m_ACCData = ACCData()
    m_ACCData.allPeriod()
    m_ACCData.CFDataProcess()
    CFDataAll = m_ACCData.CFData
    scenarioList = CFDataAll.keys()
    for scenario in scenarioList:
        CFData = CFDataAll[scenario]
        m_gruCFModelRegress = GRUCFModelRegress(CFData)
        allRes = m_gruCFModelRegress.enmurateAllTra()
        allResDF = pd.DataFrame(allRes)
        allResDF.to_excel('Results_GRURegression/CatsACCData/' + scenario + '+' + 'Memory20.xlsx')

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