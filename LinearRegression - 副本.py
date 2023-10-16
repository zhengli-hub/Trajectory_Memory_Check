
import scipy.optimize
import sklearn.metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from ACCDataProcess import ACCData
from HistoricDataProcess import HistoricData
from VanderbiltDataProcess import VanderbiltData

class linearCFModelRegress:

    # CF_Data: all tra info
    # oneTra: one tra info

    def __init__(self, CF_Data):
        self.CF_Data = CF_Data
        self.memory = None  # 后延的阶数：1, 2, 3, 4, ...
        self.maxMemory = 80

    def reorganizeData(self, oneTra):
        xData = []  # 自变量数据
        yData = []  # 应变量数据
        for t in range(self.memory, int(oneTra['time new (s)'].iloc[-1] - oneTra['time new (s)'].iloc[0])):
            oneX = []
            for lag in range(self.memory+1):
                # print(lag)
                # xData: [0]headway, [1]frontLength, [2]frontSpeed, [3]followSpeed
                # oneX.append(oneTra['tra']['SpaceHeadway'].iloc[t - lag])  # headway
                # oneX.append(oneTra['tra']['precedingLength'].iloc[t - lag])  # frontLength
                oneX.append(oneTra['distance (m)'].iloc[t - lag])
                oneX.append(oneTra['front speed (m/s)'].iloc[t - lag])  # frontSpeed
                oneX.append(oneTra['speed (m/s)'].iloc[t - lag])  # followSpeed
            xData.append(oneX)
            yData.append(oneTra['acceleration (m/s2)'].iloc[t])
            ddebug = 1

        return xData, yData

    def lineRegression(self, xData, yData):
        linearModel = LinearRegression()
        linearModel.fit(xData, yData)
        score = linearModel.score(xData, yData)
        debug = 1

        return score

    def findOrder(self, oneTra):
        r2_list = []
        for memory_count in range(0, self.maxMemory + 1):
            # 遍历每一个阶数
            self.memory = memory_count
            # 准备数据
            xData, yData = self.reorganizeData(oneTra)
            debug = 1
            # # 拟合模型
            # fitParams = self.CF_ModelParaFit(xData, yData)
            # fitParams = list(fitParams)
            # # print(fitParams)
            # # 评价
            # r2 = self.CF_ModelEva(xData, yData, fitParams)
            # r2_list.append(r2)
            # print(r2)

            # 拟合线性回归模型
            r2 = self.lineRegression(xData, yData)
            r2_list.append(r2)

        return r2_list

    # def enmurateAllTra(self):
    #     allRes = []
    #     for count in range(len(self.CF_Data)):
    #         # 提取一条跟驰轨迹
    #         oneTra = self.CF_Data[count]
    #         # 评估记忆阶数
    #         allRes.append(self.findOrder(oneTra))
    #
    #     return allRes

    def enmurateAllTra(self):
        allRes = []
        for count in self.CF_Data.keys():
            # 提取一条跟驰轨迹
            oneTra = self.CF_Data[count]
            # 评估记忆阶数
            allRes.append(self.findOrder(oneTra))

        return allRes

if __name__ == '__main__':

    # # Cats ACC Data
    # fileFolder = 'test1118'
    # fileName = 'test5'
    # m_ACCData = ACCData(resFileName=fileFolder + '/' + fileName)
    # m_ACCData.allPeriod()
    # m_ACCData.CFData()
    # CFData = m_ACCData.CFDataList
    # m_linearCFModelRegress = linearCFModelRegress(CFData)
    # allRes = m_linearCFModelRegress.enmurateAllTra()
    # allResDF = pd.DataFrame(allRes)
    # allResDF.to_excel('Results_LinearRegression/' + fileFolder + '+' + fileName + 'Memory80.xlsx')

    # Vanderbilt Data
    m_VanderbiltData = VanderbiltData()
    m_VanderbiltData.CFDataProcess()
    CFDataAll = m_VanderbiltData.CFData
    scenarioList = CFDataAll.keys()
    for scenario in scenarioList:
        CFData = CFDataAll[scenario]
        m_linearCFModelRegress = linearCFModelRegress(CFData)
        allRes = m_linearCFModelRegress.enmurateAllTra()
        allResDF = pd.DataFrame(allRes)
        allResDF.to_excel('Results_LinearRegression/VanderbiltData/' + scenario + '+' + 'Memory80.xlsx')

    #
    #
    # CFData = m_ACCData.CFDataList
    # m_linearCFModelRegress = linearCFModelRegress(CFData)
    # allRes = m_linearCFModelRegress.enmurateAllTra()
    # allResDF = pd.DataFrame(allRes)
    # allResDF.to_excel('Results_LinearRegression/' + fileFolder + '+' + fileName + 'Memory80.xlsx')
