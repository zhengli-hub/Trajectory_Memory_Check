
from geopy.distance import geodesic
import pandas as pd
import numpy as np

class ACCData:
    def __init__(self):
        self.cars = ['veh 1', 'veh 2', 'veh 3', 'veh 4', 'veh 5']
        self.tests = ['test1118/test1', 'test1118/test2', 'test1118/test3', 'test1118/test4', 'test1118/test5',
                      'test1124/test1', 'test1124/test2', 'test1124/test3', 'test1124/test6', 'test1124/test7',
                      'test1124/test8', 'test1124/test9', 'test1124/test10']
        # self.tests = ['test1118/test1']
        self.stationaryTests = ['test1118/test1', 'test1118/test2', 'test1124/test1', 'test1124/test2', 'test1124/test3']
        self.oscillationTests = ['test1118/test3', 'test1118/test4', 'test1118/test5','test1124/test6', 'test1124/test7',
                                 'test1124/test8', 'test1124/test9', 'test1124/10']
        self.delta_t = 1

        self.orginDdata = {}
        self.CFData = {}

    def allPeriod(self):  # 原数据
        for testCount in self.tests:
            print(testCount)
            originDataOneTest = {}
            for carCount in self.cars:
                # 读取数据
                dfTemp = pd.read_excel(
                    'E:/UW-Madison/Research/Trajectory_Memory_Check/Filed-Experiment-Data-ACC_Data-main/'+testCount+'.xlsx',
                    sheet_name=carCount, index_col=0)
                dfTemp.columns = ['time (s)', 'longitude (deg)', 'latitude (deg)', 'speed (m/s)']
                # 删去时间指标中无用的字符
                dfTemp['time new (s)'] = None
                for count in range(dfTemp.shape[0]):
                    dfTemp['time new (s)'].iloc[count] = float(dfTemp['time (s)'].iloc[count][5:])
                # 数据保留1秒
                dfTemp = dfTemp[dfTemp['time new (s)'] % 1 == 0]
                # 检查数据是否齐全，不全的时间找出来
                # 找出所有空缺数据
                missTime = []
                for index in range(dfTemp.shape[0] - 1):
                    timeLag = int(dfTemp['time new (s)'].iloc[index + 1] - dfTemp['time new (s)'].iloc[index])
                    if timeLag > 1:
                        for count in range(timeLag - 1):
                            missTime.append(dfTemp['time new (s)'].iloc[index] + count + 1)
                # 在空缺位置加入新的列表
                dfMiss = pd.DataFrame({'time (s)': [np.nan] * len(missTime),
                                       'longitude (deg)': [np.nan] * len(missTime),
                                       'latitude (deg)': [np.nan] * len(missTime),
                                       'speed (m/s)': [np.nan] * len(missTime),
                                       'time new (s)': missTime})
                # 将新表加入
                originDataOneTest[carCount] = dfTemp.append(dfMiss, ignore_index=True)
                # 排序
                originDataOneTest[carCount].sort_values(by=['time new (s)'], inplace=True)
                # 插值
                originDataOneTest[carCount]['speed (m/s)'] = originDataOneTest[carCount]['speed (m/s)'].astype(float)
                originDataOneTest[carCount] = originDataOneTest[carCount].interpolate()
                # 计算加速度
                originDataOneTest[carCount]['acceleration (m/s2)'] = None
                for index in range(originDataOneTest[carCount].shape[0] - 1):
                    # a = (v2 - v1) / (t2 - t1)
                    v2 = originDataOneTest[carCount]['speed (m/s)'].iloc[index + 1]
                    v1 = originDataOneTest[carCount]['speed (m/s)'].iloc[index]
                    originDataOneTest[carCount]['acceleration (m/s2)'].iloc[index] = (v2 - v1) / self.delta_t
                # 最后一个加速度假设一下
                originDataOneTest[carCount]['acceleration (m/s2)'].iloc[-1] = \
                originDataOneTest[carCount]['acceleration (m/s2)'].iloc[-2]
                # 一个测试的所有数据加入
            self.orginDdata[testCount] = originDataOneTest
            debug = 1
        debug = 1

    def CFDataProcess(self):  # 跟驰数据
        for testCount in self.tests:
            print(testCount)
            CFDataOneTest = {}
            for count in range(1, len(self.cars)):
                subVeh = self.cars[count]  # 主车
                frontVeh = self.cars[count - 1]  # 前车

                # 本车与前车的公共时间
                subVehStart = self.orginDdata[testCount][subVeh]['time new (s)'].iloc[0]
                subVehEnd = self.orginDdata[testCount][subVeh]['time new (s)'].iloc[-1]
                frontVehStart = self.orginDdata[testCount][frontVeh]['time new (s)'].iloc[0]
                frontVehEnd = self.orginDdata[testCount][frontVeh]['time new (s)'].iloc[-1]
                cfStart = max(subVehStart, frontVehStart)
                cfEnd = min(subVehEnd, frontVehEnd)
                # 根据公共时间获取公共时间下的数据
                subVehCFData = self.orginDdata[testCount][subVeh][(self.orginDdata[testCount][subVeh]['time new (s)'] >= cfStart) &
                (self.orginDdata[testCount][subVeh]['time new (s)'] <= cfEnd)]
                frontVehCFData = self.orginDdata[testCount][frontVeh][(self.orginDdata[testCount][frontVeh]['time new (s)'] >= cfStart) &
                                                  (self.orginDdata[testCount][frontVeh]['time new (s)'] <= cfEnd)]
                # 前面跟驰不稳定的速度删掉
                stableFollowTime = None
                for index_temp in range(subVehCFData.shape[0]):
                    if subVehCFData.iloc[index_temp, 3] > 10:  # 选择标准，速度大于10
                        stableFollowTime = subVehCFData.iloc[index_temp, 4]
                        break
                frontVehCFData = frontVehCFData[(frontVehCFData['time new (s)'] >= stableFollowTime)]
                subVehCFData = subVehCFData[(subVehCFData['time new (s)'] >= stableFollowTime)]

                debug = 1

                # 前车的数据添加到主车的数据中
                subVehCFData['front speed (m/s)'] = None
                subVehCFData['distance (m)'] = None

                for time in range(subVehCFData.shape[0]):
                    # 提取前车的速度，添加到主车速度中
                    frontVehSpeed = frontVehCFData['speed (m/s)'].iloc[time]
                    subVehCFData['front speed (m/s)'].iloc[time] = frontVehSpeed
                    # 提取前车的位置，计算两车的距离
                    frontVehLongitude = frontVehCFData['longitude (deg)'].iloc[time]
                    frontVehLatitude = frontVehCFData['latitude (deg)'].iloc[time]
                    subVehLongitude = subVehCFData['longitude (deg)'].iloc[time]
                    subVehLatitude = subVehCFData['latitude (deg)'].iloc[time]
                    # # 数据格式：(latitude, longitude)
                    distance = geodesic((subVehLatitude, subVehLongitude), (frontVehLatitude, frontVehLongitude)).km * 1000
                    subVehCFData['distance (m)'].iloc[time] = distance
                # debug = 1
                CFDataOneTest[subVeh] = subVehCFData
            self.CFData[testCount] = CFDataOneTest
            debug = 1

if __name__ == '__main__':

    m_ACCData = ACCData()
    m_ACCData.allPeriod()
    m_ACCData.CFDataProcess()
    CFDataAll = m_ACCData.CFData
    scenarioList = CFDataAll.keys()

    debug = 1
