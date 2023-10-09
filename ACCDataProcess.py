
from geopy.distance import geodesic
import pandas as pd
import numpy as np

class ACCData:

    def __init__(self, resFileName):
        self.cars = ['veh 1', 'veh 2', 'veh 3', 'veh 4', 'veh 5']
        self.orginDdata = {}
        self.delta_t = 1

        self.resFileName = resFileName

        self.CFDataList = []

    def allPeriod(self):  # 原数据
        for sheet in self.cars:
            # 读取数据
            dfTemp = pd.read_excel(
                'E:/UW-Madison/Research/TrajectoryMemoryCheck/Filed-Experiment-Data-ACC_Data-main/'+self.resFileName+'.xlsx',
                sheet_name=sheet, index_col=0)
            dfTemp.columns = ['time (s)', 'longitude (deg)', 'latitude (deg)', 'speed (m/s)']
            # 删去时间指标中无用的字符
            dfTemp['time new (s)'] = None
            for count in range(dfTemp.shape[0]):
                dfTemp['time new (s)'].iloc[count] = float(dfTemp['time (s)'].iloc[count][5:])
            # 数据保留1秒
            dfTemp = dfTemp[dfTemp['time new (s)'] % 1 == 0]

            # 检查数据是否齐全，不全的时间找出来
            missTime = []
            for index in range(dfTemp.shape[0] - 1):
                timeLag = int(dfTemp['time new (s)'].iloc[index + 1] - dfTemp['time new (s)'].iloc[index])
                if timeLag > 1:
                    for count in range(timeLag - 1):
                        missTime.append(dfTemp['time new (s)'].iloc[index] + count + 1)
            # 在空缺位置加入新的列表
            # dfMiss = None
            dfMiss = pd.DataFrame({'time (s)': [np.nan] * len(missTime),
                                   'longitude (deg)': [np.nan] * len(missTime),
                                   'latitude (deg)': [np.nan] * len(missTime),
                                   'speed (m/s)': [np.nan] * len(missTime),
                                   'time new (s)': missTime})
            self.orginDdata[sheet] = dfTemp.append(dfMiss, ignore_index=True)  # 将新表加入
            self.orginDdata[sheet].sort_values(by=['time new (s)'], inplace=True)  # 排序
            # 插值
            self.orginDdata[sheet] = self.orginDdata[sheet].interpolate()

            # 计算加速度
            self.orginDdata[sheet]['acceleration (m/s2)'] = None
            for index in range(self.orginDdata[sheet].shape[0] - 1):
                # a = (v2 - v1) / (t2 - t1)
                v2 = self.orginDdata[sheet]['speed (m/s)'].iloc[index + 1]
                v1 = self.orginDdata[sheet]['speed (m/s)'].iloc[index]
                self.orginDdata[sheet]['acceleration (m/s2)'].iloc[index] = (v2 - v1) / self.delta_t
            self.orginDdata[sheet]['acceleration (m/s2)'].iloc[-1] = self.orginDdata[sheet]['acceleration (m/s2)'].iloc[-2] # 最后一个加速度假设一下

    def CFData(self):  # 跟驰数据
        for count in range(1, len(self.cars)):

            subVeh = self.cars[count]
            frontVeh = self.cars[count - 1]

            # 本车与前车的公共时间
            subVehStart = self.orginDdata[subVeh]['time new (s)'].iloc[0]
            subVehEnd = self.orginDdata[subVeh]['time new (s)'].iloc[-1]
            frontVehStart = self.orginDdata[frontVeh]['time new (s)'].iloc[0]
            frontVehEnd = self.orginDdata[frontVeh]['time new (s)'].iloc[-1]
            cfStart = max(subVehStart, frontVehStart)
            cfEnd = min(subVehEnd, frontVehEnd)
            # 根据公共时间获取公共时间下的数据
            subVehCFData = self.orginDdata[subVeh][(self.orginDdata[subVeh]['time new (s)'] >= cfStart) &
            (self.orginDdata[subVeh]['time new (s)'] <= cfEnd)]
            frontVehCFData = self.orginDdata[frontVeh][(self.orginDdata[frontVeh]['time new (s)'] >= cfStart) &
                                              (self.orginDdata[frontVeh]['time new (s)'] <= cfEnd)]
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
            debug = 1
            self.CFDataList.append(subVehCFData)
            debug_2 = 1

if __name__ == '__main__':

    m_ACCData = ACCData(resFileName='test1124/test10')
    m_ACCData.allPeriod()
    m_ACCData.CFData()
    CFData = m_ACCData.CFDataList

    debug = 1
