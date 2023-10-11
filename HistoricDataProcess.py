
import pandas as pd
import numpy as np

class HistoricData:

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
                'E:/UW-Madison/Research/Trajectory_Memory_Check/Filed-Experiment-Data-ACC_Data-main/'+self.resFileName+'.xlsx',
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
