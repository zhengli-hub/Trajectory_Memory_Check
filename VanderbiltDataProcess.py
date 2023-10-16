
import pandas as pd
import numpy as np

class VanderbiltData:

    def __init__(self):
        self.cars = ['Veh'+str(i) for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
        self.tests = ['osc_max', 'osc_min', 'stepH_max', 'stepH_min', 'stepL_max', 'stepL_min']
        # self.tests = ['osc_min']
        self.stationaryTests = ['osc_max', 'osc_min']
        self.oscillationTests = ['stepH_max', 'stepH_min', 'stepL_max', 'stepL_min']
        self.delta_t = 1

        self.CFData = {}

    def CFDataProcess(self):  # 原数据
        for testCount in self.tests:
            CFDataOneTest = {}
            for carCount in self.cars:
                dfTemp = pd.read_csv(
                        'E:/UW-Madison/Research/Trajectory_Memory_Check/Filed-Experiment-Vanderbilt-ACC-Data-two-vehicle/'
                        + carCount + '_' + testCount + '.csv', index_col=None, header=None)
                dfTemp.columns = ['time new (s)', 'speed (m/s)', 'front speed (m/s)', 'distance (m)']
                # 数据保留1秒
                dfTemp = dfTemp[dfTemp['time new (s)'] % self.delta_t == 0]
                # 检查数据是否齐全，不全的时间找出来
                # 找出所有的空缺数据
                missTime = []
                for index in range(dfTemp.shape[0] - 1):
                    timeLag = int(dfTemp['time new (s)'].iloc[index + 1] - dfTemp['time new (s)'].iloc[index])
                    if timeLag > 1:
                        for count in range(timeLag - 1):
                            missTime.append(dfTemp['time new (s)'].iloc[index] + count + 1)
                # 在空缺位置加入新的列表
                dfMiss = pd.DataFrame({'time new (s)': missTime,
                                       'speed (m/s)': [np.nan] * len(missTime),
                                       'front speed (m/s)': [np.nan] * len(missTime),
                                       'distance (m)': [np.nan] * len(missTime)})
                # 将新表加入
                CFDataOneTest[carCount] = dfTemp.append(dfMiss, ignore_index=True)
                # 排序
                CFDataOneTest[carCount].sort_values(by=['time new (s)'], inplace=True)
                # 插值
                CFDataOneTest[carCount] = CFDataOneTest[carCount].interpolate()
                # 计算加速度
                CFDataOneTest[carCount]['acceleration (m/s2)'] = None
                for index in range(CFDataOneTest[carCount].shape[0] - 1):
                    # a = (v2 - v1) / (t2 - t1)
                    v2 = CFDataOneTest[carCount]['speed (m/s)'].iloc[index + 1]
                    v1 = CFDataOneTest[carCount]['speed (m/s)'].iloc[index]
                    CFDataOneTest[carCount]['acceleration (m/s2)'].iloc[index] = (v2 - v1) / self.delta_t
                # 最后一个加速度假设一下
                CFDataOneTest[carCount]['acceleration (m/s2)'].iloc[-1] = CFDataOneTest[carCount]['acceleration (m/s2)'].iloc[-2]
            # 一个测试的所有数据加入
            self.CFData[testCount] = CFDataOneTest
            debug = 1
        debug = 1

        return

if __name__ == '__main__':
    m_VanderbiltData = VanderbiltData()
    m_VanderbiltData.CFDataProcess()
    print('done!------allPeriod()')
    # m_HistoricData.CFDataProcess()
    # CFData = m_HistoricData.CFDataList

    debug = 1