
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

        self.orginDdata = {}
        self.CFData = {}

    def allPeriod(self):  # 原数据
        for testCount in self.tests:
            originDataOneTest = {}
            for carCount in self.cars:
                dfTemp = pd.read_csv(
                        'E:/UW-Madison/Research/Trajectory_Memory_Check/Filed-Experiment-Vanderbilt-ACC-Data-two-vehicle/'
                        + carCount + '_' + testCount + '.csv', index_col=None, header=None)
                dfTemp.columns = ['time (s)', 'subSpeed (m/s)', 'frontSpeed (m/s)', 'spacing (m)']
                # 数据保留1秒
                dfTemp = dfTemp[dfTemp['time (s)'] % 1 == 0]
                # 检查数据是否齐全，不全的时间找出来
                # 找出所有的空缺数据
                missTime = []
                for index in range(dfTemp.shape[0] - 1):
                    timeLag = int(dfTemp['time (s)'].iloc[index + 1] - dfTemp['time (s)'].iloc[index])
                    if timeLag > 1:
                        for count in range(timeLag - 1):
                            missTime.append(dfTemp['time (s)'].iloc[index] + count + 1)
                # 在空缺位置加入新的列表
                dfMiss = pd.DataFrame({'time (s)': missTime,
                                       'subSpeed (m/s)': [np.nan] * len(missTime),
                                       'frontSpeed (m/s)': [np.nan] * len(missTime),
                                       'spacing (m)': [np.nan] * len(missTime)})
                # 将新表加入
                originDataOneTest[carCount] = dfTemp.append(dfMiss, ignore_index=True)
                # 排序
                originDataOneTest[carCount].sort_values(by=['time (s)'], inplace=True)
                # 插值
                originDataOneTest[carCount] = originDataOneTest[carCount].interpolate()
                # 计算加速度
                originDataOneTest[carCount]['acceleration (m/s2)'] = None
                for index in range(originDataOneTest[carCount].shape[0] - 1):
                    # a = (v2 - v1) / (t2 - t1)
                    v2 = originDataOneTest[carCount]['subSpeed (m/s)'].iloc[index + 1]
                    v1 = originDataOneTest[carCount]['subSpeed (m/s)'].iloc[index]
                    originDataOneTest[carCount]['acceleration (m/s2)'].iloc[index] = (v2 - v1) / self.delta_t
                # 最后一个加速度假设一下
                originDataOneTest[carCount]['acceleration (m/s2)'].iloc[-1] = originDataOneTest[carCount]['acceleration (m/s2)'].iloc[-2]
            # 一个测试的所有数据加入
            self.orginDdata[testCount] = originDataOneTest
            debug = 1
        debug = 1

        return

if __name__ == '__main__':
    m_VanderbiltData = VanderbiltData()
    m_VanderbiltData.allPeriod()
    print('done!------allPeriod()')
    # m_HistoricData.CFDataProcess()
    # CFData = m_HistoricData.CFDataList

    debug = 1