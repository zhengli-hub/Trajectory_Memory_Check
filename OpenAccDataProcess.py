
import pandas as pd
import numpy as np

class OpenAccData:
    def __init__(self):
        self.cars = ['1', '2', '3', '4', '5']
        self.tests = []
        self.tests = ['example_origin']
        # self.stationaryTests = ['test1118/test1', 'test1118/test2', 'test1124/test1', 'test1124/test2', 'test1124/test3']
        # self.oscillationTests = ['test1118/test3', 'test1118/test4', 'test1118/test5','test1124/test6', 'test1124/test7',
        #                          'test1124/test8', 'test1124/test9', 'test1124/10']
        self.delta_t = 0.1

        self.CFData = {}

    def originDataProcess(self, testCount):  # 原数据

        print(testCount)
        #  read data from csv
        dfTemp = pd.read_csv(
            'E:/UW-Madison/Research/Trajectory_Memory_Check/OpenACC/data/'+testCount+'.csv',
            index_col=None, header=5)
        # delete the trajectory with human driven, keep acc data
        columns_to_check = ['Driver'+name for name in self.cars]
        target_string = 'ACC'
        condition = dfTemp[columns_to_check].ne(target_string).any(axis=1)
        dfTemp = dfTemp[~condition]
        # delete useless columns
        columns_to_drop = [f'{att}{name}' for name in self.cars for att in ['Lat', 'Lon', 'Alt', 'E', 'N', 'U', 'Driver']]
        dfTemp = dfTemp.drop(columns=columns_to_drop)
        # calculate speed difference
        print(len(self.cars)-1)
        for nameCount in range(len(self.cars)-1):
            dfTemp['DELTAV'+self.cars[nameCount]] = - (dfTemp['Speed'+self.cars[nameCount]] - dfTemp['Speed'+self.cars[nameCount+1]])
        # change the column name for spacing and speed difference
        column_name_mapping = {}
        for nameCount in range(len(self.cars)-1):
            column_name_mapping['IVS'+self.cars[nameCount]] = 'IVS'+self.cars[nameCount+1]
            column_name_mapping['DELTAV' + self.cars[nameCount]] = 'DELTAV' + self.cars[nameCount + 1]
        dfTemp = dfTemp.rename(columns=column_name_mapping)
        # calculate accelerate
        for name in self.cars:
            dfTemp['ACC'+name] = (dfTemp['Speed'+name] - dfTemp['Speed'+name].shift(1)) / self.delta_t

        debug = 1

        return dfTemp

    def structureTransfer(self):  # 跟驰数据
        for testCount in self.tests:
            print(testCount)
            oneTestDF = self.originDataProcess(testCount)
            CFDataOneTest = {}
            for count in range(1, len(self.cars)):
                name = self.cars[count]  # 主车
                columnName = ['Time'] + [f'{att}{name}' for att in ['Speed', 'IVS', 'DELTAV', 'ACC']]
                CFDataOneTest[name] = oneTestDF[columnName]
                # delete unstable speed
                CFDataOneTest[name] = CFDataOneTest[name][CFDataOneTest[name]['Speed'+name] > 10]
                # rename columns
                nameMapping = {'Speed'+name: 'speed',
                               'DELTAV'+name: 'speed difference',
                               'IVS'+name: 'spacing',
                               'ACC'+name: 'acceleration',
                               'Time': 'time'}
                CFDataOneTest[name] = CFDataOneTest[name].rename(columns=nameMapping)
            self.CFData[testCount] = CFDataOneTest
            debug = 1

if __name__ == '__main__':

    m_ACCData = OpenAccData()
    m_ACCData.structureTransfer()
    # m_ACCData.CFDataProcess()
    # CFDataAll = m_ACCData.CFData
    # scenarioList = CFDataAll.keys()

    debug = 1
