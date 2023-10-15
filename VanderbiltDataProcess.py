
import pandas as pd
import numpy as np

class VanderbiltData:

    def __init__(self):
        self.cars = ['Veh'+str(i) for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
        # self.tests = ['osc max', 'osc min', 'stepH max', 'stepH min', 'stepL max', 'stepL min']
        self.tests = ['osc max']
        self.stationaryTests = ['osc max', 'osc min']
        self.oscillationTests = ['stepH max', 'stepH min', 'stepL max', 'stepL min']
        self.delta_t = 1

        self.orginDdata = {}
        self.CFData = {}

