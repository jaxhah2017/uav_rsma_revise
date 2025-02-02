import numpy as np

import matplotlib.pyplot as plt


def plot(map):
    uav_pos = map['pos_ubs']
    eve_pos = map['pos_eves']
    gts_pos = map['pos_gts']
    range_pos = map['range_pos']
    area = map['area']

    print(len(uav_pos))

    fig, ax = plt.subplots()
    ax.axis([0, range_pos, 0, range_pos])
    for (x, y) in uav_pos:
        ubs, = ax.plot(x, y, marker='o', color='b')
    for (x, y) in eve_pos:
        eve, = ax.plot(x, y, marker='o', color='r', markersize=5)
    for (x, y) in gts_pos:
        gts, = ax.plot(x, y, marker='o', color='y', markersize=5)
    ax.legend(handles=[ubs, eve, gts],
              labels=['uav_pos', 'eve_pos', 'gts_pos'],
              loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.75)
    # plt.show()
    plt.savefig('map.png')

class General4uavMap:
    def __init__(self, range_pos=400, n_eve=16, n_gt=20, n_uav=4, r_sense=np.inf):
        self.n_eve = n_eve
        self.n_gt = n_gt
        self.n_uav = n_uav
        self.pos_eves = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gt, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_uav, 2), dtype=np.float32)
        self.range_pos = range_pos
        self.fen = 4
        self.area = self.range_pos / self.fen
        self.r_sense = np.inf

    def set_eve(self):
        self.pos_eves[0] = [-500, -500]
        self.pos_eves[1] = [-500, -500]
        self.pos_eves[2] = [-500, -500]
        self.pos_eves[3] = [-500, -500]
        self.pos_eves[4] = [-500, -500]
        self.pos_eves[5] = [110, 110]
        self.pos_eves[6] = [280, 120]
        self.pos_eves[7] = [-500, -500]
        self.pos_eves[8] = [-500, -500]
        self.pos_eves[9] = [190, 290]
        self.pos_eves[10] = [220, 220]
        self.pos_eves[11] = [-500, -500]
        self.pos_eves[12] = [-500, -500]
        self.pos_eves[13] = [-500, -500]
        self.pos_eves[14] = [-500, -500]
        self.pos_eves[15] = [-500, -500]

    def set_gts(self):
        n_gt_in_cell = int(self.n_gt / 4)
        for i in range(n_gt_in_cell):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_cell, n_gt_in_cell * 2):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_cell * 2, n_gt_in_cell * 3):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_cell * 3, n_gt_in_cell * 4):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

    def set_ubs(self):
        self.pos_ubs[0] = [80, 80]
        self.pos_ubs[1] = [320, 80]
        self.pos_ubs[2] = [80, 320]
        self.pos_ubs[3] = [320, 320]

        # DEBUG
        # self.pos_ubs[0] = [150, 150]
        # self.pos_ubs[1] = [250, 150]
        # self.pos_ubs[2] = [150, 250]
        # self.pos_ubs[3] = [260, 260]

    def get_map(self, flag: int = 0):
        self.set_eve()
        if flag == 1:
            self.set_gts()
        self.set_ubs()

        return self.__dict__
    

class General2uavMap:
    def __init__(self, range_pos=400, n_eve=16, n_gt=20, n_uav=2, r_sense=np.inf):
        self.n_eve = n_eve
        self.n_gt = n_gt
        self.n_uav = n_uav
        self.pos_eves = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gt, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_uav, 2), dtype=np.float32)
        self.range_pos = range_pos
        self.fen = 4
        self.area = self.range_pos / self.fen
        self.r_sense = np.inf

    def set_eve(self):
        self.pos_eves[0] = [-500, -500]
        self.pos_eves[1] = [-500, -500]
        self.pos_eves[2] = [-500, -500]
        self.pos_eves[3] = [-500, -500]
        self.pos_eves[4] = [-500, -500]
        self.pos_eves[5] = [110, 110]
        self.pos_eves[6] = [280, 120]
        self.pos_eves[7] = [-500, -500]
        self.pos_eves[8] = [-500, -500]
        self.pos_eves[9] = [190, 290]
        self.pos_eves[10] = [220, 220]
        self.pos_eves[11] = [-500, -500]
        self.pos_eves[12] = [-500, -500]
        self.pos_eves[13] = [-500, -500]
        self.pos_eves[14] = [-500, -500]
        self.pos_eves[15] = [-500, -500]
        

    def set_gts(self):
        n_gt_in_cell = int(self.n_gt / 4)
        for i in range(n_gt_in_cell):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_cell, n_gt_in_cell * 2):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_cell * 2, n_gt_in_cell * 3):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_cell * 3, n_gt_in_cell * 4):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

    def set_ubs(self):
        self.pos_ubs[0] = [80, 80]
        self.pos_ubs[1] = [320, 80]

    def get_map(self, flag: int = 0):
        self.set_eve()
        if flag == 1:
            self.set_gts()
        self.set_ubs()

        return self.__dict__


map = {'General4uavMap': General4uavMap, 
       'General2uavMap': General2uavMap}

if __name__ == '__main__':
    np.random.seed(10)
    general4uavMap = General4uavMap()
    generalMap = general4uavMap.get_map()
    plot(generalMap)

    print(generalMap)

    # print(generalMap)
    # gts_in_community = generalMap['gts_in_community']
    # print(gts_in_community)

    # general2uavMap = General2uavMap()
    # generalMap = general2uavMap.get_map()
    # plot(generalMap)
    

