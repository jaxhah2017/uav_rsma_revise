import numpy as np

import matplotlib.pyplot as plt


def plot(map):
    uav_pos = map['pos_ubs']
    eve_pos = map['pos_eve']
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
    # plot axis0
    for x in range(0, range_pos, int(area)):
        if x == 0:
            continue
        ax.plot([0, range_pos], [x, x], linestyle='--', color='b')

    # plot axis1
    for y in range(0, range_pos, int(area)):
        if y == 0:
            continue
        ax.plot([y, y], [0, range_pos], linestyle='--', color='b')

    ax.legend(handles=[ubs, eve, gts],
              labels=['uav_pos', 'eve_pos', 'gts_pos'],
              loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.75)
    # plt.show()
    plt.savefig('map.png')

class General4uavMap:
    def __init__(self, range_pos=400, n_eve=16, n_gts=20, n_ubs=4, n_community=16):
        self.n_eve = n_eve
        self.n_gts = n_gts
        self.n_ubs = n_ubs
        self.pos_eve = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)
        self.range_pos = range_pos
        self.fen = 4
        self.area = self.range_pos / self.fen
        self.n_community = n_community
        self.gts_in_community = [[] for _ in range(self.n_community)]

    def set_eve(self):
        # for i in range(self.fen):
        #     range_eve_ly = self.area * i
        #     range_eve_ry = self.area * (i + 1)
        #     for j in range(self.fen):
        #         range_eve_lx = self.area * j
        #         range_eve_rx = self.area * (j + 1)
        #         x = np.random.uniform(range_eve_lx, range_eve_rx)
        #         y = np.random.uniform(range_eve_ly, range_eve_ry)
        #         self.pos_eve[i * self.fen + j] = x, y
        self.pos_eve[0] = [-500, -500]
        self.pos_eve[1] = [-500, -500]
        self.pos_eve[2] = [-500, -500]
        self.pos_eve[3] = [-500, -500]
        self.pos_eve[4] = [-500, -500]
        self.pos_eve[5] = [110, 110]
        self.pos_eve[6] = [280, 120]
        self.pos_eve[7] = [-500, -500]
        self.pos_eve[8] = [-500, -500]
        self.pos_eve[9] = [190, 290]
        self.pos_eve[10] = [220, 220]
        self.pos_eve[11] = [-500, -500]
        self.pos_eve[12] = [-500, -500]
        self.pos_eve[13] = [-500, -500]
        self.pos_eve[14] = [-500, -500]
        self.pos_eve[15] = [-500, -500]

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]

        n_gt_in_community = int(self.n_gts / 4)
        for i in range(n_gt_in_community):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_community, n_gt_in_community * 2):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_community * 2, n_gt_in_community * 3):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_community * 3, n_gt_in_community * 4):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

        for i in range(self.n_gts):
            x, y = self.pos_gts[i]
            num_community = (y // self.area) * self.fen + (x // self.area)
            self.gts_in_community[int(num_community)].append(i)

    def set_ubs(self):
        # self.pos_ubs[0] = [80, 80]
        # self.pos_ubs[1] = [320, 80]
        # self.pos_ubs[2] = [80, 320]
        # self.pos_ubs[3] = [320, 320]

        # DEBUG
        self.pos_ubs[0] = [150, 150]
        self.pos_ubs[1] = [250, 150]
        self.pos_ubs[2] = [150, 250]
        self.pos_ubs[3] = [260, 260]

    def get_map(self):
        self.set_eve()
        self.set_gts()
        self.set_ubs()

        return dict(pos_gts=self.pos_gts,
                    pos_eve=self.pos_eve,
                    pos_ubs=self.pos_ubs,
                    area=self.area,
                    range_pos=self.range_pos,
                    gts_in_community=self.gts_in_community)

class General2uavMap:
    def __init__(self, range_pos=400, n_eve=16, n_gts=20, n_ubs=2, n_community=16):
        self.n_eve = n_eve
        self.n_gts = n_gts
        self.n_ubs = n_ubs
        self.pos_eve = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)
        self.range_pos = range_pos
        self.fen = 4
        self.area = self.range_pos / self.fen
        self.n_community = n_community
        self.gts_in_community = [[] for _ in range(self.n_community)]

    def set_eve(self):
        self.pos_eve[0] = [-500, -500]
        self.pos_eve[1] = [-500, -500]
        self.pos_eve[2] = [-500, -500]
        self.pos_eve[3] = [-500, -500]
        self.pos_eve[4] = [-500, -500]
        self.pos_eve[5] = [110, 110]
        self.pos_eve[6] = [280, 120]
        self.pos_eve[7] = [-500, -500]
        self.pos_eve[8] = [-500, -500]
        self.pos_eve[9] = [190, 290]
        self.pos_eve[10] = [220, 220]
        self.pos_eve[11] = [-500, -500]
        self.pos_eve[12] = [-500, -500]
        self.pos_eve[13] = [-500, -500]
        self.pos_eve[14] = [-500, -500]
        self.pos_eve[15] = [-500, -500]
        

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]

        n_gt_in_community = int(self.n_gts / 4)
        for i in range(n_gt_in_community):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_community, n_gt_in_community * 2):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(220, 280)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_community * 2, n_gt_in_community * 3):
            x = np.random.uniform(120, 180)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

        for i in range(n_gt_in_community * 3, n_gt_in_community * 4):
            x = np.random.uniform(220, 280)
            y = np.random.uniform(120, 180)
            self.pos_gts[i] = [x, y]

        for i in range(self.n_gts):
            x, y = self.pos_gts[i]
            num_community = (y // self.area) * self.fen + (x // self.area)
            self.gts_in_community[int(num_community)].append(i)

    def set_ubs(self):
        self.pos_ubs[0] = [80, 80]
        self.pos_ubs[1] = [320, 80]

    def get_map(self):
        self.set_eve()
        self.set_gts()
        self.set_ubs()

        return dict(pos_gts=self.pos_gts,
                    pos_eve=self.pos_eve,
                    pos_ubs=self.pos_ubs,
                    area=self.area,
                    range_pos=self.range_pos,
                    gts_in_community=self.gts_in_community)

if __name__ == '__main__':
    np.random.seed(10)
    general4uavMap = General4uavMap()
    generalMap = general4uavMap.get_map()
    plot(generalMap)
    # print(generalMap)
    # gts_in_community = generalMap['gts_in_community']
    # print(gts_in_community)

    # general2uavMap = General2uavMap()
    # generalMap = general2uavMap.get_map()
    # plot(generalMap)
    

