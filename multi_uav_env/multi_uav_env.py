from channel_model import *

from utils import *

from maps import General2uavMap, General4uavMap

class MultiUbsRsmaEvn:
    h_ubs = 100 # 无人机的高度
    p_tx_c_dbm = 35 # 公共信息功率 (dBm)
    p_tx_p_dbm = 10 # 私有信息功率 (dBm)
    p_forward_c_dbm = 10  # GT转发公有信息功率 (dbm)
    p_forward_p_dbm = 10  # GT转发私有信息功率 (dbm)
    p_tx_c = 1e-3 * np.power(10, p_tx_c_dbm / 10) # 公共信息传输功率 (w)
    p_tx_p = 1e-3 * np.power(10, p_tx_p_dbm / 10) # 私有信息传输功率 (w)
    p_forward_c = 1e-3 * np.power(10, p_forward_c_dbm / 10) # 转发公有信息传输功率 (w)
    p_forward_p = 1e-3 * np.power(10, p_forward_p_dbm / 10) # 转发私有信息传输功率 (w)
    n0 = 1e-3 * np.power(10, -200 / 10) # 噪声的PSD
    bw = 180e3 # 频带
    fc = 2.4e9 # 载频
    scene = 'urban' # 场景

    def __init__(self, args) -> None:
        # 数量
        self.n_uav = args.n_uav
        self.n_gt = args.n_gt
        self.n_eve = args.n_eve
        
        self.cov_range = args.cov_range  # 覆盖范围
        self.comm_range = args.comm_range # 通信范围
        self.serv_capacity = args.serv_capacity # 服务范围

        self.range_pos = args.range_pos # 场景范围
        
        self.apply_small_fading = args.apply_small_fading # 是否应用小尺度衰落

        # 地图map、位置信息
        self.map = args.map()
        self.map_info = self.map.get_map()
        self.pos_ubs = self.map_info['pos_ubs']
        self.pos_gts = self.map_info['pos_gts']
        self.pos_eves = self.map_info['pos_eve']
        self.range_pos = self.map_info['range_pos']
        self.gts_in_community = self.map_info['gts_in_community']
        self.n_community = len(self.gts_in_community)

        # 信道
        self.atg_chan_model = AirToGroundChannel(scene=self.scene, fc=self.fc, apply_small_fading=self.apply_small_fading) # ATG信道
        self.gtg_chan_model = GroundToGroundChannel(fc=self.fc) # GTG信道

        # 距离矩阵
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32) # UAV->GT距离
        self.dis_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32) # UAV->Eve距离
        self.dis_U2U = np.zeros((self.n_uav, self.n_uav), dtype=np.float32) # UAV->UAV距离
        self.dis_G2G = np.zeros((self.n_gt, self.n_gt), dtype=np.float32) # GT->GT距离
        self.dis_G2E = np.zeros((self.n_gt, self.n_eve), dtype=np.float32) # GT->Eve距离

        # 关联矩阵
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool) 
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav), dtype=bool)
        self.sche_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool)
        self.sche_U2E = np.zeros((self.n_uav, self.n_eve), dtype=bool)

        self.gt_serv_by = np.zeros((self.n_gt), dtype=np.int32) # gt i被那个无人机服务

        self.t = 0  # 时间步

        # 信道
        self.H_U2G = np.zeros((self.n_uav, self.n_gt))
        self.H_U2G_norm_2 = np.zeros((self.n_uav, self.n_gt))
        self.H_U2E = np.zeros((self.n_uav, self.n_eve))
        self.H_U2E_norm_2 = np.zeros((self.n_uav, self.n_eve))
        self.H_G2G = np.zeros((self.n_gt, self.n_gt))
        self.H_G2G_norm_2 = np.zeros((self.n_gt, self.n_gt))
        self.H_G2E = np.zeros((self.n_gt, self.n_eve))
        self.H_G2E_norm_2 = np.zeros((self.n_gt, self.n_eve))
        self.gt_norm_2 = np.zeros((self.n_gt), dtype=np.float32)

        # # 可能的action
        # self.velocity = args.velocity_bound
        # move_amounts = np.array([self.velocity]).reshape(-1, 1)
        # ang = 2 * np.pi * np.arange(self.n_dirs) / self.n_dirs
        # move_dirs = np.stack([np.cos(ang), np.sin(ang)]).T  # 其分别的正余弦
        # self.avail_moves = np.concatenate((np.zeros((1, 2)), np.kron(move_amounts, move_dirs)))
        # self.avail_jamming_powers = [0]
        # self.jamming_power_bound = args.jamming_power_bound
        # for i in range(self.n_powers - 1):
        #     self.avail_jamming_powers.append(1 * (self.jamming_power_bound / 1) ** (i / (self.n_powers - 2)))
        # self.avail_jamming_powers = np.array(self.avail_jamming_powers, dtype=np.float32)
        # self.avail_jamming_powers = 1e-3 * np.power(10, self.avail_jamming_powers / 10)  # to w
        # self.n_moves = self.avail_moves.shape[0]    

        # 数据速率
        self.comm_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.priv_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.comm_rate_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)
        self.priv_rate_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)   
        self.comm_rate_gt = np.zeros((self.n_gt), dtype=np.float32)
        self.priv_rate_gt = np.zeros((self.n_gt), dtype=np.float32)
        self.comm_rate_eve = np.zeros((self.n_eve), dtype=np.float32)
        self.priv_rate_eve = np.zeros((self.n_eve), dtype=np.float32)


    def update_dist_conn(self) -> None:
        # UAV与GT
        gt_becov = [[] for i in range(self.n_gt)]  # UAV k 覆盖的GT
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool)
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                self.dis_U2G[k][i] = np.linalg.norm(self.pos_ubs[k] - self.pos_gts[i])
                self.cov_U2G[k][i] = 1 if self.dis_U2G[k][i] <= self.cov_range else 0  # 覆盖关系 
                gt_becov[i].append(k) if self.cov_U2G[k][i] == 1 else None

        # UAV与Eve
        self.sche_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)
        self.dis_U2E = np.zeros((self.n_uav, self.n_eve), dtype=bool)
        for k in range(self.n_uav):
            for e in range(self.n_eve):
                self.dis_U2E[k][e] = np.linalg.norm(self.pos_ubs[k] - self.pos_eves[e])
                self.sche_U2E[k][e] = 1 if self.dis_U2E[k][e] <= self.cov_range else 0 # 窃听关系

        # UAV与UAV
        self.dis_U2U = np.zeros((self.n_uav, self.n_uav), dtype=np.float32)
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav), dtype=bool)
        for k in range(self.n_uav):
            for l in range(self.n_uav):
                self.dis_U2U[k][l] = np.linalg.norm(self.pos_ubs[k] - self.pos_ubs[l])
                self.cov_U2U[k][l] = 1 if self.dis_U2U[k][l] <= self.comm_range else 0 # 通信关联

        # GT与GT
        if self.t == 0:
            self.dis_G2G = np.zeros((self.n_gt, self.n_gt), dtype=np.float32)
            for i in range(self.n_gt):
                for j in range(self.n_gt):
                    self.dis_G2G[i][j] = np.linalg.norm(self.pos_gts[i] - self.pos_gts[j])

        # GT与Eve
        if self.t == 0:
            self.dis_G2E = np.zeros((self.n_gt, self.n_eve), dtype=np.float32)
            for i in range(self.n_gt):
                for e in range(self.n_eve):
                    self.dis_G2E[i][e] = np.linalg.norm(self.pos_gts[i] - self.pos_eves[e])

        # 生成信道
        self.generate_channel()

        self.sche_U2G = np.zeros((self.n_uav, self.n_gt))
        self.gt_serv_by = np.zeros((self.n_gt), dtype=np.int32)
        for i in range(self.n_gt):
            gt_becov[i] = sorted(gt_becov[i], key=lambda k: self.H_U2G_norm_2[k][i], reverse=True)
            for k in gt_becov[i]:
                if sum(self.sche_U2G[k]) < self.serv_capacity:
                    self.sche_U2G[k][i] = 1 # UAV 服务 GT i
                    self.gt_serv_by[i] = k
                    break


    def generate_channel(self) -> None:
        # 生成UAV-GT的信道
        self.H_U2G = np.zeros((self.n_uav, self.n_gt))
        self.H_U2G_norm_2 = np.zeros((self.n_uav, self.n_gt))
        self.gt_norm_2 = np.zeros((self.n_gt), dtype=np.float32)
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                if self.cov_U2G[k][i] == 1:
                    g = self.atg_chan_model.estimate_chan_gain(d_level=self.dis_U2G[k][i], h_ubs=self.h_ubs)
                    self.H_U2G[k][i] = g
                    self.H_U2G_norm_2[k][i] = np.linalg.norm(g) ** 2
                    self.gt_norm_2[i] = self.H_U2G_norm_2[k][i]
        
        # 生成UAV-Eve的信道 (TODO:不完美状态信息)
        self.H_U2E = np.zeros((self.n_uav, self.n_eve))
        self.H_U2E_norm_2 = np.zeros((self.n_uav, self.n_eve))
        for k in range(self.n_uav):
            for e in range(self.n_eve):
                if self.sche_U2E[k][e] == 1:
                    g = self.atg_chan_model.estimate_chan_gain(d_level=self.dis_U2G[k][e], h_ubs=self.h_ubs)
                    self.H_U2E[k][e] = g
                    self.H_U2E_norm_2[k][e] = np.linalg.norm(g) ** 2
        
        # 生成GT-GT的信道
        self.H_G2G = np.zeros((self.n_gt, self.n_gt))
        self.H_G2G_norm_2 = np.zeros((self.n_gt, self.n_gt))
        for i in range(self.n_gt):
            for j in range(self.n_gt):
                g = self.gtg_chan_model.estimate_chan_gain(d=self.dis_G2G[i][j])
                self.H_G2G[i][j] = g
                self.H_G2G_norm_2[i][j] = np.linalg.norm(g) ** 2

        # 生成GT-Eve信道 (TODO:不完美状态信息)
        self.H_G2E = np.zeros((self.n_gt, self.n_eve))
        self.H_G2E_norm_2 = np.zeros((self.n_gt, self.n_eve))
        for i in range(self.n_gt):
            for e in range(self.n_eve):
                g = self.gtg_chan_model.estimate_chan_gain(d=self.dis_G2E[i][e])
                self.H_G2E[i][e] = g
                self.H_G2E_norm_2[i][e] = np.linalg.norm(g) ** 2
    
    def transmit_data(self, jamming_power):
        """stage 1: 直接传输阶段"""
        # stage 1: 计算 UAV->GTs 公共信息速率
        self.comm_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.comm_rate_gt = np.zeros((self.n_gt), dtype=np.float32)
        for i in range(self.n_gt):
            serv_flag = False
            n = self.n0 * self.bw
            for k in range(self.n_uav):
                if self.sche_U2G[k][i] == 1:
                    s = self.H_U2G_norm_2[k][i] * self.p_tx_c * sum(self.sche_U2G[k]) # 公有信号
                    n = n + self.H_U2G_norm_2[k][i] * self.p_tx_p * sum(self.sche_U2G[k]) # 本无人机的所有私有作为干扰
                    serv_flag = True
                else:
                    n = n + self.cov_U2G[k][i] * self.H_U2G_norm_2[k][i] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2G[k]) # 系统间干扰 (其他无人机)
            if serv_flag:
                # 计算香农容量
                self.comm_rate_U2G[k][i] = self.shannon_capacity(s, n)
                self.comm_rate_gt[i] = self.comm_rate_U2G[k][i]

        # stage 1: 计算 UAV->GTs 私有信息速率
        self.priv_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.priv_rate_gt = np.zeros((self.n_gt), dtype=np.float32)
        for i in range(self.n_gt):
            serv_flag = False
            n = self.n0 * self.bw # 本地噪声
            for k in range(self.n_uav):
                if self.sche_U2G[k][i] == 1: # 服务的无人机
                    s = self.H_U2G_norm_2[k][i] * self.p_tx_p 
                    n = n + self.H_U2G_norm_2[k][i] * self.p_tx_p * (sum(self.sche_U2G[k]) - 1)
                    serv_flag = True 
                else: # 其他无人机
                    n = n + self.cov_U2G[k][i] * self.H_U2G_norm_2[k][i] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2G[k]) # 系统间干扰 (其他无人机)
            if serv_flag:
                # 计算香农容量
                self.priv_rate_U2G[k][i] = self.shannon_capacity(s, n)
                self.priv_rate_gt[i] = self.priv_rate_U2G[k][i]

        # stage 1: 计算 UAV->Eve 公有信息速率
        self.comm_rate_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)
        self.comm_rate_eve = np.zeros((self.n_eve), dtype=np.float32)
        for e in range(self.n_eve):
            n = self.n0 * self.bw # 本地噪声
            for k in range(self.n_uav):
                if self.sche_U2E[k][e] == 1:
                    s = self.H_U2E_norm_2[k][e] * self.p_tx_c * sum(self.sche_U2E[k])
                    n = n + self.H_U2E_norm_2[k][e] * self.p_tx_p * sum(self.sche_U2E[k]) # 本无人机的所有私有作为干扰
                    for l in range(self.n_uav): # 其他无人机的干扰
                        if self.sche_U2E[l][e] == 1 and l != k:
                            n = n + self.H_U2E_norm_2[l][e] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2E[l])
                    self.comm_rate_U2E[k][e] = self.shannon_capacity(s, n)
                    self.comm_rate_eve[e] = self.comm_rate_U2E[k][e]
                     
        # stage 1: 计算 UAV->Eve 私有信息速率
        self.priv_rate_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)
        for e in range(self.n_eve):
            n = self.n0 * self.bw # 本地噪声
            for k in range(self.n_uav):
                if self.sche_U2E[k][e] == 1:
                    s = self.H_U2E_norm_2[k][e] * self.p_tx_p
                    n = n + self.H_G2E_norm_2[k][e] * (self.p_tx_c * sum(self.sche_U2E[k]) +  # 本无人机的公有
                                                       self.p_tx_p * (sum(self.sche_U2E[k]) - 1)) # 本无人机的其他私有作为干扰
                    for l in range(self.n_uav): # 其他无人机的干扰  
                        if self.sche_U2E[l][e] == 1 and l != k:
                            n = n + self.H_U2E_norm_2[l][e] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2E[l])
                    self.priv_rate_U2E[k][e] = self.shannon_capacity(s, n)
                    self.priv_rate_eve[e] = self.priv_rate_U2E[k][e]

        """stage 2: 转发传输阶段
            其中每个社区信道质量最好的GT,采用解码转发的方式, 转发信息给其他GT
        """
        # stage 2: 计算GT->GT的速率 
        
        # 首先在每一个社区需要先找一个信道质量最好的
        best_gt_in_community = np.zeros(self.n_community, dtype=np.int32)
        for m in range(self.n_community):
            if len(self.gts_in_community[m]) != 0:
                best_gt_in_community[m] = max(self.gts_in_community[m], key=lambda i: self.gt_norm_2[i])

        # i直接转发所有信息
        for m in range(self.n_community):
            i = best_gt_in_community[m]
            n = self.bw * self.n0
            k = self.gt_serv_by[i] # 最好的i被k服务
            e = m
            num_gt_in_community = len(self.gts_in_community[m])
            # j的公有信息速率为
            for j in self.gts_in_community[m]:
                if j != i:
                    s = self.H_G2G_norm_2[i][j] * self.p_forward_c * (num_gt_in_community - 1)
                    n = n + (self.H_G2G_norm_2[i][j] * self.p_forward_p * num_gt_in_community + 
                             self.cov_U2G[k][j] * self.H_U2G_norm_2[k][j] * jamming_power[k])
                    self.comm_rate_gt[i] += self.shannon_capacity(s, n)
            # eve的公有信息速率为
            s_eve = self.H_G2E_norm_2[i][e] * self.p_forward_c * (num_gt_in_community - 1)
            n_eve = self.bw * self.n0 + (self.H_G2E_norm_2[i][e] * self.p_forward_p * num_gt_in_community + 
                                         self.sche_U2E[k][e] * self.H_U2E_norm_2[k][e] * jamming_power[k])
            self.comm_rate_eve[i] += self.shannon_capacity(s_eve, n_eve)
            
        for m in range(self.n_community):
            i = best_gt_in_community[m]
            n = self.bw * self.n0
            k = self.gt_serv_by[i] # 最好的i被k服务
            num_gt_in_community = len(self.gts_in_community[m])
            # j的私有信息速率为
            for j in self.gts_in_community[m]:
                if j != i:
                    s = self.H_G2G_norm_2[i][j] * self.p_forward_p
                    n = n + (self.H_G2G_norm_2[i][j] * self.p_forward_p * (num_gt_in_community - 1) + 
                             self.cov_U2G[k][j] * self.H_U2G_norm_2[k][j] * jamming_power[k])
                    self.priv_rate_gt[i] += self.shannon_capacity(s, n)
            # eve的私有信息速率为
            s_eve = self.H_G2E_norm_2[i][e] * self.p_forward_p
            n_eve = self.bw * self.n0 + (self.H_G2E_norm_2[i][e] * self.p_forward_p * (num_gt_in_community - 1) +
                     self.H_G2E_norm_2[i][e] * self.p_forward_c * num_gt_in_community + 
                     self.sche_U2E[k][e] * self.H_U2E_norm_2[k][e] * jamming_power[k])
            self.comm_rate_eve[i] += self.shannon_capacity(s_eve, n_eve)

    def shannon_capacity(self, s, n):
        # 计算香农容量 (Mbps)
        return self.bw * np.log2(1 + s / n) * 1e-6

    def reset(self):
        self.t = 0
        self.map_info = self.map.get_map()
        self.pos_ubs = self.map_info['pos_ubs'] # 初始化位置
        self.pos_gts = self.map_info['pos_gts'] # 每个episode随机生成

        self.update_dist_conn() # 初始距离、关联关系、生成信道

        pass

    def step(self, actions):
        self.t = self.t + 1
        pass

    def get_env_info(self):
        pass

    def get_uav_trajectory(self):
        pass

    def get_jamming_power(self):
        pass

    def get_fair_index(self):
        pass

    def get_ssr(self):
        pass

    def get_throughput(self):
        pass

    def get_throughput_gt(self):
        pass

    def sercurity_model(self):
        pass

    def get_obs(self) -> list:
        pass

    def get_obs_agent(self, agent_id: int) -> dict:
        pass

    def get_obs_size(self) -> dict:
        pass

    @property
    def obs_own_feats_size(self) -> int:
        pass

    @property
    def obs_ubs_feats_size(self) -> tuple:
        pass

    @property
    def obs_gt_feats_size(self) -> tuple:
        pass

    def get_state(self) -> np.ndarray:
        pass

    def get_state_size(self) -> int:
        pass

    def state_ubs_feats_size(self) -> tuple:
        pass

    def state_gt_feats_size(self) -> tuple:
        pass

    def get_reward(self, reward_scale_rate) -> float:
        pass

    def get_terminate(self) -> bool:
        pass

if __name__ == '__main__':
    set_rand_seed(seed=10)
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_uav', type=int, default=4, help='the number of UAV')
    parser.add_argument('--n_gt', type=int, default=20, help='the number of Gt') 
    parser.add_argument('--n_eve', type=int, default=2, help='the number of Eve')
    parser.add_argument('--apply_small_fading', type=bool, default=False)
    parser.add_argument('--cov_range', type=float, default=50, help='coverage range (m)')
    parser.add_argument('--comm_range', type=float, default=np.inf, help='communication range (m)')
    parser.add_argument('--serv_capacity', type=int, default=5, help='service capacity')
    parser.add_argument('--range_pos', type=float, default=400, help='scene range (m)')
    parser.add_argument('--map', default=General4uavMap, help='map type')
    parser.add_argument('--velocity_bound', type=float, default=20, help='velocity of UAV (m/s)')
    parser.add_argument('--jamming_power_bound', type=float, default=15, help='jamming power of UAV (dBm)')
    
    args = parser.parse_args()

    # print(args)

    # x = args.map(range_pos=400, n_eve=16, n_gts=20, n_ubs=4, n_community=16)
    # print(x.get_map())

    multi_uav_env = MultiUbsRsmaEvn(args=args)

    multi_uav_env.update_dist_conn()
    multi_uav_env.transmit_data(jamming_power=1)

    