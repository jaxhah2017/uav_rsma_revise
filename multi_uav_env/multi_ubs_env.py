from multi_uav_env.channel_model import *

from multi_uav_env.utils import *

from multi_uav_env.maps import General2uavMap, General4uavMap

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
    scene = 'dense-urban' # 场景
    safe_dist = 5 # 无人机之间的安全距离 (m)

    def __init__(self, args) -> None:
        self.avoid_collision = args.avoid_collision
        self.penlty = args.penlty

        self.cov_range = args.cov_range  # 覆盖范围
        self.comm_range = args.comm_range # 通信范围
        self.serv_capacity = args.serv_capacity # 服务范围
        
        self.apply_small_fading = args.apply_small_fading # 是否应用小尺度衰落

        self.episode_length = args.episode_length # 每个episode的长度

        self.theta_opt = args.theta_opt # 是否优化theta

        # 地图map、位置信息
        self.map = args.map()
        self.map_params = self.map.get_map()
        for k, v in self.map_params.items():
            setattr(self, k, v)

        self.n_agents = self.n_uav # 智能体个数=无人机的个数

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
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool)  # GT被UAV覆盖
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav), dtype=bool) # UAV与UAV能够联系
        self.sche_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool) # GT与UAV通信
        self.sche_U2E = np.zeros((self.n_uav, self.n_eve), dtype=bool) # Eve窃听UAV

        self.gt_serv_by = np.zeros((self.n_gt), dtype=np.int32) # gt i被那个无人机服务
        self.uav_serv_gt = None # 无人机k服务的GT

        self.t = 1  # 时间步

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

        # 可能的action
        self.n_dirs = args.n_dirs
        self.velocity = args.velocity_bound
        move_amounts = np.array([self.velocity]).reshape(-1, 1)
        ang = 2 * np.pi * np.arange(self.n_dirs) / self.n_dirs
        move_dirs = np.stack([np.cos(ang), np.sin(ang)]).T  # 其分别的正余弦
        self.avail_moves = np.concatenate((np.zeros((1, 2)), np.kron(move_amounts, move_dirs)))
        
        self.n_powers = args.n_powers
        self.avail_jamming_powers = [0]
        self.jamming_power_bound = args.jamming_power_bound
        for i in range(self.n_powers - 1):
            self.avail_jamming_powers.append(1 * (self.jamming_power_bound / 1) ** (i / (self.n_powers - 2)))
        self.avail_jamming_powers = np.array(self.avail_jamming_powers, dtype=np.float32)
        self.avail_jamming_powers = 1e-3 * np.power(10, self.avail_jamming_powers / 10)  # to w
        self.n_moves = self.avail_moves.shape[0]    

        self.avail_theta = np.arange(0.25, 1.25, 0.25) # test
        self.n_thetas = len(self.avail_theta) # test


        # 数据速率
        self.comm_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32) # 第一阶段直传的公共信息速率
        self.priv_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.comm_rate_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)
        self.priv_rate_E7G = np.zeros((self.n_eve, self.n_gt), dtype=np.float32) # Eve窃听GT的私有信息
        
        self.secrecy_rate_c_k_t = np.zeros((self.n_uav), dtype=np.float32) # 第k个无人机cell的公共信息保密率
        self.secrecy_rate_p_i_t = np.zeros((self.n_gt), dtype=np.float32) # 每个GT的私有信息保密率

        # 需要记录的数据
        self.uav_traj = []
        self.jamming_power_list = []
        self.ssr_list = []
        self.throughput_list = []
        self.fair_idx_list = []
        self.theta_list = []

        self.episo_return = np.zeros(self.n_uav, dtype=np.float32)
        self.mean_returns = 0
        
        # 最大数据速率用于归一化
        g_atg_max = self.atg_chan_model.estimate_chan_gain(d_level=0.00000000000001, h_ubs=self.h_ubs)
        self.snr_c_atg_max = self.p_tx_c * self.n_gt * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)  # UAV->GTs common info
        self.snr_p_atg_max = self.p_tx_p * self.n_gt * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)  # UAV->GTs private info
        self.snr_c_gtg_max = self.p_forward_c * (self.n_gt - 1) * np.power(abs(1), 2) / (self.n0 * self.bw)  # GT->GTs common info
        self.snr_p_gtg_max = self.p_forward_p * (self.n_gt - 1) * np.power(abs(1), 2) / (self.n0 * self.bw)  # GT->GTs common info
        self.achievable_rate_c_ubs_max = self.bw * np.log(1 + self.snr_c_atg_max) * 1e-6
        self.achievable_rate_p_ubs_max = self.bw * np.log(1 + self.snr_p_atg_max) * 1e-6
        self.achievable_rate_c_gts_max = self.bw * (np.log(1 + self.snr_c_atg_max) + np.log(1 + self.snr_c_gtg_max)) * 1e-6
        self.achievable_rate_p_gts_max = self.bw * (np.log(1 + self.snr_p_atg_max) * 1e-6 + np.log(1 + self.snr_p_gtg_max)) * 1e-6
        self.achievable_rate_gts_max = self.achievable_rate_c_gts_max + self.achievable_rate_p_gts_max
        self.achievable_rate_ubs_max = self.achievable_rate_c_ubs_max + self.achievable_rate_p_ubs_max

        # 全局指标
        self.fair_idx_t = None # 实时公平因子
        self.avg_epi_fair_index = None # 每个episode平均公平因子
        
        self.throughput_t = None # 实时吞吐量
        self.throughput = None # 总吞吐量
        self.avg_epi_throughput = None # 每个episode平均吞吐量

        self.sec_rate_t = None # 实时安全容量
        self.avg_epi_sec_rate = None # 每个episode平均吞吐量
        self.sec_rate = 0 # 总实时安全容量
        self.sec_rate_gt = np.zeros(self.n_gt, np.float32)  # 计算GT的累积SSR

        self.rate_gt_t = np.zeros((self.n_gt), dtype=np.float32) # 每个GT的实时吞吐量
        self.avg_epi_rate_gt = np.zeros((self.n_gt), dtype=np.float32) # 每个GT在一个episode的平均吞吐量

        self.rate_ubs_t = np.zeros((self.n_uav), dtype=np.float32) # 每个无人机的数据速率

        self.global_util = 0 # 全局效用

    def update_dist_conn(self) -> None:
        # UAV与GT
        gt_becov = [[] for _ in range(self.n_gt)]  # UAV k 覆盖的GT
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool)
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                self.dis_U2G[k][i] = np.linalg.norm(self.pos_ubs[k] - self.pos_gts[i])
                self.cov_U2G[k][i] = 1 if self.dis_U2G[k][i] <= self.cov_range else 0  # 覆盖关系 
                if self.cov_U2G[k][i] == 1:
                    gt_becov[i].append(k)

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

        # GT与GT GT静止，所以距离只在最初定下来就可以
        if self.t == 1:
            self.dis_G2G = np.zeros((self.n_gt, self.n_gt), dtype=np.float32)
            for i in range(self.n_gt):
                for j in range(self.n_gt):
                    self.dis_G2G[i][j] = np.linalg.norm(self.pos_gts[i] - self.pos_gts[j])

        # GT与Eve GT与Eve同样静止，所以距离只在最初定下来就可以
        if self.t == 1:
            self.dis_G2E = np.zeros((self.n_gt, self.n_eve), dtype=np.float32)
            for i in range(self.n_gt):
                for e in range(self.n_eve):
                    self.dis_G2E[i][e] = np.linalg.norm(self.pos_gts[i] - self.pos_eves[e])

        # 生成信道
        self.generate_channel()

        self.sche_U2G = np.zeros((self.n_uav, self.n_gt), dtype=bool)
        self.gt_serv_by = np.zeros((self.n_gt), dtype=np.int32)
        self.uav_serv_gt = [[] for _ in range(self.n_uav)]  # UAV k 服务的GT
        for i in range(self.n_gt):
            gt_becov[i] = sorted(gt_becov[i], key=lambda k: self.H_U2G_norm_2[k][i], reverse=True)
            for k in gt_becov[i]:
                if sum(self.sche_U2G[k]) < self.serv_capacity:
                    self.sche_U2G[k][i] = 1 # UAV 服务 GT i
                    self.gt_serv_by[i] = k
                    self.uav_serv_gt[k].append(i)
                    break

        # 如果不服务那就取消信道，因为要根据有无信道来判断是否被服务
        # for k in range(self.n_uav):
        #     for i in range(self.n_gt):
        #         if self.sche_U2G[k][i] == 0:
        #             self.H_U2G[k][i] = 0
        #             self.H_U2G_norm_2[k][i] = 0
        #             self.gt_norm_2[i] = 0


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
                    g = self.atg_chan_model.estimate_chan_gain(d_level=self.dis_U2E[k][e], h_ubs=self.h_ubs)
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
    
    def transmit_data(self, jamming_power, thetas):
        """stage 1: 直接传输阶段"""
        # stage 1: 计算 UAV->GTs 公共信息速率
        self.comm_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        for i in range(self.n_gt):
            serv_uav = -1
            n = self.n0 * self.bw
            for k in range(self.n_uav):
                if self.sche_U2G[k][i] == 1:
                    s = self.H_U2G_norm_2[k][i] * self.p_tx_c * sum(self.sche_U2G[k]) # 公有信号
                    n = n + self.H_U2G_norm_2[k][i] * self.p_tx_p * sum(self.sche_U2G[k]) # 本无人机的所有私有作为干扰
                    serv_uav = k
                else: # 其他无人机的干扰
                    n = n + self.cov_U2G[k][i] * self.H_U2G_norm_2[k][i] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2G[k]) # 系统间干扰 (其他无人机)
            if serv_uav != -1:
                # 计算香农容量
                self.comm_rate_U2G[serv_uav][i] = thetas[serv_uav] * self.shannon_capacity(s, n)
                

        # stage 1: 计算 UAV->GTs 私有信息速率
        self.priv_rate_U2G = np.zeros((self.n_uav, self.n_gt), dtype=np.float32)
        for i in range(self.n_gt):
            serv_uav = -1
            n = self.n0 * self.bw # 本地噪声
            for k in range(self.n_uav):
                if self.sche_U2G[k][i] == 1: # 服务的无人机
                    s = self.H_U2G_norm_2[k][i] * self.p_tx_p 
                    n = n + self.H_U2G_norm_2[k][i] * self.p_tx_p * (sum(self.sche_U2G[k]) - 1) # 本无人机的其他私有
                    serv_uav = k 
                else: # 其他无人机
                    n = n + self.cov_U2G[k][i] * self.H_U2G_norm_2[k][i] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2G[k]) # 系统间干扰 (其他无人机)
            if serv_uav != -1:
                # 计算香农容量
                self.priv_rate_U2G[serv_uav][i] = thetas[serv_uav] * self.shannon_capacity(s, n)

        # stage 1: 计算 UAV->Eve 公有信息速率
        self.comm_rate_U2E = np.zeros((self.n_uav, self.n_eve), dtype=np.float32)
        for e in range(self.n_eve):
            n = self.n0 * self.bw # 本地噪声
            for k in range(self.n_uav):
                if self.sche_U2E[k][e] == 1:
                    s = self.H_U2E_norm_2[k][e] * self.p_tx_c * sum(self.sche_U2E[k])
                    n = n + self.H_U2E_norm_2[k][e] * self.p_tx_p * sum(self.sche_U2E[k]) # 本无人机的所有私有作为干扰
                    for l in range(self.n_uav): # 其他无人机的干扰
                        if self.sche_U2E[l][e] == 1 and l != k:
                            n = n + self.H_U2E_norm_2[l][e] * (self.p_tx_c + self.p_tx_p) * sum(self.sche_U2E[l])
                    self.comm_rate_U2E[k][e] = thetas[k] * self.shannon_capacity(s, n)
                     
        # stage 1: 计算 UAV->Eve 私有信息速率
        self.priv_rate_E7G = np.zeros((self.n_eve, self.n_gt), dtype=np.float32) 
        for k in range(self.n_uav):
            n = self.n0 * self.bw
            len_uav_serv = len(self.uav_serv_gt[k])
            for e in range(self.n_eve):
                if self.sche_U2E[k][e] == 1: # Eve e正在窃听无人机UAV k
                    for l in range(self.n_uav): # 其他无人机的干扰
                        if self.sche_U2E[l][e] == 1 and l != k:
                            n = n + self.H_U2E_norm_2[l][e] * (self.p_tx_c + self.p_tx_p) * len(self.uav_serv_gt[l])
 
                    for i in self.uav_serv_gt[k]: # 窃听无人机服务的GT
                        s = self.H_U2E_norm_2[k][e] * self.p_tx_p
                        n = n + self.H_U2E_norm_2[k][e] * (self.p_tx_c * len_uav_serv +  # 本无人机的公有
                                                       self.p_tx_p * (len_uav_serv - 1)) # 本无人机的其他私有作为干扰
                        self.priv_rate_E7G[e][i] = thetas[k] * self.shannon_capacity(s, n)

        """stage 2: 转发传输阶段
            其中每个无人机cell中信道质量最好的GT,采用解码转发的方式, 转发信息给其他GT, 最后就是一个无人机一个公共信息计算，然后私有信息分别计算。
        """
        # stage 2: 计算GT->GT的速率 
        # 首先在每一个无人机cell中需要先找一个信道质量最好的对无人机cell外的
        best_gt_in_cell = np.zeros(self.n_uav, dtype=np.int32)
        for k in range(self.n_uav):
            if len(self.uav_serv_gt[k]) != 0:
                best_gt_in_cell[k] = max(self.uav_serv_gt[k], key=lambda i: self.gt_norm_2[i])

        for k in range(self.n_uav): # 每个无人机cell中信道最好的用户转发，计算公有信息和私有信息
            i_forward = best_gt_in_cell[k]
            n = self.n0 * self.bw
            num_gt_in_cell_k = len(self.uav_serv_gt[k])
            # GT->GT
            for i in self.uav_serv_gt[k]:
                if i != i_forward:
                    s_c = self.H_G2G_norm_2[i_forward][i] * self.p_forward_c * num_gt_in_cell_k
                    n_c = n + self.H_G2G_norm_2[i_forward][i] * self.p_forward_p * num_gt_in_cell_k
                    for l in range(self.n_uav):
                        n_c = n_c + self.cov_U2G[l][i] * self.H_U2G_norm_2[l][i] * jamming_power[l]
                    self.comm_rate_U2G[k][i] += (1 - thetas[k]) * self.shannon_capacity(s_c, n_c) # k cell中的GT i的公有信息速率

                    s_p = self.H_G2G_norm_2[i_forward][i] * self.p_forward_p
                    n_p = n + self.H_G2G_norm_2[i_forward][i] * self.p_forward_p * (num_gt_in_cell_k - 1)
                    for l in range(self.n_uav):
                        n_p = n_p + self.cov_U2G[l][i] * self.H_U2G_norm_2[l][i] * jamming_power[l]
                    self.priv_rate_U2G[k][i] += (1 - thetas[k]) * self.shannon_capacity(s_p, n_p) # k cell中的GT i的私有信息速率
            for e in range(self.n_eve):
                if self.sche_U2E[k][e] == 1: # e在k的Cell中
                    s_eve_c = self.H_G2E_norm_2[i_forward][e] * self.p_forward_c * num_gt_in_cell_k
                    n_eve_c = n + self.H_G2E_norm_2[i_forward][e] * self.p_forward_p * num_gt_in_cell_k
                    for l in range(self.n_uav):
                        n_eve_c = n_eve_c + self.sche_U2E[l][e] * self.H_U2E_norm_2[l][e] * jamming_power[l]
                    self.comm_rate_U2E[k][e] += (1 - thetas[k]) * self.shannon_capacity(s_eve_c, n_eve_c) # k cell中的Eve e的公有信息速率

                    s_eve_p = self.H_G2E_norm_2[i_forward][e] * self.p_forward_p
                    n_eve_p = n + (self.H_G2E_norm_2[i_forward][e] * self.p_forward_p * (num_gt_in_cell_k - 1) +  # 私有
                                   self.H_G2E_norm_2[i_forward][e] * self.p_forward_c * num_gt_in_cell_k)        # 公有
                    for l in range(self.n_uav):
                        n_eve_p = n_eve_p + self.sche_U2E[l][e] * self.H_U2E_norm_2[l][e] * jamming_power[l] # 干扰

                    self.priv_rate_E7G[e][i] += (1 - thetas[k]) * self.shannon_capacity(s_eve_p, n_eve_p)
        
    def cal_glo_metric(self):
        """
        计算所需要的所有全局指标：
            实时公平因子: fair_index_t
            每个episode平均公平因子: avg_epi_fair_index
            
            GT实时数据吞吐: rate_gt_t
            GT每个episode平均数据吞吐: avg_epi_rate_gt

            实时吞吐量: throughput_t
            总吞吐量: throughput
            每个episode平均吞吐量: avg_epi_throughput

            实时安全容量: sec_rate_t
            每个episode平均吞吐量: avg_epi_sec_rate
            每个episode的总安全容量: sec_rate
            每个episode中GT的总安全容量: sec_rate_gt

            每个无人机的实时速率: rate_ubs_t

            全局效用: global_util

        """
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                if self.sche_U2G[k][i]:
                    self.rate_gt_t[i] = (self.comm_rate_U2G[k][i] + self.priv_rate_U2G[k][i])
                    self.rate_ubs_t[k] = self.rate_ubs_t[k] + self.rate_gt_t[i]
                    self.sec_rate_gt[i] = self.secrecy_rate_c_k_t[k] + self.secrecy_rate_p_i_t[i]

        self.avg_epi_rate_gt = (self.avg_epi_rate_gt * self.t + self.rate_gt_t) / (self.t + 1)

        self.fair_idx_t = compute_jain_fairness_index(self.avg_epi_rate_gt)
        self.avg_epi_fair_index = (self.avg_epi_fair_index * self.t + self.fair_idx_t) / (self.t + 1)
        
        self.throughput_t = self.rate_gt_t.sum()
        self.avg_epi_throughput = (self.avg_epi_throughput * self.t + self.throughput_t) / (self.t + 1)
        self.throughput = self.throughput + self.throughput_t
        
        self.sec_rate_t = self.secrecy_rate_c_k_t.sum() + self.secrecy_rate_p_i_t.sum()
        self.avg_epi_sec_rate = (self.avg_epi_sec_rate * self.t + self.sec_rate_t) / (self.t + 1)
        self.sec_rate = self.sec_rate + self.sec_rate_t # 总实时安全容量

        self.global_util = self.global_util + self.fair_idx_t * self.sec_rate_t

        
    def shannon_capacity(self, s, n):
        # 计算香农容量 (Mbps)
        return self.bw * np.log(1 + s / n) * 1e-6

    def collision_detection(self):
        self.mask_collision = ((self.dis_U2U + 99999 * np.eye(self.n_uav)) < self.safe_dist).any(1)

    def reset(self):
        self.uav_traj = []
        self.jamming_power_list = []
        self.ssr_list = []
        self.throughput_list = []
        self.fair_idx_list = []
        self.theta_list = []
        # 全局数据初始化
        self.fair_idx_t = 0 # 实时公平因子
        self.avg_epi_fair_index = 0 # 每个episode平均公平因子
        self.throughput_t = 0 # 实时吞吐量
        self.avg_epi_throughput = 0 # 每个episode平均吞吐量
        self.throughput = 0 # 每个episode的总吞吐量
        self.sec_rate_t = 0 # 实时安全容量
        self.avg_epi_sec_rate = 0 # 每个episode平均吞吐量
        self.sec_rate = 0 # 总实时安全容量
        self.sec_rate_gt = np.zeros((self.n_gt), np.float32)  # 计算GT的累积SSR
        self.rate_gt_t = np.zeros((self.n_gt), dtype=np.float32) # 每个GT的实时吞吐量，用于计算公平因子
        self.avg_epi_rate_gt = 0
        self.rate_ubs_t = np.zeros((self.n_uav), dtype=np.float32) # 每个无人机的数据速率
        self.global_util = 0
        
        # 初始化环境
        self.t = 1
        self.episo_return = np.zeros(self.n_uav, dtype=np.float32)
        self.map_info = self.map.get_map(flag=1)
        self.pos_ubs = self.map_info['pos_ubs'] # 初始化位置
        self.pos_gts = self.map_info['pos_gts'] # 每个episode随机生成
        self.pos_eves = self.map_info['pos_eves'] # 初始化位置

        self.reward = 0
        self.mean_returns = 0
        self.reward_scale = 0.1

        self.update_dist_conn() # 初始距离、关联关系、生成信道
        self.collision_detection() # 碰撞检测
        jamming_power = np.array([0 for _ in range(self.n_uav)])
        thetas = [0.5 for _ in range(self.n_uav)]
        self.transmit_data(jamming_power=jamming_power, thetas=thetas)  # 传输数据
        self.sercurity_model()  # 计算保密容量
        self.cal_glo_metric() # 计算指标: 吞吐量、奖励因子、安全容量

        obs = wrapper_obs(self.get_obs())

        state = wrapper_state(self.get_state())

        init_info = dict(range_pos=self.range_pos,
                         uav_init_pos=self.pos_ubs,
                         eve_init_pos=self.pos_eves,
                         gts_init_pos=self.pos_gts)

        return obs, state, init_info


    def step(self, actions):
        self.t = self.t + 1 # 时间步+1
        action_moves = actions['moves']
        action_powers = actions['powers']
        if self.theta_opt:
            action_thetas = actions['thetas']
        moves = self.avail_moves[np.array(action_moves, dtype=int)]  # 所有无人机的移动
        jamming_powers = self.avail_jamming_powers[np.array(action_powers, dtype=int)]  # 所有无人机的干扰功率
        thetas = [0.5 for _ in range(self.n_uav)]
        if self.theta_opt:
            thetas = self.avail_theta[np.array(action_thetas, dtype=int)] # 所有无人机的传输间隙
        self.pos_ubs = np.clip(self.pos_ubs + moves,
                               a_min=0,
                               a_max=self.range_pos)
        
        self.update_dist_conn() # 更新距离与信道
        self.collision_detection() # 碰撞检测
        self.transmit_data(jamming_power=jamming_powers, thetas=thetas) # 传输数据
        self.sercurity_model() # 计算安全模型
        self.cal_glo_metric() # 计算指标: 吞吐量、奖励因子、安全容量
        reward = self.get_reward(self.reward_scale) # 计算奖励
        self.episo_return = self.episo_return + reward # 计算回合累积回报
        self.mean_returns = self.mean_returns + reward.mean() # 计算回合平均回报
        done = self.get_terminate()
        info = dict(EpRet=self.episo_return, # TODO
                    EpLen=self.t,
                    mean_returns=self.mean_returns,  # episode的平均汇报
                    total_throughput=self.throughput,  # episode的总吞吐量
                    Ssr_Sys=self.sec_rate,  # 系统安全容量
                    global_util=self.global_util, # 全局效用
                    avg_fair_idx_per_episode=self.avg_epi_fair_index)  # 系统公平因子
        obs = wrapper_obs(self.get_obs())
        state = wrapper_state(self.get_state())
        # Mark whether termination of episode is caused by reaching episode limit.
        info['BadMask'] = True if (self.t == self.episode_length) else False
        
        # 绘图所需数据
        self.uav_traj.append(self.pos_ubs)
        self.jamming_power_list.append(jamming_powers)
        self.fair_idx_list.append(self.fair_idx_t)
        self.ssr_list.append(self.sec_rate_t)
        self.throughput_list.append(self.throughput_t)
        self.theta_list.append(thetas)
        
        return obs, state, reward, done, info


    def get_data(self):
        return dict(traj=self.uav_traj,
                    jamming=self.jamming_power_list,
                    fair_idx=self.fair_idx_list,
                    secrecy_rate=self.ssr_list,
                    thetas = self.theta_list, 
                    throughput=self.throughput_list)

    def get_env_info(self):
        obs = self.get_obs_size()
        gt_features_dim = obs['gt'][1]
        other_features_dim = obs['agent'] + np.prod(obs['ubs'])
        env_info = dict(n_ubs=self.n_uav,
                        gt_features_dim=gt_features_dim,
                        other_features_dim=other_features_dim,
                        state_shape=self.get_state_size(),
                        n_moves=self.n_moves,
                        n_powers=self.n_powers,
                        n_thetas=self.n_thetas,
                        n_agents=self.n_agents,
                        episode_limit=self.episode_length)
        
        return env_info

    def sercurity_model(self):
        # Eve以cell k为单位进行计算窃听
        self.secrecy_rate_c_k_t = np.zeros((self.n_uav), dtype=np.float32)
        for k in range(self.n_uav):
            # 计算公有
            k_cell = self.uav_serv_gt[k]
            if len(k_cell) != 0:
                self.secrecy_rate_c_k_t[k] = max(0.0, np.min(self.comm_rate_U2G[k][k_cell]) - np.max(self.comm_rate_U2E[k]))

        self.secrecy_rate_p_i_t = np.zeros((self.n_gt), dtype=np.float32)       
        for k in range(self.n_uav):
            for i in self.uav_serv_gt[k]:
                for e in range(self.n_eve):
                    if self.sche_U2E[k][e] == 1:
                        self.secrecy_rate_p_i_t[i] = max(0.0, self.priv_rate_U2G[k][i] - np.max(self.priv_rate_E7G[e]))

    def get_all_data(self):
        return dict(traj=self.uav_traj,
                    jamming_power=self.jamming_power_list,
                    fair_idx=self.fair_idx_list,
                    sec_rate=self.ssr_list,
                    throughput=self.throughput_list)

    def get_obs(self) -> list:
        return [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> dict:
        """Returns local observation of specified agent as a dict."""
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ubs_feats = np.zeros(self.obs_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        # own feats
        own_feats[0:2] = self.pos_ubs[agent_id] / self.range_pos
        own_feats[2] = ((self.secrecy_rate_c_k_t[agent_id] + self.secrecy_rate_p_i_t[self.uav_serv_gt[agent_id]].sum()) 
                        / self.achievable_rate_ubs_max)

        # UBS features
        other_ubs = [ubs_id for ubs_id in range(self.n_agents) if ubs_id != agent_id]
        for j, ubs_id in enumerate(other_ubs):
            if self.cov_U2U[agent_id][ubs_id]:
                ubs_feats[j, 0] = 1  # vis flag
                ubs_feats[j, 1:3] = (self.pos_ubs[ubs_id] - self.pos_ubs[agent_id]) / self.range_pos  # relative pos

        # GTs features
        for i in range(self.n_gt):
            if self.cov_U2G[agent_id][i]:
                gt_feats[i, 0] = 1  # vision flag
                gt_feats[i, 1:3] = (self.pos_gts[i] - self.pos_ubs[agent_id]) / self.range_pos  # relative pos
                gt_feats[i, 3] = self.sec_rate_gt[i] / self.achievable_rate_gts_max 

        return dict(agent=own_feats, ubs=ubs_feats, gt=gt_feats)
        

    def get_obs_size(self) -> dict:
        return dict(agent=self.obs_own_feats_size, ubs=self.obs_ubs_feats_size, gt=self.obs_gt_feats_size)

    @property
    def obs_own_feats_size(self) -> int:
        """
        Features of agent itself include:
        - Normalized position (x, y)
        - Normalized Security Sum Rate(SSR)
        """
        o_fs = 2 + 1
        return o_fs

    @property
    def obs_ubs_feats_size(self) -> tuple:
        """
        Observed features of each UBS include
        - Visibility flag 0 or 1
        - Normalized distance (x, y) when visible
        """
        u_fs = 1 + 2
        return self.n_agents - 1, u_fs

    @property
    def obs_gt_feats_size(self) -> tuple:
        """
        - Visibility flag 1
        - Normalized distance (x, y) when visible 2
        - Normalized instance QoS 1
        # - Normalized instance ssr gt rate 1
        """
        gt_fs = 1 + 2 + 1

        return self.n_gt, gt_fs

    def get_state(self) -> np.ndarray:
        """
        Returns features of all UBSs and GTs as global drqn_env state.
        Note that state is only used for centralized training and should be inaccessible during inference.
        """
        ubs_feats = np.zeros(self.state_ubs_feats_size(), dtype=np.float32)
        gt_feats = np.zeros(self.state_gt_feats_size(), dtype=np.float32)

        # Features of UBSs
        ubs_feats[:, 0:2] = self.pos_ubs / self.range_pos
        ubs_feats[:, 2] = self.rate_ubs_t / self.achievable_rate_ubs_max

        # Features of GTs
        gt_feats[:, 0:2] = self.pos_gts / self.range_pos
        gt_feats[:, 2] = self.rate_gt_t / self.achievable_rate_gts_max

        return np.concatenate((ubs_feats.flatten(), gt_feats.flatten()))

    def get_state_size(self) -> int:
        return np.prod(self.state_ubs_feats_size()) + np.prod(self.state_gt_feats_size())

    def state_ubs_feats_size(self) -> tuple:
        """
        State of each UBS includes
        - Normalized distance (x, y)
        - Normalized Security Sum Rate(SSR)
        """
        su_fs = 2 + 1

        return self.n_uav, su_fs

    def state_gt_feats_size(self) -> tuple:
        """
        tate of each GT includes
        - Normalized position (x, y)
        - Normalized QoS
        """
        sg_fs = 2 + 1

        return self.n_gt, sg_fs

    def get_reward(self, reward_scale_rate) -> float:
        ubs_rewards = self.fair_idx_t * self.sec_rate_t * np.ones(self.n_agents, dtype=np.float32)
        ubs_rewards = reward_scale_rate * ubs_rewards / self.achievable_rate_ubs_max
        idle_ubs_mask = (self.rate_ubs_t == 0)  
        ubs_rewards = ubs_rewards * (1 - idle_ubs_mask)  # 空闲无人机不能获得奖励
        
        if self.avoid_collision:
            ubs_rewards = ubs_rewards - self.mask_collision * self.penlty
        
        return ubs_rewards


    def get_terminate(self) -> bool:
        return True if (self.t == self.episode_length) else False

if __name__ == '__main__':
    set_rand_seed(seed=10)
    
    import argparse

    parser = argparse.ArgumentParser()

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

    