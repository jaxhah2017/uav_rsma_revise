import numpy as np


class AirToGroundChannel(object):
    """Air-to-ground (ATG) channel model"""
    # from https://github.com/zhangxiaochen95/uav_bs_ctrl/blob/main/envs/common.py/
    chan_params = {
        'suburban': (4.88, 0.43, 0.1, 21),
        'urban': (9.61, 0.16, 1, 20),
        'dense-urban': (12.08, 0.11, 1.6, 23),
        'high-rise-urban': (27.23, 0.08, 2.3, 34)
    }

    def __init__(self, scene, fc: float = 2.4e9, apply_small_fading: bool = False):
        # Determine the scene-specific parameters.
        params = self.chan_params[scene]
        self.a, self.b = params[0], params[1]  # Constants for computing p_los
        self.eta_los, self.eta_nlos = params[2], params[3]  # Path loss exponents (LoS/NLoS)
        self.fc = fc  # Central carrier frequency (Hz)
        self.apply_small_fading = apply_small_fading # 应用小尺度衰落

    def estimate_chan_gain(self, d_level: float, h_ubs: float = 100, K_richan=-40) -> float:
        """Estimates the channel gain from horizontal distance."""
        """large scale"""
        # Estimate probability of LoS link emergence.
        p_los = 1 / (1 + self.a * np.exp(-self.b * (np.arctan(h_ubs / (d_level + 1e-5)) - self.a)))
        # Get direct link distance.
        d = np.sqrt(np.square(d_level) + np.square(h_ubs)) + 0.0000000000000000000000000000000000000000000001
        # Compute free space path loss (FSPL).
        fspl = (4 * np.pi * self.fc * d / 3e8) ** 2
        # Path loss is the weighted average of LoS and NLoS cases.
        pl = p_los * fspl * 10 ** (self.eta_los / 10) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 10)
        h = 1 / pl

        if self.apply_small_fading:
            """small scale"""
            h_rayleigh = (np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2)
            # K_richan = np.power(10, K_richan / 10)  # K_richan (DB)
            # h_richan = np.sqrt(K_richan / (K_richan + 1)) + np.sqrt(1 / (K_richan + 1)) * h_rayleigh
            
            """channel gain"""
            h = np.sqrt(h) * h_rayleigh
            h = complex(h[0])
            print(type(h))

        return h


class GroundToGroundChannel(object):
    def __init__(self, fc, apply_small_fading: bool = False):
        self.fc = fc
        self.apply_small_fading = apply_small_fading

    def estimate_chan_gain(self, d: float) -> float:
        """channel gain"""
        """large scale"""
        d = d + 0.0000000000000000000000000000001
        fspl = (4 * np.pi * self.fc * d / 3e8) ** 2
        h = 1 / fspl
        
        if self.apply_small_fading:
            """small scale"""
            h_rayleigh = (np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2)
            h = np.sqrt(h) * h_rayleigh
            h = complex(h[0])
            print(type(h))

        return h


class ChannelEstimateError(object):
    def __init__(self, std):
        self.var = std ** 2

    def estimate_chan_error_gain(self):
        g_delta = (np.random.randn(1) + 1j * np.random.randn(1)) * np.sqrt(
            self.var / 2)  # Complex Gaussian distribution (0, 0.01)

        return g_delta[0]


if __name__ == '__main__':
    scene = 'urban'
    fc = 2.4e9
    ATGChannel = AirToGroundChannel(scene, fc, apply_small_fading=True)  # UAV->GT, Eve
    GTGChannel = GroundToGroundChannel(fc, apply_small_fading=True)  # GT->GT, GT->Eve
    g_atg_max = ATGChannel.estimate_chan_gain(d_level=0, h_ubs=1, K_richan=-40)  # ATG Maximum channel gain

    # for i in range(10):
    #     print(ATGChannel.estimate_chan_gain(d_level=i*10, h_ubs=1, K_richan=-40))

    print(g_atg_max)
    # print(type(g_atg_max))
    # print("ATG channel max:", abs(g_atg_max) ** 2)
    # print(abs(ATGChannel.estimate_chan_gain(d_level=400, h_ubs=100, K_richan=-40)) ** 2)

    g_gtg_max = GTGChannel.estimate_chan_gain(1)  # GTG channel gain
    print(g_gtg_max)
    # print(type(g_gtg_max))
    # print("GTG channel max:", abs(g_gtg_max) ** 2)
    # snr_c_atg_max = self.p_tx_c * self.n_gts * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)
    # snr_c_gtg_max = self.p_forward * self.n_gts * np.power(abs(g_gtg_max), 2) / (self.n0 * self.bw)
    # snr_p_max = self.p_tx_p * np.power(abs(g_atg_max), 2) / (self.n0 * self.bw)
    # achievable_c_max = self.bw * (np.log(1 + snr_c_atg_max) + np.log(1 + snr_c_gtg_max)) * 1e-6
    # achievable_p_max = self.bw * np.log(1 + snr_p_max) * 1e-6
    # self.max_achievable_rate = achievable_c_max + achievable_p_max
