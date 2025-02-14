import numpy as np

def generate_channel(Nt, sigma_err_sq):
    """
    生成含估计误差的小尺度衰落信道
    :param Nt: 无人机天线数
    :param sigma_err_sq: 估计误差方差 (σ_err^2)
    :return: 实际信道 h_k_e (复数向量), 估计信道 hat_h (复数向量)
    """
    # 1. 生成估计信道部分 (复高斯分布)
    # 实部和虚部分别的方差为 (1 - sigma_err_sq)/2
    real_hat = np.random.normal(0, np.sqrt((1 - sigma_err_sq)/2), Nt)
    imag_hat = np.random.normal(0, np.sqrt((1 - sigma_err_sq)/2), Nt)
    hat_h = real_hat + 1j * imag_hat  # 合并为复数
    
    # 2. 生成估计误差部分 (复高斯分布)
    # 实部和虚部分别的方差为 sigma_err_sq/2
    real_delta = np.random.normal(0, np.sqrt(sigma_err_sq/2), Nt)
    imag_delta = np.random.normal(0, np.sqrt(sigma_err_sq/2), Nt)
    delta_h = real_delta + 1j * imag_delta  # 合并为复数
    
    # 3. 合成实际信道
    h = hat_h + delta_h
    
    return h, hat_h

# 参数设置示例
Nt = 1          # 无人机天线数
sigma_err_sq = 0.1  # 估计误差方差
h, hat_h = generate_channel(Nt, sigma_err_sq)

print(h)

# 验证信道总功率是否接近 1（理论值）
print("实际信道功率:", np.mean(np.abs(h)**2))  # 应接近 1.0
print("估计信道功率:", np.mean(np.abs(hat_h)**2))  # 应接近 1 - sigma_err_sq