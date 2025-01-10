import numpy as np

if __name__ == '__main__':
    n_gt = 10
    gt_norm_2 = np.zeros((n_gt), dtype=np.float32)
    print(gt_norm_2)
    gts_in_community = np.array([5, 6, 7, 8, 9])
    gt_norm_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.1, 0.4, 0.5])
    best_gt_in_community = max(gts_in_community, key=lambda x: gt_norm_2[x])
    print(best_gt_in_community)

    for i in range(5):
        x = 5
        for j in range(2):
            x = x + 1
            print(x)
        print()