import numpy as np

if __name__ == '__main__':
    n_uav = 3
    
    a = np.array([[0, 4, 7], 
               [4, 0, 8],
               [7, 8, 0]])
    
    print(a[1].sum())