import numpy as np
import matplotlib.pyplot as plt

def closest_node(data, t, map, m_rows, m_cols):
    # (row,col) of map node closest to data[t]
    result = (0,0)
    small_dist = 1.0e20
    for i in range(m_rows):
        for j in range(m_cols):
            ed = euc_dist(map[i][j], data[t])
            if ed < small_dist:
                small_dist = ed
                result = (i, j)
    return result

def euc_dist(v1, v2):
    return np.linalg.norm(v1 - v2) 

def manhattan_dist(r1, c1, r2, c2):
    return np.abs(r1-r2) + np.abs(c1-c2)

def most_common(lst, n):
    # lst is a list of values 0 . . n
    if len(lst) == 0: return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[lst[i]] += 1
    return np.argmax(counts)

if __name__=="__main__":
    np.random.seed(1)
    Dim = 13
    Rows = 30; Cols = 30
    RangeMax = Rows + Cols
    LearnMax = 0.5
    StepsMax = 3000
    
    # 1. load data
    print("\nLoading data into memory \n")
    data_file = "wine.txt"
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,13),
        dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", usecols=[0],
        dtype=np.int)

    # 2. construct the SOM
    print("Constructing SOM")
    map = np.random.random_sample(size=(Rows,Cols,Dim))
    for s in range(StepsMax):
        if s % (StepsMax/10) == 0: print("step = ", str(s))
        pct_left = 1.0 - ((s * 1.0) / StepsMax)
        curr_range = (int)(pct_left * RangeMax)
        curr_rate = pct_left * LearnMax

        t = np.random.randint(len(data_x))
        (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
        for i in range(Rows):
            for j in range(Cols):
                if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])
    print("SOM construction complete \n")

    # 3. associate each data label with a map node
    print("Associating each data label to one map node ")
    mapping = np.empty(shape=(Rows,Cols), dtype=object)
    for i in range(Rows):
        for j in range(Cols):
            mapping[i][j] = []

    for t in range(len(data_x)):
        (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
        mapping[m_row][m_col].append(data_y[t])

    label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
    for i in range(Rows):
        for j in range(Cols):
            label_map[i][j] = most_common(mapping[i][j], 1000)
 
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 13))
    plt.colorbar()
    plt.show()