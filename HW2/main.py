import numpy as np
from math import log
from math import sqrt
from math import exp
from itertools import combinations
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.model_selection import train_test_split

def readData(file_name, label_position=0):
    feature_list = []
    label_list = []
    with open(file_name, 'r') as fp:
        line = fp.readline()
        while line:
            line_list = line.strip('\n').split(',')
            feature = []
            for i, item in enumerate(line_list):
                if i == label_position:
                    label_list.append(item)
                else:
                    feature.append(float(item))
            feature_list.append(feature)
            line = fp.readline()
    return feature_list, label_list

def getEachClassDatasetDict(feature_list, label_list):
    result_dict = {}
    for i, item in enumerate(label_list):
        if item not in result_dict:
            result_dict[item] = {}
            result_dict[item]['count'] = 1
            result_dict[item]['feature_array'] = np.array(feature_list[i], ndmin=2)
        else:
            result_dict[item]['count'] += 1
            result_dict[item]['feature_array'] = np.vstack((result_dict[item]['feature_array'], feature_list[i]))
    return result_dict

def calculateEachClassParameters(dataset_dict):
    result_dict = {}
    total_number_of_data = 0
    for key in dataset_dict.keys():  
        total_number_of_data += dataset_dict[key]['count']
    for key in dataset_dict.keys():
        result_dict[key] = {}
        # calculate data number
        result_dict[key]['count'] = dataset_dict[key]['count']
        # calculate probability
        result_dict[key]['probability'] = dataset_dict[key]['count']/total_number_of_data
        # calculate mean
        result_dict[key]['mean'] = np.mean(dataset_dict[key]['feature_array'], axis=0)
        # calculate covariance matrix
        result_dict[key]['cov'] = np.cov(np.transpose(dataset_dict[key]['feature_array']))
        # calculate inverse of covariance matrix
        result_dict[key]['inv_of_cov'] = inv(result_dict[key]['cov'])
        # calculate determinant of covariance matrix
        result_dict[key]['det_of_cov'] = det(result_dict[key]['cov'])

    return result_dict

def bhattacharyyaBound(first_class, second_class):
    part1 = np.transpose(second_class['mean']-first_class['mean']).dot(inv((first_class['cov']+second_class['cov'])/2)).dot(second_class['mean']-first_class['mean'])/8
    part2 = log(det((first_class['cov']+second_class['cov'])/2)/sqrt(first_class['det_of_cov']*second_class['det_of_cov']))*0.5
    P1 = first_class['count']/(first_class['count']+second_class['count'])
    P2 = second_class['count']/(first_class['count']+second_class['count'])
    return sqrt(P1*P2)*exp(-(part1+part2))

def calculateBhattacharyyaBound(each_class_parameters_dict):
    all_class_list = sorted(each_class_parameters_dict.keys())
    combinations_list = list(combinations(all_class_list, 2))
    for combination in combinations_list:
        bhattacharyya_bound = bhattacharyyaBound(
            each_class_parameters_dict[combination[0]], 
            each_class_parameters_dict[combination[1]])
        print(f'Bhattacharyya Bound of class {combination[0]} and class {combination[1]}: {bhattacharyya_bound}')

if __name__ == "__main__":
    # 資料檔案名稱，wine.data:紅酒資料集、iris.data:鳶尾花資料集
    data_file_name = 'wine.data'

    # 資料中label所在的位置，ex.紅酒在0，鳶尾花在4
    label_position = 0
    
    # 讀取資料，分別取得 feature list 和 label list
    feature_list, label_list = readData(data_file_name, label_position=label_position)
    
    # 把 dataset依照 label 分類
    each_class_dataset_dict = getEachClassDatasetDict(feature_list, label_list)
    
    # 依照不同 label 的 data 算出各自的參數
    each_class_parameters_dict = calculateEachClassParameters(each_class_dataset_dict)
    
    # 計算不同類別之間的 Bhattacharyya Bound
    calculateBhattacharyyaBound(each_class_parameters_dict)
