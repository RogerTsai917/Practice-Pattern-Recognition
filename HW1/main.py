import numpy as np
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

def calculateEachClassProbabiluty(x, parameters_dict):
    log_p = np.log(parameters_dict['probability'])
    matrices_dot = np.transpose(x-parameters_dict['mean']).dot(parameters_dict['inv_of_cov']).dot(x-parameters_dict['mean'])
    log_det_of_cov = np.log(parameters_dict['det_of_cov'])

    return -log_p + (matrices_dot/2) + (log_det_of_cov/2)

def predict(feature_test, label_test, each_class_parameters_dict):
    correct = 0
    error = 0
    for i, item in enumerate(feature_test):
        predict_label = 0
        min_probability = np.finfo(np.float).max
        for key in each_class_parameters_dict:
            probabiluty = calculateEachClassProbabiluty(item, each_class_parameters_dict[key])
            if probabiluty < min_probability:
                predict_label = key
                min_probability = probabiluty
        if predict_label == label_test[i]:
            correct += 1
        else:
            error += 1
    return correct/(correct+error)

if __name__ == "__main__":
    data_file_name = 'wine.data'
    label_position = 0
    test_size = 0.5
    random_state = 1
    
    feature_list, label_list = readData(data_file_name, label_position=label_position)
    
    feature_train, feature_test, label_train, label_test = train_test_split(feature_list, label_list, test_size=test_size, random_state=random_state)
    
    each_class_dataset_dict = getEachClassDatasetDict(feature_train, label_train)
    
    each_class_parameters_dict = calculateEachClassParameters(each_class_dataset_dict)
    
    predict_accuracy = predict(feature_test, label_test, each_class_parameters_dict)

    print(f"data set name: {data_file_name}")
    print(f"predict accuracy: {predict_accuracy*100}%")