
from __future__ import division
import pandas as pd
import numpy as np
import operator

# Please change the path to change the user-item matrix
path = 'E:\Fall 2014\Social Media Mining\Homeworks\ml-100k\ml-100k\u.data'
columns = ['user_id','item_id','rating','timestamp']
mat1 = pd.read_csv(path, names=columns, delimiter='\t')
p1 = mat1.as_matrix()
#print p1
#converting the input into numpy array
user_item_matrix = np.zeros((943,1682), dtype=np.int)
user_item_matrix -= 1
for x in p1:
    user_item_matrix[x[0]-1][x[1]-1] = x[2]
print user_item_matrix

user_id = raw_input('Please Enter User_id:')
user_id = int(user_id) - 1

neighbourhood_size = raw_input('Please enter the size of neighbourhood:')
neighbourhood_size = int(neighbourhood_size)

item_id = raw_input('Please Enter item_id:')
item_id = int(item_id)-1


user_list = []
iter_list = []



def prediction(user_item_matrix):
    dict = {}
    for i in range(0 , user_item_matrix.shape[0]):
        user_list=[]
        iter_list=[]
        sum = 0
        if i == user_id:
            i += 1
            continue

        for j in range(0, user_item_matrix.shape[1]):
            if (user_item_matrix[user_id][j] > -1 and user_item_matrix[i][j] > -1):
                user_list.append(user_item_matrix[user_id][j])
                iter_list.append(user_item_matrix[i][j])

        user_arr = np.array(user_list)
        iter_arr = np.array(iter_list)
        for p in range(len(user_arr)):
            sum += user_arr[p]*iter_arr[p]
        norm_user = np.linalg.norm(user_arr)
        norm_iter = np.linalg.norm(iter_arr)
        if (norm_iter == 0 or norm_user == 0):
            continue
        sim = sum / ((norm_user)*(norm_iter))
        dict[i] = sim

    sorted_dict = sorted(dict.items(), key = operator.itemgetter(1), reverse=True)
    avg_rating_dict = {}
    for n in range(neighbourhood_size):
        avg_temp = 0
        count = 0
        temp_row = user_item_matrix[sorted_dict[n][0]]
        for item in temp_row:
            if item > -1:
                avg_temp += item
                count += 1
        avg_rating_dict[sorted_dict[n][0]] = avg_temp/count

    user_row = user_item_matrix[user_id]
    sum_rating = 0
    count = 0
    for rating in user_row:
        if rating > -1:
            sum_rating += rating
            count += 1
    avg_rating_user_id = sum_rating/count
    temp_sum = 0
    denominator = 0
    #Final Prediction:
    for n in range(neighbourhood_size):
        if(not user_item_matrix[sorted_dict[n][0]][item_id] == -1):
            temp_sum += sorted_dict[n][1]*(user_item_matrix[sorted_dict[n][0]][item_id]- avg_rating_user_id)
        denominator += sorted_dict[n][1]

    final_predicted_value_user_based = avg_rating_user_id + (temp_sum/denominator)
    return final_predicted_value_user_based

print 'user based similarity:' + str(prediction(user_item_matrix))
print 'item based similarity:' + str(prediction(user_item_matrix.transpose()))
raw_input('')



