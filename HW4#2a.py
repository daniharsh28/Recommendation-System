__author__ = 'Harsh'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)
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

########################################
# User-item matrix is ready here.

movie_id = [63,126,186,55,176,317,177,68,356,181]

user_item_matrix = user_item_matrix[:,movie_id]
print user_item_matrix
U, _S, V = np.linalg.svd(user_item_matrix)#, full_matrices= True)
V = V.transpose()
print V
U = U[:,[0,1]] #Extarcting first two columns of U


S = np.zeros((943,1682), dtype= np.int)
np.fill_diagonal(S,_S)
S = S[[0,1],:]
S = S[:,[0,1]] #Extracting first two rows and columns of U

V = V[[0,1],:]
 #Extracting first two rows of U
######################################################
# Plotting User graph
#row_num = 0
#for row in U:
#    x = row[0]
#    y = row[1]
#    plt.plot(x,y,'bx')
#    plt.text(x,y, str(row_num+1), fontsize = 5)
#    row_num += 1

#plt.savefig('User graph')
#plt.close()

##########################################################
# Plotting Item graph

row_num1 = 0
V_prime = V.transpose()
for row1 in V_prime:
    x1 = row1[0]
    y1 = row1[1]
    plt.plot(x1,y1,'bx')
    plt.text(x1,y1, str(movie_id[row_num1]+1), fontsize=10)
    row_num1 += 1


plt.savefig('Item graph')
plt.close()