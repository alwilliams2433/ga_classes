import pickle
import argparse
import numpy as np
import sys


# Example:
# python interface.py -slen 4 -swidth 6 -plen 8 -pwidth 10


#######################
# READ THE INPUT DATA #
#######################

parser = argparse.ArgumentParser()
#sepal length (cm) 	
parser.add_argument('-slen', action='store', dest='slen', help='sepal length (cm) ')
#sepal width (cm) 
parser.add_argument('-swidth', action='store', dest='swidth', help='sepal width (cm)')
#petal length (cm) 
parser.add_argument('-plen', action='store', dest='plen', help='petal length (cm)')
#petal width (cm)
parser.add_argument('-pwidth', action='store', dest='pwidth', help='petal width (cm)')
results = parser.parse_args()


data = np.array([float(results.slen), 
				 float(results.swidth), 
				 float(results.plen), 
				 float(results.pwidth)]).reshape(1, -1)


############################################
# LOAD THE MODEL AND OUTPUT THE PREDICTION #
############################################

with open('mymodel.pickle', 'rb') as handle:
    model = pickle.load(handle)
    prediction = model.predict(data)
    print('My prediction is class {}'.format(prediction[0]))