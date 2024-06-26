'''like the process of elimination the decision tree algorithm uses a collection of information to best guess the correct answer. It's like the game of Guess who, the algorithm uses your input to best guess the relational outcome'''

'''Note: the decision tree will give you different results each time because it gives you a probable answer which is never 100% accurate'''
# https://www.w3schools.com/python/python_ml_decision_tree.asp

# DEcision Tree Rules: 
# - all data must be numerical. strings must be converted 
# 

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import sys


# training model
def training_model(): 
# step 1: read CSV
    data_set = pd.read_csv('data.csv')
    
# step 2: map Data (like cleaning the data/changing the value) 
    d= {'UK':0, 'USA':1, 'N':2}
    data_set['Nationality']=data_set['Nationality'].map(d)
    
    d= {'YES': int(1), 'NO':int(0)}
    data_set['Go'] = data_set['Go'].map(d)
    
    
   
#Step 3:  separate the feature (Learning data ("all other columns")) column from the target column (result data ("Go" column))
    features = ['Age', 'Experience', 'Rank', 'Nationality'] #"Go is missing"
    target = ['Go']
    # Training DAta
    x = data_set[features] # 
    y = data_set[target] #results data
    
    
     
# step 4:
# Model instance 
    model = DecisionTreeClassifier()
    model.fit(x,y) 

# Step 5: plot the tree 
    tree.plot_tree(model, feature_names=features)
    
# Step 6: Human visible display-- these Two  lines make our compiler able to draw

    # plt.savefig('dtree_diagram.png')#to save diagram to file 
    plt.show() #plots the images you created in your code
    sys.stdout.flush() #this flushed the write-buffer, making sure anything in the stream is displayed



training_model()

# NOTE: matplotlib.pyplot module is designed to display all figures that have been created during a scrip
