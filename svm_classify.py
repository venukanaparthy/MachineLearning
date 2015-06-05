import sys
import time
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def classifySVM():
    print 'Classify..'
    target_names = ['unacc', 'acc','good','v-good']
    df = pd.read_csv("data/cars-cleaned.txt", delimiter=",");    
    print df
    print df.dtypes
    df_y = df['accept']
    df_x = df.ix[:,:-1]

    #print df_y
    #print df_x
    train_y, test_y, train_x, test_x = train_test_split(df_y, df_x, test_size = 0.3, random_state=33)
    print len(train_y)
    print train_x
    print '--------------------'
    print len(test_y)    
    print test_x
    
    clf = svm.SVC(kernel="linear", C=0.01)
    tstart=time.time()
    model = clf.fit(train_x, train_y)
    print "training time:", round(time.time()-tstart, 3), "seconds"
    y_predictions = model.predict(test_x)
    print "Accuracy : " , model.score(test_x, test_y)
    print y_predictions
    c_matrix = confusion_matrix(test_y,y_predictions)
    print "confusion matrix:"
    print c_matrix

    plt.matshow(c_matrix)
    plt.colorbar();
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)    
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()
    
    #scaler = MinMaxScaler()
    #train_x_scaled = scaler.fit_transform(train_x)
    #print train_x_scaled
    #test_x_scaled = scaler.fit_transform(test_x)
    #print test_x_scaled
    
if __name__ == "__main__":
     classifySVM()
