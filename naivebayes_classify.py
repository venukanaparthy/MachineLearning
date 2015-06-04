import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

def classify():
    print 'Classify..'
    df = pd.read_csv("data/cars-cleaned.txt", delimiter=",");    
    print df
    print df.dtypes
    df_y = df['accept']
    df_x = df.ix[:,:-1]

    #print df_y
    #print df_x
    train_y, test_y, train_x, test_x = train_test_split(df_y, df_x, test_size = 0.3, random_state=33)
    print len(train_y)
    print len(train_x)
    print '--------------------'
    print len(test_y)    
    print len(test_x)
    
    clf = GaussianNB()
    model = clf.fit(train_x, train_y)
    predictions = model.predict(test_x)
    print "Accuracy : " , model.score(test_x, test_y)
    print "predictions---------------"
    print len(predictions==test_y)/len(predictions)
    
if __name__ == "__main__":
     classify()
