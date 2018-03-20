import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import os
import csv
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np
data_train = np.loadtxt('completetrain.csv', delimiter=',')
X = data_train[:, 1:]
y = data_train[:, 0].astype(np.int)

def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = 7480, output_dim = 5, input_length = input_length))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
model = create_model(len(X[0]))

hist = model.fit(X, y, batch_size=200, nb_epoch=10, validation_split = 0.1, verbose = 1)

score, acc = model.evaluate(X, y, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)
'''
#clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)
#clf = svm.SVC(kernel="linear", C= 1.0).fit(X,y)
clf = LinearSVC(random_state=0)
#clf = RandomForestClassifier(max_depth=2000, random_state=0)
clf.fit(X, y)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
data_test = np.loadtxt('criminal_test.csv', delimiter=',')
print(clf.predict(data_test))
#print(clf1.predict(data_test))

def predictTest():
    import os
    FTEST_PATH = '/home/cse/Downloads/criminal_ml'     
    outputpath = '/home/cse/Downloads/criminal_ml/net5Predictions.csv'
    inputNamesPath = '/home/cse/Downloads/criminal_ml/criminal_test.csv'
    with open(outputpath, "w") as outfile:
        outfile.write("grade\n")    
    os.chdir(FTEST_PATH)
    j=0
    for fname in sorted(os.listdir('.'), key=os.path.getmtime):
        FTEST = FTEST_PATH+'/'+fname
	data_test = np.loadtxt('criminal_test.csv', delimiter=',')
        X = data_test[:]
        a_pred = clf.predict(X)
        with open(outputpath, "a") as outfile:
            for i in range(0,X.shape[0]):
                #if i+j < names.shape[0]:
                
                label = str(a_pred[i])
                outfile.write( label+"\n")
        j=j+X.shape[0]
    os.chdir('../')

predictTest()
'''
