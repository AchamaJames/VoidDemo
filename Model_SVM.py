
import os
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
import numpy as np
from scipy.signal import find_peaks
import math
import scipy as sp
import scipy.signal as sig
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time
from datetime import timedelta
from Feature_Extraction import Signal_Feature_Extraction

class model_SVM:
    def __init__(self,feature_Vector_File_Path='/home/achama/Projects/Void/Feature_Vector_Extracted/feature_Vector_dev_full.npy',data_Set_For="train"):

        self.scaler=MinMaxScaler()
        self.feature_Vector_File_Path=feature_Vector_File_Path
        self.data_Set_For=data_Set_For
        self.labels_dict = { 0 : "Live", 1 : "Replay"}
        if os.path.isfile(self.feature_Vector_File_Path):
            # Load the extracted features and corresponding labels:
            self.data_Set_Features = np.load(self.feature_Vector_File_Path)
            #print(self.data_Set_Features.shape)
            if True:
                self.data_Set_Features = self.data_Set_Features[:28000,:]
            
            self.X_data, self.y_data = np.split(self.data_Set_Features,[-1],axis=1)
            self.y_data=self.y_data.ravel()
            self.y_data=self.y_data.astype(np.int32)
            #print("X shape:",self.X_data.shape)
            #print("Y shape:",self.y_data.shape)
            #print("Y_data",self.y_data)
            self.X_data=np.nan_to_num(self.X_data)
            #print("q max:",self.X_data[:,1].max())
            self.X_data[:,1] = np.where(self.X_data[:,1]>100,100,self.X_data[:,1])
            #print("q min:",self.X_data[:,1].min())
            #self.X_data[:,49:] = self.X_data[:,49:]*100

            '''
            print(self.X_data[:,:5].max(axis=0))
            self.scaler.fit(self.X_data[:,:5])
            self.X_data[:,:5] = self.scaler.transform(self.X_data[:,:5])
            print(self.X_data[:,:5].max(axis=0))
            self.X_data[:,49:] = self.X_data[:,49:]*100
            #LPCC=self.X_data[:,37:49]
            #print(LPCC)
            #norm_arr = LPCC / np.linalg.norm(LPCC, axis=1, keepdims=True)
            #norm_arr=(LPCC - LPCC.min(1)[:,None]) / (LPCC.max(axis=1)[:, None] - LPCC.min(axis=1)[:, None])
            #self.X_data[:,37:49]=norm_arr
            #print(norm_arr)
            #print(self.X_data[0,37:49])
            '''
        else:
            print('Feature extraction has not been done. Please extract Void features using data_Set_preprocess.py!')

    def Outlier_capping(self,X):
        
        print("q max:",X[:,1].max())
        X[:,1] = np.where(X[:,1]>100,100,X[:,1])
        print("q min:",self.X[:,1].min())
        return X
        #self.X_data[:,:5] = self.scaler.transform(self.X_data[:,:5])  


    def Scaling_fit_transorm(self):
        self.X_train[:,:5] = self.scaler.fit(self.X_train[:,:5])
        self.X_train[:,:5] = self.scaler.transform(self.X_train[:,:5])

    def train_model(self):
        #self.X_data = self.Outlier_capping(self.X_data)
        self.X_train,self.X_test, self.y_train, self.y_test = train_test_split(self.X_data,self.y_data, test_size = 0.4, random_state=42,stratify=self.y_data)

        print(self.X_data.shape, self.X_train.shape, self.X_test.shape)
        print("y_train",self.y_train)
        self.X_train[:,:5]=self.scaler.fit_transform(self.X_train[:,:5])
        #self.X_train[:,:5]=self.X_train[:,:5]/100
        self.classifier = SVC(C=10, kernel='rbf')
        
        
        #self.scaler.transform(self.X_train[:,:5])
        #self.Scaling_fit_transorm()
        self.classifier.fit(self.X_train,self.y_train)
        self.X_train_prediction = self.classifier.predict(self.X_train)
        print("X_train prediction\n",pd.Series(self.X_train_prediction).value_counts())
        training_data_accuracy = accuracy_score( self.y_train,self.X_train_prediction)
        
        print("Training Accuracy:",training_data_accuracy)

        self.X_test[:,:5] = self.scaler.transform(self.X_test[:,:5])
        #self.X_test[:,:5]=self.X_test[:,:5]/100
        self.X_test_prediction = self.classifier.predict(self.X_test)
        print("X_test prediction\n",pd.Series(self.X_test_prediction).value_counts())
        test_data_accuracy = accuracy_score( self.y_test,self.X_test_prediction)
        
        print("Test Accuracy:",test_data_accuracy)

    def save_model(self):
        current_path=os.getcwd()
        self.model_dir_Path = os.path.join(current_path,"Models")
        if (os.path.exists(self.model_dir_Path) ):
            pass
            #print("Already exists")
            
        else:
            os.mkdir(self.model_dir_Path)          
            print("New folder created")

        model_to_be_saved = pickle.dumps(self.classifier)
        self.save_model_filename = os.path.join(self.model_dir_Path, 'svm_with_std.pkl')
        f = open(self.save_model_filename, 'wb+')
        f.write(model_to_be_saved)
        f.close()
        print("Model saved into " + self.save_model_filename)
    
    def load_model(self):
        with open(self.save_model_filename,'rb') as file:
            print("The model pickle file Loaded:",self.save_model_filename)
            self.classifier = pickle.load(file)
    def predict_model(self,X):
        #X[:,:5] = self.scaler.transform(X[:,:5])
        X[:,1] = np.where(X[:,1]>100,100,X[:,1])
        X[:,:5] = (X[:,:5])/100
        #print("The signal Vector:\n",X)
        #X[:,49:] = X[:,49:]*100
        prediction = self.classifier.predict(X) 
        print("Prediction :", self.labels_dict[prediction[0]])

    def evaluate_model(self):
        self.load_model()
        self.X_data_prediction = self.classifier.predict(self.X_data)
        test_data_accuracy = accuracy_score( self.y_data,self.X_data_prediction)
        
        print("Accuracy of the dataset given:",test_data_accuracy)

if(__name__ == "__main__"):

    model1=model_SVM()
    current_path=os.getcwd()
    model1.model_dir_Path = os.path.join(current_path,"Models")
    model1.save_model_filename = os.path.join(model1.model_dir_Path, 'svm_with_std.pkl')
    #model1.Outlier_capping()
    model1.train_model()
    #model1.save_model()
    #current_path=os.getcwd()
    #model1.model_dir_Path = os.path.join(current_path,"Models")
    #model1.save_model_filename = os.path.join(model1.model_dir_Path, 'svm_1200.pkl')
    
    start_time = time.monotonic()
    model1.load_model()
    FV_sample1=Signal_Feature_Extraction(os.path.join(current_path,"audio_data/sample1_live_ted_1096.wav")) 
    FV_Input_File = FV_sample1.get_all_feature_vectors()
    #print("Shape of input file:",FV_Input_File.shape)
    FV_Input_File_reshaped = FV_Input_File.reshape(1,-1)
    #print("Shape of input file:",FV_Input_File_reshaped.shape)
    model1.predict_model(FV_Input_File_reshaped)    
    end_time = time.monotonic()
    print("Total time for execution: ",timedelta(seconds=end_time - start_time).total_seconds())
   