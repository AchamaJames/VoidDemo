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

from Feature_Extraction import Signal_Feature_Extraction
BASE_PATH = '/home/achama/AudioML/audio_data'
METADATA_PATH = f'{BASE_PATH}/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt'
DATASET_PATH = f'{BASE_PATH}/PA/ASVspoof2019_PA_dev/flac/'

class Data_Preparation:
    
    def __init__(self,meta_Data_Path = METADATA_PATH,data_Set_Path=DATASET_PATH,data_Set_name="dummy"):
        self.meta_Data_Path = meta_Data_Path
        self.data_Set_Path = data_Set_Path
        self.data_Set_name = data_Set_name

    def read_Metadata(self):
        self.metadata_df = pd.read_csv(self.meta_Data_Path, sep=" ", header=None) 
        self.metadata_df.columns = ['speaker_id','filename','system_id','null','class_name']   
        self.metadata_df.drop(columns = ['null'],inplace=True)
        print(self.metadata_df)

    def get_Data_Subset(self,SAMPLE_SIZE=1200):
        self.metadata_df['filepath'] = self.data_Set_Path+self.metadata_df.filename+'.flac'
        self.metadata_df['target'] = (self.metadata_df.class_name=='spoof').astype('int32') # set labels 1 for fake and 0 for real
        print(self.metadata_df["target"].value_counts())
        if True:
            self.metadata_df = self.metadata_df.groupby(['target']).sample(SAMPLE_SIZE).reset_index(drop=True)
        print(f'Train Samples: {len(self.metadata_df)}')
        print(self.metadata_df.head(3))
        print(self.metadata_df["target"].value_counts())

    def get_filenames_labels(self):
        self.filename_seq=[]
        self.label_seq=[]
        file_with_filenames = open("filenames_ASV2019_dev.txt", "w")
        for i in range(len(self.metadata_df)):
            self.filename_seq.append(self.metadata_df.loc[i,"filepath"])
            self.label_seq.append(self.metadata_df.loc[i,"target"])
            file_with_filenames.write(self.metadata_df.loc[i,"filepath"] + " \n")
        #print("Number of filenames: ",len(self.filename_seq))
        #print("Number of labels: ",len(self.label_seq))
        #print(f"The First File: {self.filename_seq[0]}\n Label(0-real, 1-spoof):{self.label_seq[0]}")
        file_with_filenames.close()

    def get_Feature_Vector_Of_Dataset(self):
        current_path=os.getcwd()
        feature_dir_Path = os.path.join(current_path,"Feature_Vector_Extracted")
        if (os.path.exists(feature_dir_Path) ):
            print("Already exists")
            
        else:
            os.mkdir(feature_dir_Path)          
            print("New folder created")
            
        feature_File = os.path.join(feature_dir_Path, 'feature_Vector_{}.npy'.format(self.data_Set_name))
        if os.path.isfile(feature_File):
            print("The features has already been extracted!")
        else:
            print("The features has to be extracted")
            fl = np.zeros((len(self.filename_seq), 98))
            
            for name_Idx in range(len(self.filename_seq)) :            
                file_name = self.filename_seq[name_Idx]
                label = self.label_seq[name_Idx]
                if os.path.isfile(file_name):
                    audio_signal_file=Signal_Feature_Extraction(file_name)
                    FV_Void = audio_signal_file.get_all_feature_vectors()
                    print(name_Idx)
                    fl[name_Idx,0:97] = FV_Void
                    fl[name_Idx,97] = label
                else:
                    print(f"{file_name} is missing")
                    with open("Missingfilename.txt", "a") as file:
                        file.write(f"{file_name}\n")
       
        np.save(feature_File, fl)


if(__name__ == "__main__"):
    
    d1=Data_Preparation() 
    d1.read_Metadata()
    d1.get_Data_Subset()
    d1.get_filenames_labels()
    d1.get_Feature_Vector_Of_Dataset()
        #sample_signal_filename=f'{BASE_PATH}/liveaudio_Arun/vp01.wav'
    #sample1 = Signal_Feature_Extraction(d1.filename_seq[0])
    #print("hi sample1")
    #print(sample1.get_all_feature_vectors())