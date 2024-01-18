import os
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
import numpy as np
from scipy.signal import find_peaks
import math
import scipy as sp
import scipy.signal as sig


class Signal_Feature_Extraction:
    WIN_LEN=1024
    FFT_SIZE = 4096
    HOP_SIZE = 256
    SR=44100
    

    def __init__(self,signal_filename) -> None:
        self.signal_filename=signal_filename
        self.signal_array,_=librosa.load( signal_filename,sr=Signal_Feature_Extraction.SR)
        self.stft_signal=None
        self.power_signal=None
        self.power_db_signal=None
        self.S_pow = None
        self.S_pow_db = None
        self.power_vector_norm01=None
        self.power_vector_normal=None
        self.FREQ_ROW_WISE=Signal_Feature_Extraction.SR/Signal_Feature_Extraction.FFT_SIZE

        self.FV_LDF=None
        self.FV_LFP=None
        self.FV_HPF=None
        self.FV_LPCC=None



    def stft_signal_process(self):
        self.stft_signal = librosa.stft(self.signal_array,n_fft=Signal_Feature_Extraction.FFT_SIZE,
                                   win_length=Signal_Feature_Extraction.WIN_LEN, hop_length=Signal_Feature_Extraction.HOP_SIZE,
                                   window='hamming')
        self.power_signal=np.abs(self.stft_signal) ** 2
        self.power_db_signal=librosa.power_to_db(self.power_signal)
        #return spectrogram 

    #Function to plot spectogram in power
    def plot_spectrogram(self,Y,  y_axis="linear"):
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(Y,
                             sr=Signal_Feature_Extraction.SR,
                             hop_length=Signal_Feature_Extraction.HOP_SIZE,
                             x_axis="time",
                             y_axis=y_axis,
                             )
        plt.colorbar(format="%+2.f")

    
    #Function to plot spectogram in power(db)
    def plot_spectrogram_power_db(self):
        self.plot_spectrogram(self.power_db_signal)


    #Cumulative spectral power S_pow is calculated
    #spectral power S_pow is converted into db for plotting
    def get_S_power(self):
        self.S_pow = np.sum(np.abs(self.stft_signal)**2/Signal_Feature_Extraction.FFT_SIZE, axis=1)
        self.S_pow_db=librosa.power_to_db(self.S_pow)
        
    #Function to normalize between 0 and 1
    def normalize_array(self,x):
        x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
        return x_norm
    
    #Plotting the curve of the best fit polynomial for replay sample
    def plot_poly_curve(x_values,parameters):
        new_x=[]
        new_y=[]
        poly=np.poly1d(parameters)
        for i in x_values:
            new_x.append(i)
            calc=poly(i)
            new_y.append(calc)
        plt.plot(new_x,new_y)    
        plt.show()


    def plot_power_spectrum(self,FREQ_SIZE=800,x_label="Frequency",
                        title="The cumulative power spectral decay ",y_label="Power"
                      ):
        #sampling frequncy sr=44100 ,nfft =4096,so each frequency binof stft output =44100/4096=10.77
        #Here we are plotting the curve 8K frequency range 

        #frequency range for plotting the graph is set
        #freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
        

        freqs=np.linspace(0,10.77*self.S_pow.size,self.S_pow.size)
        plt.plot(freqs[:FREQ_SIZE],self.S_pow[:FREQ_SIZE],c='b',label="Signal")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("The cumulative power spectral decay ")
        plt.legend()
        plt.grid(True)
        plt.savefig("powerspectrum")
        
        
    def plot_power_spectrum_db(self,FREQ_SIZE=800,x_label="Frequency",
                        title="The cumulative power spectral decay ",y_label="Power"
                      ):
        #sampling frequncy sr=44100 ,nfft =4096,so each frequency binof stft output =44100/4096=10.77
        #Here we are plotting the curve 8K frequency range 

        #frequency range for plotting the graph is set
        #freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
        

        freqs=np.linspace(0,10.77*self.S_pow_db.size,self.S_pow_db.size)
        plt.figure()
        plt.plot(freqs[:FREQ_SIZE],self.S_pow_db[:FREQ_SIZE],c='b',label="Signal")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("The cumulative power spectral decay ")
        plt.legend()
        plt.grid(True)
        plt.savefig("powerspectrum")
        
    #Function to plot general x_values and y_values
    def plot_spectrum(self,y_values,x_label="Frequency",y_label="Power",FREQ_SIZE=80):                     
        #sampling frequncy sr=44100 ,nfft =4096,so each frequency binof stft output =44100/4096=10.77
        #Here we are plotting the curve 8K frequency range 

        #frequency range for plotting the graph is set
        #freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
        x_values=np.linspace(0,self.FREQ_ROW_WISE*y_values.size,y_values.size)
        plt.figure()
        plt.plot(x_values[:FREQ_SIZE],y_values[:FREQ_SIZE],c='b',label="Signal")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("The power Spectrum ")
        plt.legend()
        plt.grid(True)
        plt.savefig("powerspectrum.png")
        
    def plot_rho_q_LDF(self):
        freq=np.linspace(0,self.FREQ_ROW_WISE*self.power_cdf.size,self.power_cdf.size)
        plt.figure()
        plt.plot(freq[:80],self.power_cdf[:80],label="ρ - Correlation Coefficient\nq=Quadratic coefficient")

        plt.title("Cumulative distribution of spectral power density over frequencies, showing up to 8kHz ")
        plt.text(4000,.8, f'ρ={self.correlation_coefficient:.4f}',  
                fontsize =10, color = 'r') 

        plt.text(4000,.75, f'q={self.q:.4f}',  
                fontsize =10, color = 'b') 

        plt.xlabel('Frequency (Hz)')
        plt.ylabel("cdf")
        plt.legend()
        plt.grid(True)
        plt.savefig("rho_q_LDF.png")
       
   
   
    #Function to extract Low Frequency Power Features
    def extract_FV_LFP(self):
        #STFT of the signal calculated
        self.stft_signal_process()

        ##S_pow of the signal
        self.get_S_power()

        #Plotting just one signal
        #self.plot_power_spectrum(y_label="Power in db")
        
        self.S_pow_cumsum=np.cumsum(self.S_pow)
        TOTAL_SIZE=1500  #The number of rows(ie frequency bins) of S_pow
        FREQUENCY_RANGE=16155 #The total frequency range taken into consideration, 1500*10.77=16155Hz
        self.S_pow=self.S_pow[:TOTAL_SIZE]
        TOTAL_SIZE=1500  #The number of rows(ie frequency bins) of S_pow
        FREQUENCY_RANGE=16155 #The total frequency range taken into consideration, 1500*10.77=16155Hz
        self.S_pow=self.S_pow[:TOTAL_SIZE]

        #Step 3 - Calculate k for the algorithm where k = size(Spow)/W
        W=10
        k=len(self.S_pow)//W
        
        #Step 4 - Divide Spow into k short segments with segment size W
        S_POWER_SIZE=k*W # taking the floor division to take exactly k segments
        #print("S_POWER_SIZE:",S_POWER_SIZE)
        #Making sizeof(S_power) exactly divisible by W,(W=10)
        self.S_pow=self.S_pow[:S_POWER_SIZE]

        #Step 5 - Compute the sum of power in each segment
        # Calculate the sum of power in each segment (in total k segments)
        self.power_vector=self.S_pow.reshape(-1,W).sum(axis=-1)
        POWER_VECTOR_SIZE=self.power_vector.shape[0]

        #Each row in f represents 10.77*10=107.7Hz,  so f[50] represents 5385HZ
        self.FREQ_ROW_WISE=self.FREQ_ROW_WISE*W

        #Step 6 : Normalization of power_spectrum
        self.power_vector_norm01=self.normalize_array((self.power_vector))

        # Feature 1: FV_LFP - low frequencies power features
        self.FV_LFP=np.array(self.power_vector_norm01[0:48])

    #Function to extract Linearity Degree Features
    def extract_FV_LDF(self):
        #Normalize  power_vector with sum(power_vector) to obtain  power_normal
        self.power_vector_normal=self.power_vector/np.sum(self.power_vector)

        #Accumulate the values of power_normal to obtain pow_cdf
        self.power_cdf=np.cumsum(self.power_vector_normal)

        #Feature 2:Correlation coefficient and q 
        #Calculate Correlation coefficient
        freq=np.linspace(0,self.power_cdf.size*self.FREQ_ROW_WISE,self.power_cdf.size)
        self.correlation_coefficient = np.corrcoef(self.power_cdf,freq)[0,1]

        #Calculating quadratic curve fitting coefficients q of pow_cdf
        #x_values = np.arange(0, 8+7/(power_cdf_s1_live.size-1),8/(power_cdf_s1_live.size-1))
        x_values=np.linspace(0,8,self.power_cdf.size)
        #150 points spaced at 107 hz (last point corresponds to 16K frequency) is mapped to equally spaced 150 points between 0 and 8
        #This help to scale down the plot which we are trying to do polyfit.

        #Plotting the curve with x=pow_cdf as specified in the paper

        #A polynomial q(x) of degree n = 2 with respective coefficients are given below as:
        #q(x) = q1x^2 + q2x + q3, 
        # where x = powcdf in the above equation. We use the quadratic
        # coefficient q1 in our features which is denoted by q for simplicity.
        #plt.plot(power_cdf,x_values,".")
        parameter_2 = np.polyfit( self.power_cdf,x_values, 2)
        self.q = parameter_2[0]
        #print("The q of the signal :",q)
        #Storing the ρ and q in an 
        self.FV_LDF = np.array([self.correlation_coefficient, self.q])

    #Function to extract High Power Frequency Features
    def extract_FV_HPF(self,OMEGA=0.6):
            peaks_idx, _ = find_peaks(self.FV_LFP, height=0)
    
            #Plotting the peaks of the signal
            #plt.plot(FV_LFP)
            #plt.plot(peaks_idx, FV_LFP[peaks_idx], "*")
            #plt.plot(np.zeros_like(FV_LFP), "--", color="green")
            #plt.show()
            
            # Obtain corresponding values of the peaks:
            peaks_val=self.FV_LFP[peaks_idx]
            
            #print("The number of peaks in the signal: ",peaks_idx.size)
            #print("The index of the peaks in the signal: ",peaks_idx)
            #print("The values of the peaks in the signal: ",peaks_val)
            
            
            if len(peaks_val>0):
                # 2. Compute the threshold of selecting peaks using omega:
                T_peak =OMEGA*max(peaks_val)

                #print("The threshold value of selecting the peak : ",T_peak)
                # 3. Remove peaks lower than T_peak (insignificant peaks):
                peaks_idx=peaks_idx[np.where(peaks_val >= T_peak)]
            else:
                T_peak= OMEGA
                peaks_idx=np.array([])
            
            #print("The index of the peaks that are above the T_peak: ",peaks_idx)
            
            # 4. Obtain the number of remaining peaks:
            N_peak = peaks_idx.size
            #print("The number of the peaks that are above the T_peak: ",N_peak)
            
            # 5. Compute the mean of the locations of remaining peaks:
            mu_peak = peaks_idx.mean()
            #print("The mean of the locations of remaining peaks: ",mu_peak)
            
            # 6. Compute the standard deviation of the locations of remaining peaks:
            sigma_peak = np.std(peaks_idx)
            #print("The standard deviation of the locations of remaining peaks: ",sigma_peak)
            
            # 7. Use a 6-order polynomial to fit FV_LFP and take first 32 estimatied values as P_est:
            #plt.plot(np.arange(FV_LFP.size),FV_LFP)
            parameter_6 = np.polyfit(np.arange(self.FV_LFP.size), self.FV_LFP, 6)
            #print(parameter_6)
            #Plotting the best fit 6-order polynomial
            #plot_poly_curve(np.arange(FV_LFP.size),parameter_6)
            
            value_est = np.polyval(parameter_6, np.arange(self.FV_LFP.size))
            P_est = np.array(value_est[0:32])
            #P_est=self.normalize_array(P_est)
            #print("The P_est of order 6 polynomial",P_est)
            HPF_array1=np.array([N_peak,mu_peak,sigma_peak])
            
            self.FV_HPF=np.concatenate([HPF_array1,P_est])
            #print("The Total number of High Frequency Power features:",self.FV_HPF.size)

    # Function to find LPC coefficient of order 12 using Librosa 
    def lpc_Librosa(self,order=12):
        self.lpc_librosa = librosa.lpc(y=self.signal_array, order=order)

    # Functions to find LPC coefficient of order 12 using Levinson Durbin algorithm
    def bac(self,x, p):
        # compute the biased autocorrelation for x up to lag p
        L = len(x)
        r = np.zeros(p+1)
        for m in range(0, p+1):
            for n in range(0, L-m):
                r[m] += x[n] * x[n+m]
            r[m] /= float(L)
        return r
        
    def ld(self,r, p):
        # solve the toeplitz system using the Levinson-Durbin algorithm
        g = r[1] / r[0]
        a = np.array([g])
        v = (1. - g * g) * r[0]
        for i in range(1, p):
            g = (r[i+1] - np.dot(a, r[1:i+1])) / v
            a = np.r_[ g,  a - g * a[i-1::-1] ]
            v *= 1. - g*g
        # return the coefficients of the A(z) filter
        return np.r_[1, -a[::-1]]
    def lpc_Levinson(self, p=12):
        # compute p LPC coefficients for a speech segment
        return self.ld(self.bac(self.signal_array, p), p)
    
    def extract_FV_LPCC(self,  order=12):
        '''
        Function: lpcc
        Summary: Computes the linear predictive cepstral compoents. Note: Returned values are in the frequency domain
        Examples: audiofile = AudioFile.open('file.wav',16000)
                frames = audiofile.frames(512,np.hamming)
                for frame in frames:
                    frame.lpcc()
                Note that we already preprocess in the Frame class the lpc conversion!
        Attributes:
            @param (seq):A sequence of lpc components. Need to be preprocessed by lpc()
            @param (err_term):Error term for lpc sequence. Returned by lpc()[1]
            @param (order) default=None: Return size of the array. Function returns order+1 length array. Default is len(seq)
        Returns: List with lpcc components with default length len(seq), otherwise length order +1
        '''
        self.lpc_levinson=self.lpc_Levinson()
        seq=self.lpc_levinson
        err_term=order - 1
        if order is None:
            order = len(seq) - 1
    
        lpcc_coeffs = [np.log(err_term), -seq[0]]
        for n in range(2, order + 1):
            # Use order + 1 as upper bound for the last iteration
            upbound = (order + 1 if n > order else n)
        
            lpcc_coef = -sum(i * lpcc_coeffs[i] * seq[n - i - 1] for i in range(1, upbound)) * 1. / upbound
            lpcc_coef -= seq[n - 1] if n <= len(seq) else 0
            lpcc_coeffs.append(lpcc_coef)
        self.FV_LPCC= np.array(lpcc_coeffs[1:13])
        self.FV_LPCC=self.normalize_array(self.FV_LPCC)
    def get_all_feature_vectors(self):
        self.extract_FV_LFP()
        self.extract_FV_LDF()
        self.extract_FV_HPF()
        self.extract_FV_LPCC()
        self.FV = np.concatenate((self.FV_LDF,self.FV_HPF,self.FV_LPCC,self.FV_LFP))
        return self.FV


if(__name__ == "__main__"):
    BASE_PATH = '/home/achama/AudioML/audio_data'
    
    sample_signal_filename=f'{BASE_PATH}/Live/liveaudio_Arun/vp01.wav'
    sample1 = Signal_Feature_Extraction(sample_signal_filename)
    print("hi sample1")
    #sample1.extract_FV_LFP()
    #print(sample1.FV_LFP)
    #sample1.plot_power_spectrum_db()
    #sample1.plot_spectrum(sample1.power_vector_norm01)
    #sample1.extract_FV_LDF()
    #print(sample1.FV_LDF)
    #sample1.plot_rho_q_LDF()
    #sample1.extract_FV_HPF()
    #print("FV_HPF: ",sample1.FV_HPF)
    #sample1.extract_FV_LPCC()
    #print("FV_LPCC: ",sample1.FV_LPCC)
    print(sample1.get_all_feature_vectors())