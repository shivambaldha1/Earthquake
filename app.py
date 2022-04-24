#============================ import the some library here======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import scipy
from statsmodels import robust
from scipy.stats import skew
from scipy.stats import kurtosis
from tqdm import tqdm
from scipy import signal
import lightgbm as lgb
import pickle
import joblib

#============================================= given the some title and image here ==========================================================================

st.title("Welcome To Earthquake Prediction Webapp")
st.write('Problem statement : Here we have to predict is the remaining time before the next laboratory earthquakes based on the seismic data.')
st.write('Remaining time meant by the remaining time between current earthquake and occurrence of next earthquake')
st.image('img.jpg' , caption = 'SOURCE - https://www.aessoil.com/the-truth-about-earthquake-predictions/')
st.write('You can enter the sample of the acoustic signal (continuous single) values to find the remaining time (in seconds) to the next Earthquake and the model is trained on the acoustic signal with the remaining time. ')

with st.expander("To download the sample file of an acoustic signal"):
     st.write(""" If you want to check the model, you can download the sample file of the acoustic signal here. https://rb.gy/kcx76l """)
     st.write(''' To explore the more test files https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/data?select=test ''')

#==================================================== take input as csv file =================================================================================

uploaded_file = st.file_uploader("Choose a .csv file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)
if uploaded_file is None:
	df = pd.read_csv('test.csv')
    #st.write('here this default file output')

	#st.write(df)


btn = st.button("Predict")



#======================================================== final function which predict the output ============================================================

def final_fuction_1(x):


    '''This function takes raw data as acoustic signal value with the 150k data points as input and returns the remaining time of the next earthquake.'''

    features = {}
    image_pixel_feature = []
    pixel_intensity = []
    features['average_value'] = x.values.mean()
    features['std_value'] = x.values.std()
    features['max_value'] = x.values.max()
    features['min_value'] = x.values.min()
    features['kurtosis']  = x.kurtosis()
    features['skew'] = x.skew()
    features['MAD_value'] = robust.mad(x.values)
    #take absolute values
    features['abs_average_value'] = np.abs(x.values).mean()
    features['abs_std_value'] = np.abs(x.values).std()
    features['abs_max_value'] = np.abs(x.values).max()
    features['abs_median'] = np.median(np.abs(x.values))
    #take top 3 quantile
    features['99_quantile'] = np.quantile(x.values, 0.99)
    features['95_quantile'] = np.quantile(x.values, 0.95)
    features['90_quantile'] = np.quantile(x.values, 0.90)
    #take low 3 quantile
    features['01_quantile'] = np.quantile(x.values, 0.01)
    features['05_quantile'] = np.quantile(x.values, 0.05)
    features['10_quantile'] = np.quantile(x.values, 0.10)
    #take interquantile range
    features['interquantile'] = np.quantile(x.values, 0.27) - np.quantile(x.values, 0.25)
    #mean and std of last and first 5k datapoints
    features['mean_of_first_50000'] = x[:50000].values.mean()
    features['mean_of_last_50000'] = x[:-50000].values.mean()
    
    for i in [10,50,100,150,200]:
        rolling_mean = x.rolling(i).mean().shift().dropna().values
        rolling_std = x.rolling(i).std().shift().dropna().values

        features['avg_of_'+ str(i)+'_rolling_mean'] = rolling_mean.mean()
        features['std_of_'+ str(i)+'_rolling_mean'] = rolling_mean.std()
        features['1_qua_of_'+ str(i)+'_rolling_mean'] = np.quantile(rolling_mean, 0.01)
        features['5_qua_of_'+ str(i)+'_rolling_mean'] = np.quantile(rolling_mean, 0.05)
        features['90_qua_of_'+ str(i)+'_rolling_mean'] = np.quantile(rolling_mean, 0.90)
        features['95_qua_of_'+ str(i)+'_rolling_mean'] = np.quantile(rolling_mean, 0.95)
        features['skew_of'+ str(i)+'_rolling_mean'] = skew(rolling_mean)
        features['kurtosis_of'+ str(i)+'_rolling_mean'] = kurtosis(rolling_mean)
        features['min_of'+ str(i)+'_rolling_mean'] = rolling_mean.min()
        features['max_of'+ str(i)+'_rolling_mean'] = rolling_mean.max()
        features['interquantile_of' + str(i)+'_rolling_mean'] = np.quantile(rolling_mean, 0.27) - np.quantile(rolling_mean, 0.25)

        features['avg_of_'+ str(i)+'_rolling_std'] = rolling_std.mean()
        features['std_of_'+ str(i)+'_rolling_std'] = rolling_std.std()
        features['1_qua_of_'+ str(i)+'_rolling_std'] = np.quantile(rolling_std, 0.01)
        features['5_qua_of_'+ str(i)+'_rolling_std'] = np.quantile(rolling_std, 0.05)
        features['90_qua_of_'+ str(i)+'_rolling_std'] = np.quantile(rolling_std, 0.90)
        features['95_qua_of_'+ str(i)+'_rolling_std'] = np.quantile(rolling_std, 0.95)
        features['skew_of'+ str(i)+'_rolling_std'] = skew(rolling_std)
        features['kurtosis_of'+ str(i)+'_rolling_std'] = kurtosis(rolling_std)
        features['min_of'+ str(i)+'_rolling_std'] = rolling_std.min()
        features['max_of'+ str(i)+'_rolling_std'] = rolling_std.max()
        features['interquantile_of' + str(i)+'rolling_std'] = np.quantile(rolling_std, 0.27) - np.quantile(rolling_std, 0.25)
    feature_df = pd.DataFrame(features , index=[0])
        
    frequencies, times, spectrogram = signal.spectrogram(x)
    pixel_intensity.append(spectrogram.flatten()[:500])
    for i in range(500):
        image_pixel_feature.append('PI' + str(i))
    img_df = pd.DataFrame(pixel_intensity , columns = image_pixel_feature)
    result = pd.concat([feature_df , img_df], axis=1, join="inner")
    #print(result)
    
    model = joblib.load('model.pkl')
    predict = model.predict(result)
    return predict[0]

#================================================define the button here=============================================================================================

if btn:

	a = final_fuction_1(df)
	a = np.round(a , 5)
	#st.write("The Remaining time Of the Next Earthquake Is")
	#st.subheader('{} Sec'.format(a))
	st.metric(label="The Remaining time of the Next Earthquake is", value= ('{} Sec'.format(a)))




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ end of the app thank you   +++++++++++++++++++++++++++++===================================================================	

