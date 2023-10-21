import streamlit as st
import pickle
import numpy as np


#import the model
st.title("Welcome!")
st.title("Predict Your Laptop Price")
pipe =  pickle.load(open('pipe.pkl', 'rb'))
clf= pickle.load(open('df.pkl','rb'))

# brand
company = st.selectbox('Brand',clf['Company'].unique())

# type of laptop
type = st.selectbox('Type',clf['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',clf['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',clf['Gpu brand'].unique())

os = st.selectbox('OS',clf['os'].unique())


if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = np.array(query, dtype=object)

    query = query.reshape(1, 12)

    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

