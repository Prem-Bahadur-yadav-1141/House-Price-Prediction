import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://www.bing.com/th/id/OGC.a0d928b00ef43581460c12faf9c1ff22?o=7&pid=1.7&rm=3&rurl=https%3a%2f%2fmedia0.giphy.com%2fmedia%2fKeEPTWaZc96RKqJnGs%2fgiphy.gif&ehk=GpF%2fTLp10K2s3fUwqgsh6HWZvNc8iGKhHmys6%2fD2nVc%3d')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

final_X = X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)

st.sidebar.title('select house features:')
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/b/b7/House-animated.gif')
all_value = []
for i in final_X:
  result = st.sidebar.slider(f'select {i} value')
  all_value.append(result)

st.write(all_value)

