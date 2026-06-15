import numpy as np
import pickle
import pandas as pd
import streamlit as st 

# from PIL import Image


pickle_in = open("mlr.pkl","rb")
mlr=pickle.load(pickle_in)


def welcome():
    return "Welcome All"

def predict_authentication(hp,vol,sp,wt):   
    prediction=mlr.predict([[hp,vol,sp,wt]])
    return prediction



def main():
    st.title("Car Mileage Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Car Mileage Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    hp = st.number_input("HP",min_value=0, step=1)
    vol = st.number_input("VOL",min_value=0, step=1)
    sp = st.number_input("SP",min_value=0, step=1)
    wt = st.number_input("WT",min_value=0, step=1)
    
    result=""
    if st.button("Predict"):
        result=predict_authentication(hp,vol,sp,wt)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
