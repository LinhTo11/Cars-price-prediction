import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# @st.cache #If we execute the def load_data it will be cached and won't run again
def load_data():
    df = pd.read_csv("C:/Users/LinhTo/Desktop/BU/Class - Teacher/BA 780/vehicle_cleaned.csv")
    data = pd.read_csv("C:/Users/LinhTo/Desktop/BU/Class - Teacher/BA 780/vehicle_ml.csv")
    return df, data

df, data = load_data()

def show_explore_page(data):
    st.title('Used Car Price Prediction')

    st.write(
        """
        ### Using ML to predict used car price as a consumer
        """
    )


    # scatterplot of average price by manufacturer
    st.bar_chart(df['manufacturer'])