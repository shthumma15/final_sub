# app.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
from utils.segmentation_analyzer import SegmentationAnalyzer
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Define remote data file
REMOTE_DATA = 'Train_subset.csv'

# Initialize Backblaze connection
b2 = B2(endpoint=os.environ.get('B2_ENDPOINT'),
        key_id=os.environ.get('B2_KEYID'),
        secret_key=os.environ.get('B2_applicationKey'))

def get_data():
    """Retrieve data from Backblaze"""
    try:
        # Set bucket and retrieve DataFrame
        b2.set_bucket(os.environ.get('B2_BUCKETNAME'))
        df = b2.get_df(REMOTE_DATA)
        return df
    except Exception as e:
        st.error(f"An error occurred while retrieving data: {e}")
        return None

# Title of the application
st.title("Customer Segmentation Analysis")

# Load data and perform data cleaning
df = get_data()
if df is not None:
    train = df.drop(['ID', 'Var_1'], axis=1)
    data_cleaned = df.dropna(subset=['Age', 'Segmentation'])

    # Data Overview
    st.header("Data Overview")
    st.write("Number of Rows:", train.shape[0])
    st.write("Number of Columns:", train.shape[1])
    st.write("Data Types:", train.dtypes)
    st.write("Summary Statistics:", train.describe())

  # Data Visualization
st.header("Data Visualization")
selected_variables = st.multiselect('Select Variables for Visualization:', train.columns)
if selected_variables:
    for var in selected_variables:
        if train[var].dtype in ['int64', 'float64']:
            st.subheader(f'{var} Distribution')
            fig, ax = plt.subplots()
            sns.histplot(train[var], kde=True, ax=ax)
            st.pyplot(fig)
        elif train[var].dtype == 'object':
            st.subheader(f'{var} Count')
            fig, ax = plt.subplots()
            sns.countplot(train[var], ax=ax)
            st.pyplot(fig)


    # Filtering and Sorting
    st.header("Filtering and Sorting")
    filter_column = st.selectbox('Select a column to filter:', train.columns)
    filter_value = st.text_input('Enter the value to filter by:')
    filtered_data = train[train[filter_column] == filter_value]

    sort_column = st.selectbox('Select a column to sort:', train.columns)
    sort_ascending = st.checkbox('Sort in ascending order', value=True)
    sorted_data = filtered_data.sort_values(by=sort_column, ascending=sort_ascending)

    st.write("Filtered and Sorted Data:")
    st.dataframe(sorted_data)

    # Segmentation Analysis
    st.header("Segmentation Analysis")
    segment_column = st.selectbox('Select a column for segmentation analysis:', train.columns)
    segment_analyzer = SegmentationAnalyzer(train)
    segment_counts = segment_analyzer.count_segments(segment_column)
    st.write("Segment Counts:")
    st.write(segment_counts)

# Correlation Analysis
st.header("Correlation Analysis")

# Select only numeric columns for correlation analysis
numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns

if len(numeric_columns) > 1:
    correlation_matrix = train[numeric_columns].corr()
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.write("No numeric columns available for correlation analysis.")

    # Data table
    st.header("First 30 rows of the DataFrame:")
    st.dataframe(train.head(30))
