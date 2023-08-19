import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sqlalchemy import create_engine

st.title("Automated EDA Tool")

data_source = st.selectbox("Select data source", ["Upload CSV/ Txt/ Excel", "SQL Database"])
data = None

if data_source == "Upload CSV/ Txt/ Excel":
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx","txt"],)
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")
        st.write(f"We have {data.shape[0]} rows and {data.shape[1]} columns.")


elif data_source == "SQL Database":
    db_url = st.text_input("Enter the database URL (e.g., sqlite:///database.db)")
    if db_url:
        engine = create_engine(db_url)
        available_tables = engine.table_names()  # Get available table names
        table_name = st.selectbox("Select a table", available_tables)
        if table_name:
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql(query, engine)
            st.write("Data loaded successfully!")
            st.write(f"We have {data.shape[0]} rows and {data.shape[1]} columns.")
            

if data is not None:
    st.subheader("1- Raw Data")
    
    num_head_rows = st.number_input("Number of rows to display (head)", min_value=1, max_value=len(data), value=5)
    st.write("Head of the Data:")
    st.write(data.head(num_head_rows))

    num_tail_rows = st.number_input("Number of rows to display (tail)", min_value=1, max_value=len(data), value=5)
    st.write("Tail of the Data:")
    st.write(data.tail(num_tail_rows))


    st.subheader("2- Data Pre-processing")
    
    st.write("2.1- Handling Missing Values:")
    columns_with_missing = data.columns[data.isnull().any()]

    # Handle missing values based on column type
    for col in columns_with_missing:
        num_missing = data[col].isnull().sum()
        st.write(f"Number of missing values in '{col}': {num_missing}")
    
        if data[col].dtype == "object":
            # Handle missing values in categorical columns with mode
            mode_value = data[col].mode()[0] # [0] to get value
            data[col].fillna(mode_value, inplace=True)
        else:
            # numeric columns with median if found outliears
            if data[col].skew() > 1:
                median_value = data[col].median()
                data[col].fillna(median_value, inplace=True)
            # numeric columns with mean if not found outliers
            else:
                mean_value = data[col].mean()
                data[col].fillna(mean_value, inplace=True)

    st.write("Missing values handled:")
    st.write(data.isnull().sum())
    st.write(data.head())
    

    st.subheader("3- Data Visualization")

    # Select column to visualize
    columns = data.columns.tolist()
    selected_column = st.selectbox("Select a column", columns)

    # data type of the selected column
    selected_column_type = data[selected_column].dtype

    # Visualization options based on column types
    visualization_options = []
    if np.issubdtype(selected_column_type, np.number):
        visualization_options.extend(["Line Plot", "Scatter Plot", "Histogram", "Box Plot"])
    elif np.issubdtype(selected_column_type, np.datetime64):
        visualization_options.append("Line Plot")
    elif np.issubdtype(selected_column_type, object):
        visualization_options.extend(["Bar Plot", "Pie Chart", "Count Plot"])
    else:
        st.write("Visualization not supported for this column type.")

    # Select visualization type
    visualization_type = st.selectbox("Select a visualization type", visualization_options)

    # Perform the selected visualization
    if visualization_type == "Line Plot":
        # Perform line plot based on column type
        if np.issubdtype(selected_column_type, np.datetime64):
            # Line plot for time series columns
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(x=data.index, y=selected_column, data=data, ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel(selected_column)
            ax.set_title(f"Line Plot of {selected_column}")
            st.pyplot(fig)
        elif np.issubdtype(selected_column_type, np.number):
            # Line plot for numerical columns
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(x=data.index, y=selected_column, data=data, ax=ax)
            ax.set_xlabel("Data")
            ax.set_ylabel(selected_column)
            ax.set_title(f"Line Plot of {selected_column}")
            st.pyplot(fig)
        else:
            st.write("Line plot not supported for this column type.")
    
    elif visualization_type == "Scatter Plot":
        # Perform scatter plot for numerical columns
        if np.issubdtype(selected_column_type, np.number):
            x_column = st.selectbox("Select X-axis column", columns)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=x_column, y=selected_column, data=data, ax=ax)
            ax.set_xlabel(x_column) # x0axis
            ax.set_ylabel(selected_column) # y-axis
            ax.set_title(f"Scatter Plot of {selected_column} vs. {x_column}")
            st.pyplot(fig)
        else:
            st.write("Scatter plot not supported for this column type.")
    
    elif visualization_type == "Histogram":
        # Perform histogram for numerical columns
        if np.issubdtype(selected_column_type, np.number):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data[selected_column], bins=20, ax=ax)
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of {selected_column}")
            st.pyplot(fig)
        else:
            st.write("Histogram not supported for this column type.")

    elif visualization_type == "Box Plot":
        # Perform box plot for numerical columns
        if np.issubdtype(selected_column_type, np.number):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(y=selected_column, data=data, ax=ax)
            ax.set_ylabel(selected_column)
            ax.set_title(f"Box Plot of {selected_column}")
            st.pyplot(fig)
        else:
            st.write("Box plot not supported for this column type.")

    elif visualization_type == "Bar Plot":
        # Perform bar plot for categorical columns
        if np.issubdtype(selected_column_type, object):
            value_counts = data[selected_column].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            value_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Count")
            ax.set_title(f"Bar Plot of {selected_column}")
            ax.legend([selected_column]) 
            st.pyplot(fig)
        else:
            st.write("Bar plot not supported for this column type.")

    elif visualization_type == "Pie Chart":
        # Perform pie chart for categorical columns
        if np.issubdtype(selected_column_type, object):
            value_counts = data[selected_column].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140, rotatelabels=True)
            ax.axis('equal')
            ax.set_title(f"Pie Chart of {selected_column}")
            st.pyplot(fig)
        else:
            st.write("Pie chart not supported for this column type.")

    elif visualization_type == "Count Plot":
        # Perform count plot for categorical columns
        if np.issubdtype(selected_column_type, object):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x=selected_column, data=data, ax=ax)
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Count")
            ax.set_title(f"Count Plot of {selected_column}")
            ax.legend([selected_column]) 
            st.pyplot(fig)
        else:
            st.write("Count plot not supported for this column type.")
            
            
    # making after visualization for not convert categorical features for numerical before plotting
    st.write("2.2- Encoding Categorical Features:")

    # categorical columns
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    # Ask user for encoding method
    encoding_method = st.selectbox("Select a categorical encoding method", ["Label Encoding", "One-Hot Encoding"])

    if encoding_method == "Label Encoding":
        # Perform Label Encoding
        encoder = LabelEncoder()
        for col in categorical_columns:
            data[col] = encoder.fit_transform(data[col])
        st.write("Encoded Data using Label Encoding:")
        st.write(data.head())

    elif encoding_method == "One-Hot Encoding":
        # Perform One-Hot Encoding
        data_encoded = pd.get_dummies(data, columns=categorical_columns)
        st.write("Encoded Data using One-Hot Encoding:")
        st.write(data_encoded.head())
        
        
    st.write("2.3- Scaling Numerical Features")

    # Ask user for scaling type
    scaling_type = st.selectbox("Select a scaling type", ["Standard Scaling", "Min-Max Scaling"])

    if scaling_type == "Standard Scaling":
        st.write("Applying Standard Scaling:")
        numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        st.write(data.head())

    elif scaling_type == "Min-Max Scaling":
        st.write("Applying Min-Max Scaling:")
        numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        scaler = MinMaxScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        st.write(data.head())