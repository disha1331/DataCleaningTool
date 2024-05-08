import streamlit as st
import pygwalker as pyg
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from PIL import Image
import time
from pandas import json_normalize, read_json
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Lasso
from datetime import date
from traitlets import Int
import json

def config():
    st.set_page_config(
        page_icon="📊",  # You can replace this with the path to your custom icon
        page_title="SCAPT"
    )

# Call the config function to set up Streamlit app configuration
config()



class FileApp:
    def __init__(self):
        self.df = None
    def load_file(self, file_type):
        st.markdown("# Connect to a File")
        uploaded_file = st.file_uploader(f"Upload a {file_type} file", type=[file_type])

        if uploaded_file is not None:
            if file_type == "csv":
                self.df = pd.read_csv(uploaded_file)
                

            elif file_type == "xlsx":
                self.df = pd.read_excel(uploaded_file)
       

            elif file_type == "json":
                self.df = pd.read_json(uploaded_file)
          
            elif file_type == "txt":
                file_contents = uploaded_file.getvalue().decode("utf-8")
                
                # Split the file contents into lines
                lines = file_contents.split("\n")
                
                # Extract column names from the first line
                columns = lines[0].split()
                
                # Check if the number of columns in the data matches the number of columns in the column names
                num_columns = len(columns)
                
                # Create a list to store the data
                data = []
                
                # Process each subsequent line
                for line in lines[1:]:
                    # Split each line into values
                    values = line.split()  # You may need to specify a delimiter here if your file is not space-separated
                    # Check if the number of values matches the number of columns
                    if len(values) == num_columns:
                        # Append the values to the data list
                        data.append(values)
                    
                
                # Convert the data list into a DataFrame
                self.df = pd.DataFrame(data, columns=columns)
                
            
                
            elif file_type == "xml":
                self.df = pd.read_xml(uploaded_file)
               
            elif file_type == "parquet":
                self.df = pd.read_parquet(uploaded_file)
              
            else:
                st.error("File type not supported")
                return

            st.data_editor(self.df)
    def auto_configure_field_types(self):
        if self.df is not None:
            if st.sidebar.checkbox("Auto Configure Tool"):
                st.title("AutoConfigure Field Types")
                all_columns = self.df.columns.tolist()  # type: ignore
                selected_columns = st.multiselect("Select fields to autoconfigure:", all_columns)

            
                for col in selected_columns:
                    # Check if the column can be converted to numeric
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='raise')
                        self.df[col] = self.df[col].astype('Int64')   # type: ignore

                    except (ValueError, TypeError):
                        # If conversion to numeric fails, keep the original data type
                        pass

                st.success("Auto configuration complete")
    def change_data_type(self):
                # Show initial data types
        if self.df is not None:
            if st.sidebar.checkbox('Change Data Type'):
                st.title("Change Data Types")
                st.write("Initial Data Types:")
                st.write(self.df.dtypes)
            
                columns = self.df.columns.tolist() # type: ignore
                selected_columns = st.multiselect('Select columns:', columns)

                data_types = {
                    'int64': 'Integer',
                    'float64': 'Float',
                    'object': 'String', 
                    'datetime64': 'Datetime',
                    'bool': 'Boolean',
                    'category': 'Category'
                }

                data_type = st.selectbox('Select data type:', options=list(data_types.keys()))

                if selected_columns:
                        for col in selected_columns:
                            try:
                                if data_type == 'int64':
                                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')  # type: ignore
                                elif data_type == 'float64':
                                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                                elif data_type == 'object':
                                    self.df[col] = self.df[col].astype('str') # type: ignore
                                elif data_type == 'datetime64':
                                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                                elif data_type == 'bool':
                                    self.df[col] = self.df[col].astype('bool')  # type: ignore
                                elif data_type == 'category':
                                    self.df[col] = self.df[col].astype('category')  # type: ignore
                        
                                st.success(f"Changed data type of '{col}' to '{data_type}'")
                                st.write("Updated Data Types:")  
                                st.write(self.df.dtypes)
                                st.write(self.df)
                            except Exception as e:
                                st.error(f"Error changing data type of '{col}': {e}")
        # Show data types after change
                       
   
    def handle_duplicates_v2(self):
    # Your existing handle_duplicates function

        if self.df is not None:
            if st.sidebar.checkbox("Handle  Duplicates Tool"):
                st.title('Handle Duplicates')

                duplicate_action = st.selectbox('Select action:', ('Show duplicates', 'Remove duplicates'))

                if duplicate_action == 'Show duplicates':
                    duplicates = self.df[self.df.duplicated()]
                    st.write("### Duplicates Found")
                    st.write(duplicates)
                
                elif duplicate_action == 'Remove duplicates':
                    self.df.drop_duplicates(inplace=True)
                    st.success("Duplicates removed")
                    st.write(self.df)
    def data_cleansing(self):
        if self.df is not None:
            if st.sidebar.checkbox("Data Cleansing Tool"):
                st.title('Data Cleansing')
                remove_null_rows = st.checkbox("Remove Null Rows")
                remove_null_columns = st.checkbox("Remove Null Columns")
                replace_nulls_with_blanks = st.checkbox("Replace Nulls with Blanks (String Fields)")
                replace_nulls_with_zero = st.checkbox("Replace Nulls with 0 (Numeric Fields)")
                remove_whitespace = st.checkbox("Remove Leading and Trailing Whitespace")
                remove_duplicate_whitespace = st.checkbox("Remove Tabs, Line Breaks, and Duplicate Whitespace")
                remove_all_whitespace = st.checkbox("Remove All Whitespace")
                remove_letters = st.checkbox("Remove All Letters")
                remove_numbers = st.checkbox("Remove All Numbers")
                remove_punctuation = st.checkbox("Remove Punctuation")
                modify_case = st.selectbox("Modify Case", ["Upper Case", "Lower Case", "Title Case"])

        # Data cleansing logic
                if remove_null_rows:
                    self.df.dropna(inplace=True, how='all')
                if remove_null_columns:
                    self.df.dropna(inplace=True, axis=1, how='all') # type: ignore
                if replace_nulls_with_blanks:
                    self.df.fillna('', inplace=True)
                if replace_nulls_with_zero:
                    self.df.fillna(0, inplace=True)
                if remove_whitespace:
                    self.df = self.df.apply(lambda x: x.strip() if isinstance(x, str) else x)
                if remove_duplicate_whitespace:
                    self.df = self.df.replace(r'\s+', ' ', regex=True)
                if remove_all_whitespace:
                    self.df = self.df.apply(lambda x: ''.join(x.split()) if isinstance(x, str) else x)
                if remove_letters:
                    self.df = self.df.replace(r'[a-zA-Z]', '', regex=True)
                if remove_numbers:
                    self.df = self.df.replace(r'[0-9]', '', regex=True)
                if remove_punctuation:
                    punctuation_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                    self.df = self.df.apply(lambda x: x.translate(str.maketrans('', '', punctuation_chars)) if isinstance(x, str) else x)
                if modify_case == "Upper Case":
                    self.df = self.df.apply(lambda x: x.upper() if isinstance(x, str) else x)
                elif modify_case == "Lower Case":
                    self.df = self.df.apply(lambda x: x.lower() if isinstance(x, str) else x)
                elif modify_case == "Title Case":
                    self.df = self.df.apply(lambda x: x.title() if isinstance(x, str) else x)

        # Display the cleansed data
                st.write("Cleansed Data:")
                st.write(self.df)
        else:
            st.warning("No data available. Please load a dataset first.")

    def create_samples(self):
        if self.df is not None:
            if st.sidebar.checkbox("Create Samples Tool"):
                st.title('Create Samples')
                estimation_percent = st.slider("Estimation sample percent", 1, 99, 50)
                validation_percent = st.slider("Validation sample percent", 1, 99, 25)
                random_seed = st.slider("Random seed", 1, 1000, 1)

            
                if not 1 <= estimation_percent <= 99:
                    st.error("Estimation sample percent must be between 1 and 99.")
                    return
                if not 1 <= validation_percent <= 99:
                    st.error("Validation sample percent must be between 1 and 99.")
                    return
                if not 1 <= random_seed <= 1000:
                    st.error("Random seed must be between 1 and 1000.")
                    return

                total_percent = estimation_percent + validation_percent
                if total_percent > 100:
                    st.error("The total of estimation and validation sample percent cannot exceed 100.")
                    return

                # Set random seed for reproducibility
                np.random.seed(random_seed)

                # Shuffle the data
                shuffled_data = self.df.sample(frac=1).reset_index(drop=True)

                # Calculate sample sizes
                total_records = len(self.df)
                estimation_size = int(total_records * (estimation_percent / 100))
                validation_size = int(total_records * (validation_percent / 100))

                # Split the data into estimation and validation samples
                estimation_sample = shuffled_data.iloc[:estimation_size]
                validation_sample = shuffled_data.iloc[estimation_size:estimation_size + validation_size]
                holdout_sample = shuffled_data.iloc[estimation_size + validation_size:]

                # Display the samples
                st.write("Estimation Sample:")
                st.write(estimation_sample)
                st.write("Validation Sample:")
                st.write(validation_sample)
                st.write("Holdout Sample:")
                st.write(holdout_sample)
        else:
            st.warning("No data available. Please load a dataset first.")

    def filter_data(self):
        if self.df is not None:
            if st.sidebar.checkbox("Filter data Tool"):
                st.title('Filter Data')
                select_column1 = st.selectbox("Select First Column", self.df.columns, key="select_column1")
                operator1 = st.selectbox("Select Operator", ["==", "!=", ">", ">=", "<", "<=", "Is null", "Is not null", "Is empty", "Is not empty"], key="operator1")
                value1 = st.text_input("Enter Value", key="value1")

                filter_logic = st.radio("Filter Logic", ["OR", "AND"], key="filter_logic")

                if filter_logic == "AND":
                    select_column2 = st.selectbox("Select Second Column", self.df.columns, key="select_column2")
                    operator2 = st.selectbox("Select Operator", ["==", "!=", ">", ">=", "<", "<=", "Is null", "Is not null", "Is empty", "Is not empty"], key="operator2")
                    value2 = st.text_input("Enter Value", key="value2")

            
                if operator1 in ["Is null", "Is not null"]:
                    if operator1 == "Is null":
                        filtered_data = self.df[self.df[select_column1].isnull()] # type: ignore
                    else:
                        filtered_data = self.df[~self.df[select_column1].isnull()] # type: ignore
                elif operator1 in ["Is empty", "Is not empty"]:
                    if operator1 == "Is empty":
                        filtered_data = self.df[self.df[select_column1].astype(str).str.strip() == ""] # type: ignore
                    else:
                        filtered_data = self.df[self.df[select_column1].astype(str).str.strip() != ""] # type: ignore
                else:
                    try:
                        value1 = int(value1)
                    except ValueError:
                        try:
                            value1 = float(value1)
                        except ValueError:
                            pass  # Value remains as string if conversion to int or float fails

                    if operator1 == "==":
                        filtered_data = self.df[self.df[select_column1] == value1] # type: ignore
                    elif operator1 == "!=":
                        filtered_data = self.df[self.df[select_column1] != value1] # type: ignore
                    elif operator1 == ">":
                        filtered_data = self.df[self.df[select_column1] > value1] # type: ignore
                    elif operator1 == ">=":
                        filtered_data = self.df[self.df[select_column1] >= value1] # type: ignore
                    elif operator1 == "<":
                        filtered_data = self.df[self.df[select_column1] < value1] # type: ignore
                    elif operator1 == "<=":
                        filtered_data = self.df[self.df[select_column1] <= value1] # type: ignore

                if filter_logic == "AND":
                    if operator2 in ["Is null", "Is not null"]:
                        if operator2 == "Is null":
                            filtered_data = filtered_data[filtered_data[select_column2].isnull()] # type: ignore
                        else:
                            filtered_data = filtered_data[~filtered_data[select_column2].isnull()] # type: ignore
                    elif operator2 in ["Is empty", "Is not empty"]:
                        if operator2 == "Is empty":
                            filtered_data = filtered_data[filtered_data[select_column2].astype(str).str.strip() == ""] # type: ignore
                        else:
                            filtered_data = filtered_data[filtered_data[select_column2].astype(str).str.strip() != ""] # type: ignore
                    else:
                        try:
                            value2 = int(value2)
                        except ValueError:
                            try:
                                value2 = float(value2)
                            except ValueError:
                                pass  # 
                        if operator2 == "==":
                            filtered_data = filtered_data[filtered_data[select_column2] == value2] # type: ignore
                        elif operator2 == "!=":
                            filtered_data = filtered_data[filtered_data[select_column2] != value2] # type: ignore
                        elif operator2 == ">":
                            filtered_data = filtered_data[filtered_data[select_column2] > value2] # type: ignore
                        elif operator2 == ">=":
                            filtered_data = filtered_data[filtered_data[select_column2] >= value2] # type: ignore
                        elif operator2 == "<":
                            filtered_data = filtered_data[filtered_data[select_column2] < value2] # type: ignore
                        elif operator2 == "<=":
                            filtered_data = filtered_data[filtered_data[select_column2] <= value2] # type: ignore

                st.write("Filtered Data:")
                st.write(filtered_data)
        else:
            st.warning("No data available. Please load a dataset first.") 
    
    def imputation(self):
        if self.df is not None:
            if st.sidebar.checkbox("Imputation Tool"):
                st.title('Imputation')
            
                fields_to_impute = st.multiselect("Fields to impute", self.df.select_dtypes(include=['number']).columns)  # type: ignore
            
                imputation_method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Linear Interpolation"])
            
                include_imputed_indicator = st.checkbox("Include imputed value indicator field")
                output_as_separate_field = st.checkbox("Output imputed values as a separate field")

            
                for field in fields_to_impute:
                    if imputation_method == "Mean":
                        replace_value = self.df[field].mean() # type: ignore
                    elif imputation_method == "Median":
                        replace_value = self.df[field].median() # type: ignore
                    elif imputation_method == "Mode":
                        replace_value = self.df[field].mode()[0] # type: ignore
                    elif imputation_method == "Forward Fill":
                        self.df[field + "_ImputedValue"] = self.df[field].ffill() # type: ignore
                    elif imputation_method == "Backward Fill":
                        self.df[field + "_ImputedValue"] = self.df[field].bfill() # type: ignore
                    elif imputation_method == "Linear Interpolation":
                        self.df[field + "_ImputedValue"] = self.df[field].interpolate(method='linear') # type: ignore

                    if imputation_method not in ["Forward Fill", "Backward Fill", "Linear Interpolation"]:
                        self.df[field + "_ImputedValue"] = self.df[field].fillna(replace_value) # type: ignore

                    if include_imputed_indicator:
                        self.df[field + "_Indicator"] = self.df[field].isnull() # type: ignore

                if output_as_separate_field:
                    st.write("Imputed Data:")
                    st.write(self.df)
                else:
                    st.write("Original Data with Imputed Values:")
                    st.write(self.df.drop(columns=[field for field in fields_to_impute]))
        else:
            st.warning("No data available. Please load a dataset first.")
    def perform_binning(self):
        if self.df is not None:
            if st.sidebar.checkbox("Binning Tool"):
                st.title('Multi Field Binning')

                numeric_columns = self.df.select_dtypes(include=['number']).columns # type: ignore
                fields_for_binning = st.multiselect("Select fields for binning", numeric_columns)

                binning_method = st.selectbox("Binning Method", ["Equal Records", "Equal Intervals"])
                num_tiles = st.number_input("Number of Tiles", min_value=1, max_value=1000, value=10)

            
                for field in fields_for_binning:
                    if binning_method == "Equal Records":
                        bins = pd.qcut(self.df[field], q=int(num_tiles), labels=False, duplicates='drop')  # type: ignore
                    elif binning_method == "Equal Intervals":
                        bins = pd.qcut(self.df[field], q=int(num_tiles), labels=False) # type: ignore
                
                    self.df[field + "_Tile_Num"] = bins

                    st.write("Output Data:")
                    st.write(self.df)
        else:
            st.warning("No data available. Please load a dataset first.")
    def perform_sampling(self):
        if self.df is not None:
            if st.sidebar.checkbox("Sampling Tool"):
                st.title('Sampling')
        
                sample_methods = ["First N rows", "Last N rows", "Skip 1st N rows", 
                          "1 of every N rows", "1 in N chance to include each row",
                          "First N% of rows"]
                selected_sample_method = st.selectbox("Sample Method", sample_methods)

                if selected_sample_method in ["First N rows", "Last N rows", "Skip 1st N rows", 
                                      "1 of every N rows", "First N% of rows"]:
                    n_value = int(st.number_input("N", min_value=1, value=1))
        
                if selected_sample_method == "1 in N chance to include each row":
                    n_value = int(st.number_input("N", min_value=1, value=2))

                if selected_sample_method == "First N% of rows":
                    n_percent = st.number_input("N%", min_value=1, max_value=100, value=10)

                    group_by_columns = st.multiselect("Group by column (optional)", self.df.columns)

            
                if selected_sample_method == "First N rows":
                    sampled_data = self.df.head(n_value)
                elif selected_sample_method == "Last N rows":
                    sampled_data = self.df.tail(n_value)
                elif selected_sample_method == "Skip 1st N rows":
                    sampled_data = self.df.iloc[n_value:]
                elif selected_sample_method == "1 of every N rows":
                    sampled_data = self.df.iloc[::n_value]
                elif selected_sample_method == "1 in N chance to include each row":
                    sampled_data = self.df.sample(frac=1/n_value)
                elif selected_sample_method == "First N% of rows":
                    n_rows = int(len(self.df) * n_percent / 100)
                    sampled_data = self.df.head(n_rows)

                st.write("Sampled Data:")
                st.write(sampled_data)
        else:
            st.warning("No data available. Please load a dataset first.")

    def unique_tool(self):
        if self.df is not None:
            if st.sidebar.checkbox("Unique Tool"):
                st.title("Unique Tool")
                st.subheader("Configure the Tool")
                columns = st.multiselect("Select Columns", self.df.columns)

                if columns:
                # Group by the selected columns and get the first occurrence for unique records
                    unique_records = self.df.drop_duplicates(subset=columns).reset_index(drop=True) # type: ignore
                
                # Get all duplicate records
                    duplicates = self.df[self.df.duplicated(subset=columns, keep='first')] # type: ignore

                    st.write("Unique Records:")
                    st.dataframe(unique_records)
    
                    st.write("Duplicate Records:")
                    st.dataframe(duplicates)
            


    def scaling_normalization(self):
        if self.df is not None:
            if st.sidebar.checkbox("Scaling & Normalization Tool"):
                st.title('Scaling and Normalization')
                numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist() # type: ignore
                selected_numeric_columns = st.multiselect('Select numeric columns for scaling:', numeric_columns)

                scaling_options = ['Min-Max Scaling', 'Standard Scaling']
                selected_scaling = st.selectbox('Select scaling technique:', scaling_options)

                if selected_numeric_columns:
                    if selected_scaling == 'Min-Max Scaling':
                        scaler = MinMaxScaler()
                        self.df[selected_numeric_columns] = scaler.fit_transform(self.df[selected_numeric_columns]) # type: ignore
                        st.success("Selected columns scaled using Min-Max Scaling")
                    elif selected_scaling == 'Standard Scaling':
                        scaler = StandardScaler()
                        self.df[selected_numeric_columns] = scaler.fit_transform(self.df[selected_numeric_columns]) # type: ignore
                        st.success("Selected columns scaled using Standard Scaling")

                    st.write(self.df)
                else:
                    st.warning("Please select at least one column.")



    def encoding_categorical_columns(self):
        # Your existing encoding_categorical_columns function
        if self.df is not None:
            if st.sidebar.checkbox("Encoding Categorical Columns Tool"):
                st.title('Encoding categorical columns')
                categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist() # type: ignore
                selected_categorical_columns = st.multiselect('Select categorical columns to encode:', categorical_columns)

            
                if selected_categorical_columns:
                    encoder = OneHotEncoder(sparse=False, drop='first')
                    encoded_cols = pd.DataFrame(encoder.fit_transform(self.df[selected_categorical_columns])) # type: ignore

                    # Custom column names for encoded columns
                    custom_columns = []
                    for col in selected_categorical_columns:
                        unique_categories = self.df[col].unique() # type: ignore
                        for category in unique_categories[1:]:  # Start from the second category
                            custom_columns.append(f"{col}_{category}")

                    encoded_cols.columns = custom_columns
                    # Drop original categorical columns and concatenate encoded ones
                    encode_cols = pd.concat([self.df.drop(selected_categorical_columns, axis=1), encoded_cols], axis=1)
                    st.success("Selected categorical columns encoded with custom names")
                    st.write(encode_cols)

    def remove_irrelevant_columns(self):
        # Your existing remove_irrelevant_columns function
         if self.df is not None:
            if st.sidebar.checkbox("Remove Irrelevent Column Tool"):
                st.title('Remove Irrelevant Columns')
                columns = self.df.columns.tolist() # type: ignore
                irrelevant_columns = st.multiselect('Select columns to remove:', columns)

           
                if irrelevant_columns:
                    self.df.drop(irrelevant_columns, axis=1, inplace=True)
                    st.success("Selected columns removed")
                    st.write(self.df)
                    
    def data_transformation(self):
        try:
            if self.df is not None:
                if st.sidebar.checkbox("Data Transform Tool"):
                    st.title("Data Transformation")

                # Merge columns
                    merge_cols = st.multiselect("Select columns to merge", self.df.columns)
                    merge_delim = st.text_input("Merge delimiter")

                    if merge_cols:
                        self.df[f"merged_{','.join(merge_cols)}"] = self.df[merge_cols].astype(str).agg(merge_delim.join, axis=1)
                        st.success(f"Columns {merge_cols} merged into 1 column")
                        st.write(self.df)
                    else:
                        st.warning("Please select at least one column to merge")

                # Apply transformation to column
                    numeric_columns = self.df.select_dtypes(include=['number']).columns # type: ignore
                    transform_col = st.selectbox("Select column to transform", numeric_columns)
                    transform_func = st.selectbox("Select function", ["Square", "Cube Root", "Log"])

                    if transform_func == "Square":
                        self.df[transform_col] = self.df[transform_col] ** 2 # type: ignore
                    elif transform_func == "Cube Root":
                        self.df[transform_col] = self.df[transform_col].apply(lambda x: x ** (1/3)) # type: ignore
                    elif transform_func == "Log":
                        self.df[transform_col] = self.df[transform_col].apply(lambda x: np.log(x)) # type: ignore
                    st.success(f"Applied {transform_func} to column {transform_col}")
                    st.write(self.df)

                    # Split column
                    split_col = st.selectbox("Select column to split", self.df.columns)
                    delimiter = st.text_input("Enter delimiter to split by", ",")  # Default delimiter is comma
                    split_df = self.df[split_col].str.split(delimiter, expand=True)
                    self.df = pd.concat([self.df, split_df], axis=1)
                    st.success(f"{split_col} split into multiple columns") 
                    st.write(self.df)


        except Exception as e:
            st.error(f"An error occurred while performing data transformation: {str(e)}")

            

    def faceted_browsing(self):
        if self.df is not None:
            if st.sidebar.checkbox("Faceted Browsing Tool"):
                st.header('Faceted Browsing')

        # Iterate over columns with object or category dtype
                for col in self.df.select_dtypes(include=['object', 'category']).columns:
                    unique_values = self.df[col].unique()

            # Add a search bar for the current column
                    search_term = st.text_input(f"Search {col}", "")

            # Filter unique values based on the search term
                    if search_term:
                        unique_values = [value for value in unique_values if search_term.lower() in str(value).lower()]

            # Multiselect to filter by unique values
                    selected_values = st.multiselect(f"Filter by {col}", unique_values)

            # Apply filtering
                    if selected_values:
                        self.df = self.df[self.df[col].isin(selected_values)]
                        st.write(self.df)
    def binning_discretization(self):
        if self.df is not None:
            if st.sidebar.checkbox("Binning/Discretization Tool"):
                st.title('Binning/Discretization')

                numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist() # type: ignore
                selected_numeric_column = st.selectbox('Select a numeric column for binning:', numeric_columns)

          
                num_bins = st.slider('Select the number of bins:', min_value=2, max_value=20, value=5)

                if self.df[selected_numeric_column].dtype not in ['int64', 'float64']: # type: ignore
                    st.error("Selected column is not numeric!")
                else:
                    bin_labels = [f'Bin_{i}' for i in range(1, num_bins + 1)]
                    self.df[f'{selected_numeric_column}_binned'] = pd.qcut(self.df[selected_numeric_column], q=num_bins, labels=bin_labels) # type: ignore
                    st.success(f"{selected_numeric_column} binned into {num_bins} bins")
                    st.write(self.df)
    
    def text_feature_extraction(self):
        if self.df is not None:
            if st.sidebar.checkbox("Text Feature Extraction Tool"):
                st.title('Word Count')

                text_columns = self.df.select_dtypes(include=['object']).columns.tolist() # type: ignore
                selected_text_column = st.selectbox('Select a text column for word count:', text_columns)

            
                self.df[f'{selected_text_column}_word_count'] = self.df[selected_text_column].apply(lambda x: len(str(x).split())) # type: ignore
                st.success(f"Word count created for {selected_text_column}")

            # Display the DataFrame with the word count column
                st.write('Text Feature Extraction Data')
                st.write(self.df)
    
    def feature_aggregation(self):
        try:
            if self.df is not None:
                if st.sidebar.checkbox("Feature Aggregation Tool"):
                    st.title("Feature Aggregation")
                    feature_sets = []
                    aggregation_functions = []
                    while True:
                        feature_set_key = f"feature_set_{len(feature_sets)}"
                        feature_set = st.multiselect(f"Select features for aggregation (leave blank to stop)", self.df.columns, key=feature_set_key)
                        if not feature_set:
                            break
                        feature_sets.append(feature_set)
                        aggregation_func_key = f"aggregation_func_{len(aggregation_functions)}"
                        aggregation_func = st.selectbox(f"Select aggregation function for the selected features", ['sum', 'mean', 'std'], key=aggregation_func_key)
                        aggregation_functions.append(aggregation_func)

                    if not feature_sets:
                        st.warning("No feature sets selected for aggregation.")
                        return

                    new_features = pd.DataFrame()
                    for features, agg_func in zip(feature_sets, aggregation_functions):
                    # Apply aggregation function to each feature set
                        if agg_func == 'sum':
                            aggregated_feature = self.df[features].sum(axis=1) # type: ignore
                        elif agg_func == 'mean':
                            aggregated_feature = self.df[features].mean(axis=1) # type: ignore
                        elif agg_func == 'std':
                            aggregated_feature = self.df[features].std(axis=1) # type: ignore
                        else:
                            st.error("Invalid aggregation function.")
                            return

                    # Assign a descriptive name to the new aggregated feature
                        new_feature_name = "_".join(features) + "_" + agg_func
                        new_features[new_feature_name] = aggregated_feature

                # Concatenate original DataFrame with new aggregated features
                    df_aggregated = pd.concat([self.df, new_features], axis=1)

                    st.write("Aggregated Features:")
                    st.write(df_aggregated)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")



    
    from sklearn.model_selection import train_test_split

    def split_train_test_sets(self):
        if self.df is not None:
            if st.sidebar.checkbox("Split into Train/Test Sets Tool"):
                st.title('Split Data')

                test_size = st.slider('Test set size:', min_value=0.1, max_value=0.5, value=0.2, step=0.05)

            # Select target column
                target_column = st.selectbox("Select target column:", self.df.columns)

            # Select features
                selected_features = st.multiselect("Select features:", self.df.columns.drop(target_column)) # type: ignore

                X = self.df[selected_features]
                y = self.df[target_column] # type: ignore

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.success("Data split into train/test sets")

            # Display training set
                st.subheader("Training Set")
                st.write("X_train:")
                st.write(X_train)
                st.write("y_train:")
                st.write(y_train)

            # Display testing set
                st.subheader("Testing Set")
                st.write("X_test:")
                st.write(X_test)
                st.write("y_test:")
                st.write(y_test)
        else:
            st.warning("No data available. Please load a dataset first.")


    

    def perform_clustering(self):
        if self.df is not None:
            if st.sidebar.checkbox("Clustering Tool"):
                st.title('Perform Clustering')

                num_clusters = int(st.number_input("Enter number of clusters", min_value=2, max_value=10, value=3))
            
            # Handle missing values by imputing them
                imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
                numerical_data = imputer.fit_transform(self.df.select_dtypes(include='number')) # type: ignore

            # Perform clustering on numerical columns
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(numerical_data)
            
            # Assign cluster labels to the original dataframe
                self.df['Cluster'] = kmeans.labels_
            
                st.success("Clustering performed successfully")
                st.write(self.df)

    def perform_pca(self):
        try:
            if self.df is not None:
                if st.sidebar.checkbox("PCA Tool"):
                    st.title('Perform PCA')
        
                    num_components = st.number_input("Enter number of components for PCA", min_value=2, max_value=len(self.df.columns), value=5)
            
                # Handle missing values by imputing them
                    imputer = SimpleImputer(strategy='mean')
                    numerical_data = imputer.fit_transform(self.df.select_dtypes(include='number')) # type: ignore

                    pca = PCA(n_components=int(num_components))
                    pca_result = pca.fit_transform(numerical_data)
                    pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i in range(int(num_components))])

                # Concatenate PCA-transformed data with the original DataFrame
                    pca_combined_df = pd.concat([self.df, pca_df], axis=1)

                    st.write("Original DataFrame with PCA Result:")
                    st.write(pca_combined_df)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    def perform_feature_selection(self):
        if self.df is not None:
            if st.sidebar.checkbox("Feature Selection Tool"):
                try:
                    st.title('Feature Selection')

                    target_column = st.selectbox("Select target column", self.df.columns)
                    features_to_select = st.multiselect("Select features", self.df.columns)

                    features = self.df[features_to_select]
                    target = self.df[target_column] # type: ignore

                
                    min_num_features = 0
                    max_num_features = min(len(features.columns), 5)  # Adjusted maximum value
                    default_num_features = min(min_num_features, max_num_features)  # Adjusted default value
                    num_features = st.number_input("Enter number of features to select", min_value=min_num_features, max_value=max_num_features, value=default_num_features)
                    
                    # Preprocess features to handle missing values
                    imputer = SimpleImputer(strategy='mean')
                    imputer.fit_transform(features) # type: ignore


                    st.write("Original Features:")
                    st.write(features)

                    # Univariate Feature Selection (SelectKBest)
                    selector = SelectKBest(score_func=f_regression, k=int(num_features))
                    selected_features_kbest = selector.fit_transform(features, target) # type: ignore
                    selected_features_indices_kbest = selector.get_support(indices=True)

                    st.write("Selected Features (SelectKBest):")
                    st.write(features.iloc[:, selected_features_indices_kbest]) # type: ignore

                    # Recursive Feature Elimination (RFE)
                    estimator = Lasso()
                    rfe_selector = RFE(estimator, n_features_to_select=int(num_features), step=1)
                    selected_features_rfe = rfe_selector.fit_transform(features, target) # type: ignore
                    selected_features_indices_rfe = rfe_selector.support_

                    st.write("Selected Features (RFE):")
                    st.write(features.iloc[:, selected_features_indices_rfe]) # type: ignore

                    # L1 Regularization (Lasso)
                    lasso_selector = Lasso(alpha=0.01)
                    lasso_selector.fit(features, target) # type: ignore
                    selected_features_indices_lasso = lasso_selector.coef_ != 0

                    st.write("Selected Features (Lasso):")
                    st.write(features.iloc[:, selected_features_indices_lasso]) # type: ignore

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    def create_interaction_terms(self):
        if self.df is not None:
            if st.sidebar.checkbox("Feature Interaction Tool"):
                st.title('Create New Feature')

            # Select numeric columns
                numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist() # type: ignore
                selected_numeric_columns = st.multiselect('Select numeric columns:', numeric_columns)

            # Select string columns
                string_columns = self.df.select_dtypes(include=['object']).columns.tolist() # type: ignore
                selected_string_columns = st.multiselect('Select string columns:', string_columns)

            # Select operation
                operation = st.selectbox('Select operation:', ['Add', 'Subtract', 'Multiply', 'Divide', 'Concatenate'])

                if operation != 'Concatenate' and len(selected_numeric_columns) != 2:
                    st.warning("Please select two numeric columns before creating the new feature.")
                    return

                if operation == 'Concatenate' and len(selected_string_columns) < 2:
                    st.warning("Please select at least two string columns before concatenating.")
                    return

                new_feature_name = st.text_input("Enter new feature name:")

                if new_feature_name:
                    if operation == 'Concatenate':
                        concatenated_string = ''
                        for column in selected_string_columns:
                            concatenated_string += self.df[column].astype(str) # type: ignore

                        self.df[new_feature_name] = concatenated_string
                    else:
                        if operation == 'Add':
                            self.df[new_feature_name] = self.df[selected_numeric_columns[0]] + self.df[selected_numeric_columns[1]]
                        elif operation == 'Subtract':
                            self.df[new_feature_name] = self.df[selected_numeric_columns[0]] - self.df[selected_numeric_columns[1]] # type: ignore
                        elif operation == 'Multiply':
                            self.df[new_feature_name] = self.df[selected_numeric_columns[0]] * self.df[selected_numeric_columns[1]] # type: ignore
                        elif operation == 'Divide':
                        # Avoid division by zero
                            if self.df[selected_numeric_columns[1]].all() != 0: # type: ignore
                                self.df[new_feature_name] = self.df[selected_numeric_columns[0]] / self.df[selected_numeric_columns[1]] # type: ignore
                            else:
                                st.warning("Cannot divide by zero.")
                                return

                    st.success(f"New feature '{new_feature_name}' created using operation: {operation}")
                    st.write(self.df)
        else:
            st.warning("No data available. Please load a dataset first.")


    
    def summarize_tool(self):
        if self.df is not None:
            if st.sidebar.checkbox("Summarize Tool"):
                st.title('Summarize Action')

            # Select column name
                column_name = st.selectbox('Select Column:', self.df.columns)

            # Select action
                action = st.selectbox('Select Action:', ['Sum', 'Minimum', 'Maximum', 'Count', 'Group by Identical Values', 'Concatenate Strings'])

                try:
                    if action == 'Sum':
                        if self.df[column_name].dtype in ['int64', 'float64']: # type: ignore
                            result = self.df[column_name].sum() # type: ignore
                        else:
                            st.error("Cannot perform 'Sum' operation. Selected column must be of integer or float dtype.")
                            return
                    elif action == 'Minimum':
                        result = self.df[column_name].min() # type: ignore
                    elif action == 'Maximum':
                        result = self.df[column_name].max() # type: ignore
                    elif action == 'Count':
                        result = self.df[column_name].count() # type: ignore
                    elif action == 'Group by Identical Values':
                        result = self.df.groupby(column_name).size() # type: ignore
                    elif action == 'Concatenate Strings':
                        result = self.df[column_name].str.cat(sep=', ') # type: ignore
                    # Add more actions as needed

                    st.success(f"{action} for {column_name}: {result}")
                except Exception as e:
                    st.error(f"An error occurred while summarizing: {str(e)}")
        else:
            st.warning("No data available. Please load a dataset first.")
    
    def visual_data(self):
        
        if self.df is not None:
            st.sidebar.title('Visualization')
            if st.sidebar.checkbox("Vizualization Tool"):
                st.header("Visualized Data")
                
                pyg_html = pyg.to_html(self.df, hide_data_source_config=True, vegaTheme='vega')
                components.html(pyg_html, height=800, width=1300 ,scrolling=True)
 
    def run(self):
        file_type = st.sidebar.selectbox("Select file type", ["csv", "xlsx", "json", "txt", "xml","parquet"])
        self.load_file(file_type)
        
        if self.df is not None:
            st.sidebar.title("Data Processing & Cleaning Tools")
            app.auto_configure_field_types()
            app.change_data_type()
            app.handle_duplicates_v2()
            app.data_cleansing()
            app.create_samples()
            app.filter_data()
            app.imputation()
            app.perform_binning()
            app.perform_sampling()
            app.unique_tool()
            app.scaling_normalization()
            app.encoding_categorical_columns()
            app.remove_irrelevant_columns()
            app.data_transformation()
            app.faceted_browsing()

            app.binning_discretization()
            app.text_feature_extraction()
            app.feature_aggregation()
            app.split_train_test_sets()
            app.perform_clustering()
            app.perform_pca()
            app.perform_feature_selection()
            app.create_interaction_terms()
            app.summarize_tool()
            app.visual_data()
            

if __name__ == "__main__":
    app = FileApp()
    app.run()
  

     
