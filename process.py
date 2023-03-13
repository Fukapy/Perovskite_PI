# process.py
# editted 2023/3/13

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from revised_CBFV import composition
import statistics
from scipy.sparse import csr_matrix, save_npz, load_npz

# Create a no split dummy variable (one-hot) without considering delimiters.
def onehot_vector_table(x_name, df):
    vector_table_data = pd.get_dummies(df[x_name]).add_prefix(x_name+"_")
    return vector_table_data

# Create a single split dummy variable.
def multihot_vector_table_1(x_name, df):
    ### Split
    first_split = df[x_name].astype(str).str.split("|")
    single_split = pd.DataFrame(list(first_split))
            
    ### Vectorize
    start = 0
    stop = 0
    small_box = []
    single_split_reshape = np.array(single_split).transpose().reshape(-1,) 
    for j in range(np.shape(single_split)[1]):
        stop = start + len(df)
        dummy = pd.get_dummies(single_split_reshape).iloc[start:stop].reset_index(drop = True)
        small_box.append(dummy)

        start = start+len(df)

    sum_box = sum(small_box)

    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis = 1)

    sum_box.where(sum_box <= 0, 1,inplace = True) # Replace all positive values with 1
    
    sum_box=sum_box.add_prefix(x_name + "_")
    
    return sum_box



# Create a double split dummy variable.
def multihot_vector_table_2(x_name,df):
    ### Split all delimiters
    # One list per line divided by "|" and ">>".
    double_split = []
    first_split = df[x_name].astype(str).str.split("|")
    for i in range(len(df)):
        X = pd.Series(first_split[i]).str.strip().str.split(">>")
        box = []
        for k in range(len(X)):
            box.extend(X[k])
        double_split.append(list(pd.Series(box).str.strip()))

    # Set "None" to the amount that isn't enough for the maximum number of splits.
    double_split = pd.DataFrame(double_split)
            
    ### Vectorize
    start = 0
    stop = 0
    small_box = []
    double_split_reshape = np.array(double_split).transpose().reshape(-1,)
    for j in range(np.shape(double_split)[1]):
        stop = start + len(df)
        dummy = pd.get_dummies(double_split_reshape).iloc[start:stop].reset_index(drop=True)
        small_box.append(dummy)

        start = start + len(df)

    sum_box = sum(small_box)

    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis = 1)

    sum_box.where(sum_box <= 0, 1, inplace=True) # Replace all positive values with 1
    
    sum_box = sum_box.add_prefix(x_name + "_")
    
    return sum_box

# Create a triple split dummy variable.
def multihot_vector_table_3(x_name,df):
    ### Split all delimiters
    # One list per line divided by "|" , ">>", and ";".
    triple_split = []
    first_split = df[x_name].astype(str).str.split("|")
    for i in range(len(df)):
        X = pd.Series(first_split[i]).str.strip().str.split(">>")
        box = []
        for k in range(len(X)):
            box.extend(list(pd.Series(X[k]).str.strip().str.split(";"))[0])

        triple_split.append(box)

    triple_split = pd.DataFrame(triple_split)
           
    ### Vectorize
    start = 0
    stop = 0
    small_box = []
    triple_split_reshape = np.array(triple_split).transpose().reshape(-1,) 
    for j in range(np.shape(triple_split)[1]):
        stop = start + len(df)
        dummy = pd.get_dummies(triple_split_reshape).iloc[start:stop].reset_index(drop = True)
        small_box.append(dummy)

        start = start + len(df)

    sum_box = sum(small_box)

    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis = 1)

    sum_box.where(sum_box <= 0, 1,inplace = True) # Replace all positive values with 1
    
    sum_box = sum_box.add_prefix(x_name + "_")
    
    return sum_box



# Convert to one-dimensional numbers.
def numerical_sum_table(num_name, df, fill_way = "zero"):

    delimit = df[num_name].astype(str).str.split("|")
    delimit = pd.DataFrame(list(delimit))

    num_sum=[]
    for i in range(len(df)):
        box = []
        for j in range(delimit.shape[1]):
            try:
                box.append(np.float_(delimit[j][i])) # str->float
            except:
                box.append(0) # strings-> 0
        num_sum.append(sum(np.array(pd.Series(box).fillna(0))))
        
        numerical_table_data = pd.DataFrame(num_sum, columns = {num_name})
        
    # When filling in missing measurements with median values.
    if fill_way == "median":
        except_zero = numerical_table_data[num_name].iloc[np.nonzero(np.array(numerical_table_data[num_name]))]
        try:
            median = statistics.median(except_zero)
        except:
            median = 0
        for i in range(len(numerical_table_data)):
            if numerical_table_data[num_name].iloc[i] == 0:
                numerical_table_data[num_name].iloc[i] = median
                
    else:
        pass
    
    return numerical_table_data


# Convert perovskite composition into composition-based feature vectors.
def cbfv_table(x_name, df, elem_prop="oliynyk"):
    corr=pd.read_csv("revised_CBFV/Perovskite_a_ion_correspond_arr.csv") # Correspondence table of abbreviations and correct chemical formulae according to certain rules (placed in "revised_CBFV").

    df[x_name] = df[x_name].astype(str).str.replace("|", "", regex=True)
    for corr_i in range(len(corr)): # Search for strings in descending alphabetical order.
        df[x_name] = df[x_name].str.replace(corr["Abbreviation"][corr_i], corr["Formal"][corr_i], regex=True)

    # When there is nothing that cannot be converted (fast processing)
    try:
        
        data = []
        for i in range(len(df)):
            data.append([df[x_name][i], 0])
        df_temp = pd.DataFrame(data,columns = ["formula", "target"])
        X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop = elem_prop)
        
        table = pd.DataFrame(X)

    # When some items cannot be converted
    except:    
          
        cbfv = []
        for i in range(len(df)):
            df_temp = pd.DataFrame([[df[x_name][i], 0]], columns = ["formula","target"])
            # Convert to feature vectors.
            try:
                X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop = elem_prop)
                cbfv.append(np.array(X.fillna(0))[0])
            except: # If the data is not convertible (ex. "NaN"), it is filled with 0.
                if elem_prop == "oliynyk":
                    cbfv.append(np.zeros(264))
                elif elem_prop == "magpie":
                    cbfv.append(np.zeros(132))
                elif elem_prop == "mat2vec":
                    cbfv.append(np.zeros(1200))
                else:
                    print("elem_prop name error!")

        # When X is generated, the name of the descriptor goes into columns.
        try:
            table = pd.DataFrame(cbfv, columns = X.columns)
        except:
            table = pd.DataFrame(cbfv)
     
    return table



# Compressing the amount of vector data.
def vec2csr(vec, csr_file_name, columns_file_name):
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name != None:
        columns_arr = np.array(vec.columns)
        np.save(columns_file_name, columns_arr)
        
# Restore compressed data to the original vector.
def csr2vec(csr_file_name, columns_file_name):
    if columns_file_name == None:
        vec = load_npz(csr_file_name).toarray()
    else:
        vec = pd.DataFrame(load_npz(csr_file_name).toarray(),
             columns=np.load(columns_file_name,allow_pickle=True))
    return vec


# Convert the entire CSV file to a vector.
def file2vector(raw_file_name, split_way, per_elem_prop, fill_way, num_list, use_X):
    
    df = pd.read_csv(raw_file_name)
    if use_X == "all":
        df = df.iloc[:,0:248] 
    elif use_X == "per":
        df = pd.DataFrame(df["Perovskite_composition_long_form"])
    elif use_X == "mat":
        df = df[["Substrate_stack_sequence", "ETL_stack_sequence","Perovskite_composition_long_form","HTL_stack_sequence","Backcontact_stack_sequence"]]
    else:
        df = df
        pass
    
    x_list = []
    for x_name in list(df.columns):

        if x_name in num_list:
            x = numerical_sum_table(x_name, df, fill_way)
            x_list.append(x)
        elif x_name == "Perovskite_composition_long_form" and per_elem_prop != "dummy":
            x = cbfv_table(x_name, df, per_elem_prop)
            x_list.append(x)
        else:
            if split_way == 0:
                x = onehot_vector_table(x_name, df)
            elif split_way == 1:
                x = multihot_vector_table_1(x_name, df)
            elif split_way == 2:
                x = multihot_vector_table_2(x_name, df)
            elif split_way == 3:
                x = multihot_vector_table_3(x_name, df)
            x_list.append(x)

    X = pd.concat(x_list, axis=1)
    X = X.fillna(0)

#     Save vectors with compressed data.
    vec2csr(vec = X,
            csr_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz",
            columns_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy")
    
    
    return X