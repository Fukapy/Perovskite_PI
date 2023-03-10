import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from revised_CBFV import composition
import os
import datetime
import statistics
from scipy.sparse import csr_matrix, save_npz, load_npz

# Create a dummy variable (one-hot) without considering delimiters
def onehot_vector_table(x_name,DF):
    vector_table_data = pd.get_dummies(DF[x_name]).add_prefix(x_name+"_")
    return vector_table_data


def multihot_vector_table_1(x_name,DF):
    ### 分割する
    first_split = DF[x_name].astype(str).str.split("|")
    # splitsの最大数に及ばない分をnanで埋める
    single_split = pd.DataFrame(list(first_split))
            
    ### ベクトル化する
    start = 0
    stop = 0
    small_box = []
    single_split_reshape = np.array(single_split).transpose().reshape(-1,) 
    for j in range(np.shape(single_split)[1]):
        stop = start+len(DF)
        dummy = pd.get_dummies(single_split_reshape).iloc[start:stop].reset_index(drop=True)
        small_box.append(dummy)

        start = start+len(DF)

    sum_box = sum(small_box)

    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis=1)

    sum_box.where(sum_box <= 0, 1,inplace=True) # Replace all positive values with 1
    
    sum_box=sum_box.add_prefix(x_name+"_")
    
    return sum_box



# Split all delimiters (not considering stratigraphy within functional layers)
def multihot_vector_table_2(x_name,DF):
    init_time = datetime.datetime.now()
    ### Split all delimiters
    # One list per line divided by "|" and ">>".
    double_split=[]
    first_split = DF[x_name].astype(str).str.split("|")
    for i in range(len(DF)):
        X = pd.Series(first_split[i]).str.strip().str.split(">>")
        box=[]
        for k in range(len(X)):
            box.extend(X[k])
        double_split.append( list(pd.Series(box).str.strip()) )

    # Set "None" to the amount that isn't enough for the maximum number of splits.
    double_split = pd.DataFrame(double_split)

            
    ### Vectorize
    start = 0
    stop = 0
    small_box = []
    double_split_reshape = np.array(double_split).transpose().reshape(-1,)
    for j in range(np.shape(double_split)[1]):
        stop = start+len(DF)
        dummy = pd.get_dummies(double_split_reshape).iloc[start:stop].reset_index(drop=True)
        small_box.append(dummy)

        start = start+len(DF)

    sum_box = sum(small_box)

    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis=1)

    sum_box.where(sum_box <= 0, 1,inplace=True) # Replace all positive values with 1
    
    sum_box=sum_box.add_prefix(x_name+"_")
    
    fin_time = datetime.datetime.now()
    elipse_time = fin_time - init_time
    print(f'{x_name} done {elipse_time}')
    
    return sum_box


def multihot_vector_table_3(x_name,DF):
    ### 分割する
    triple_split = []
    first_split = DF[x_name].astype(str).str.split("|")
    for i in range(len(DF)):
        X = pd.Series(first_split[i]).str.strip().str.split(">>")
        box=[]
        for k in range(len(X)):
            box.extend(list(pd.Series(X[k]).str.strip().str.split(";"))[0])

        triple_split.append(box)

    triple_split = pd.DataFrame(triple_split)
           
    ### ベクトル化する
    start = 0
    stop = 0
    small_box = []
    triple_split_reshape = np.array(triple_split).transpose().reshape(-1,) # for文の外へ
    for j in range(np.shape(triple_split)[1]):
        stop = start+len(DF)
        dummy = pd.get_dummies(triple_split_reshape).iloc[start:stop].reset_index(drop=True)
        small_box.append(dummy)

        start = start+len(DF)

    sum_box = sum(small_box)

    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis=1)

    sum_box.where(sum_box <= 0, 1,inplace=True) # 正の値を全て1に置換(1/13 かなり高速化！！)
    
    sum_box=sum_box.add_prefix(x_name+"_")
    
    return sum_box




def numerical_sum_table(num_name,DF,fill_way="zero"):
    # init_time = datetime.datetime.now()
    delimit = DF[num_name].astype(str).str.split('|')
    delimit = pd.DataFrame(list(delimit))

    num_sum=[]
    for i in range(len(DF)):
        box=[]
        for j in range(delimit.shape[1]):
            try:
                box.append(np.float_(delimit[j][i])) # str型の数字をfloatに
            except:
                box.append(0) # 数字が入るべきところで文字が入っている場合は0
        num_sum.append( sum(np.array(pd.Series(box).fillna(0))) ) # nanを0にする
        
        numerical_table_data = pd.DataFrame(num_sum, columns={num_name})
        
    
    if fill_way == "median":
        except_zero = numerical_table_data[num_name].iloc[np.nonzero(np.array(numerical_table_data[num_name]))]
        try:
            median = statistics.median(except_zero)
        except:
            median = 0
        # print(median)
        for i in range(len(numerical_table_data)):
            if numerical_table_data[num_name].iloc[i] == 0:
                numerical_table_data[num_name].iloc[i] = median
                
    else:
        pass
        
    # fin_time = datetime.datetime.now()
    # elipse_time = fin_time - init_time
    # print(f'{num_name} done {elipse_time}')
    
    return numerical_table_data


# 2022/12/8 CBFVの修正
def cbfv_table(x_name,DF,elem_prop="oliynyk"):
    corr=pd.read_csv("revised_CBFV/Perovskite_a_ion_correspond_arr.csv") # 略称と正しい化学式を一定の規則に沿ってまとめた対応表("revised_CBFV" の中に置いた)

    DF[x_name]=DF[x_name].astype(str).str.replace("|", "", regex=True)
    for corr_i in range(len(corr)): # アルファベット降順に文字列を探索するので、略称間で被らない
        DF[x_name]=DF[x_name].str.replace(corr["Abbreviation"][corr_i], corr["Formal"][corr_i],regex=True)

    # 変換できないものが無いことを前提に、高速な処理方法を優先する
    try:
        
        data=[]
        for i in range(len(DF)):
            data.append([DF[x_name][i], 0])
        df = pd.DataFrame(data,columns=["formula","target"])
        X, y, formulae, skipped = composition.generate_features(df,elem_prop=elem_prop)
        
        table = pd.DataFrame(X)

    # 上で変換できないものがあった場合はこちらに突入    
    except:    
          
        cbfv=[]
        for i in range(len(DF)):
            df=pd.DataFrame([[DF[x_name][i], 0]],columns=["formula","target"])
            # 特徴量ベクトルに変換する(変換エラーで停止しないように)   
            try:
                X, y, formulae, skipped = composition.generate_features(df,elem_prop=elem_prop) # "revised_CBFV"から
                cbfv.append(np.array(X.fillna(0))[0])
            except: # 変換不能(通常“NaN”のデータ)の場合0で埋める。
                if elem_prop == "oliynyk":
                    cbfv.append(np.zeros(264))
                elif elem_prop == "magpie":
                    cbfv.append(np.zeros(132))
                elif elem_prop == "mat2vec":
                    cbfv.append(np.zeros(1200))
                else:
                    print("elem_prop name error!")

        # Xが生成された場合、記述子の名称がcolumnsに入る
        try:
            table = pd.DataFrame(cbfv,columns=X.columns)
        except:
            table = pd.DataFrame(cbfv)
     
    return table








def vec2csr(vec, csr_file_name, columns_file_name):
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name != None:
        columns_arr = np.array(vec.columns)
        np.save(columns_file_name, columns_arr)
def csr2vec(csr_file_name, columns_file_name):
    if columns_file_name == None:
        vec = load_npz(csr_file_name).toarray()
    else:
        vec = pd.DataFrame(load_npz(csr_file_name).toarray(),
             columns=np.load(columns_file_name,allow_pickle=True))
    return vec



def file2vector(raw_file_name, split_way, Per_elem_prop, fill_way, num_list, use_X):
    
    DF = pd.read_csv(raw_file_name)
    if use_X == "all":
        df = DF.iloc[:,0:248] # 2/14
        # df = DF.iloc[:,13:261] # 2/13 modified
    elif use_X == "per":
        df = pd.DataFrame(DF["Perovskite_composition_long_form"])
    elif use_X == "sotsuron": # Delete later.
        df = DF.iloc[:,14:263] # 2/15
    elif use_X == "topfti": # 3/1
        fti_sum = pd.read_csv(f"data/model/fti_sum_all_Interpret_RF_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_r0.csv")
        topfti = []
        threshold = 1/len(fti_sum)
        for k in range(len(fti_sum)):
            if fti_sum.iloc[k][1] > threshold:
                topfti.append(fti_sum.iloc[k][0])   
        df = DF[topfti]
    elif use_X == "mat": # 3/1
        df = DF[["Substrate_stack_sequence", "ETL_stack_sequence","Perovskite_composition_long_form","HTL_stack_sequence","Backcontact_stack_sequence"]]
    else:
        df = DF
        pass
    
    x_list = []
    for x_name in list(df.columns):

        if x_name in num_list:
            x = numerical_sum_table(x_name, df, fill_way)
            x_list.append(x)
        elif x_name == "Perovskite_composition_long_form" and Per_elem_prop != "dummy":
            x = cbfv_table(x_name, df, Per_elem_prop)
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
    
    
    
    vec2csr(vec = X,
            csr_file_name = f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_csr.npz',
            columns_file_name = f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_columns.npy')
    
    
    return X