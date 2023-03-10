

# Input
# mode = "vector" # "vector", ("graph")
raw_file_name = "data/raw/Perovskite_37930data.csv" # "data/raw/Perovskite_100data.csv", "data/raw/Perovskite_37930data.csv", "data/raw/Perovskite_database_content_all_data.csv"
run_mode = "Predict" # "Hyperopt", "Train", "Predict", "Interpret"
calc_shap = False

target = "JV_default_PCE"
use_X = "all" # "all", "per", ("attempt"), ("sotsuron"),"topfti", "mat"


num_list = ["Cell_area_measured","Substrate_thickness","ETL_thickness","Perovskite_thickness","HTL_thickness_list","Backcontact_thickness_list"]
fill_way = "zero" # "dummy", "zero", "median"
Per_elem_prop = "oliynyk" # "dummy", "oliynyk", "magpie", "mat2vec" -> for PCE, only oliynyk.(2/16)
split_way = 1 # 0:onehot, 1:multihot_1, 2:multihot_2, 3:multihot_3


# Train&Hyperopt
random_state = 0
test_ratio = 0.2
valid_ratio_in_train = 0.25 # 0.8*0.25=0.2

n_trials = 25
storage_name = f"data/model/hyperopt/{use_X}_optuna_study"# data/model/attempt_optuna_study.db
# "data/model/hyperopt/attempt_optuna_study"

# Model
model_name = "GBDT" # "RF", "GBDT", "NN", "SVM"
# RF, GBDT
n_estimators = 200
max_depth = None #1000
max_leaf_nodes= None #1000
min_samples_split = 3
min_samples_leaf= 1

# NN
dim = 100
n_mid = 2
activation = "relu"
solver = "adam"
lr = 1e-3
epoch = 200

# SVM
kernel_func = "rbf" #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
polydeg = 3
epsilon = 0.1 # not-minus


# Save
save_name = f'{use_X}_{run_mode}_{model_name}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_r{str(random_state)}'
model_save_name = f'model_{use_X}_{model_name}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_r{str(random_state)}'
