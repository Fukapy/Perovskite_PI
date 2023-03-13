# settings.py
# editted 2023/3/13

# Input
raw_file_name = "data/raw/Perovskite_37930data.csv"
run_mode = "Hyperopt" # "Hyperopt", "Train", "Predict", "Interpret"
calc_shap = False

target = "JV_default_PCE"
use_X = "attempt" # "all", "per", "mat"

num_list = ["Cell_area_measured","Substrate_thickness","ETL_thickness","Perovskite_thickness","HTL_thickness_list","Backcontact_thickness_list"]
fill_way = "zero" # "dummy", "zero", "median"
per_elem_prop = "oliynyk" # "dummy", "oliynyk", "magpie", "mat2vec" -> for PCE, only oliynyk.
split_way = 1 # 0:onehot, 1:multihot_1, 2:multihot_2, 3:multihot_3


# Train&Hyperopt
random_state = 0
test_ratio = 0.2
valid_ratio_in_train = 0.25 # 0.8*0.25=0.2
n_trials = 25
storage_name = f"data/model/hyperopt/{use_X}_optuna_study_attempt"

# Model
model_name = "RF" # "RF", "GBDT", "NN"
# RF, GBDT
n_estimators = 200
max_depth = None
max_leaf_nodes= None
min_samples_split = 3
min_samples_leaf= 1

# NN
dim = 100
n_mid = 2
activation = "relu"
solver = "adam"
lr = 1e-3
epoch = 200

# Save
save_name = f"{use_X}_{run_mode}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_r{str(random_state)}"
model_save_name = f"model_{use_X}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_r{str(random_state)}"