# main.py
# editted 2023/3/13

from train import *
from process import *
import settings
import datetime
import joblib
import shap
import os

def main(
    run_mode,
    model_name = None,
    random_state = None,
    raw_file_name = None,
    split_way = None,
    per_elem_prop = None,
    fill_way = None,
    save_name = None,
    model_save_name = None,
    num_list = None,
    target = None,
    use_X = None,
    test_ratio = None,
    valid_ratio_in_train = None,
    n_estimators = None,
    max_depth = None,
    max_leaf_nodes= None,
    min_samples_split = None,
    min_samples_leaf = None,
    n_trials = None,
    dim = None,
    n_mid = None,
    lr = None,
    epoch = None,
    solver = None,
    activation = None,
    storage_name = None,
    calc_shap = None,
    
):
    print(datetime.datetime.now(), "Start")
    print(split_way,per_elem_prop,fill_way)
    

    if os.path.exists(f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz"):

        X = csr2vec(csr_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz",
                    columns_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy")

        y = pd.read_csv(raw_file_name)[target]

    else:
        X = file2vector(raw_file_name, split_way, per_elem_prop, fill_way, num_list, use_X)
        y = pd.read_csv(raw_file_name)[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state = random_state)



    if run_mode == "Train":

        if os.path.exists("data/model/hyperopt/" + model_save_name + ".pkl"):
            print("Hyperparams are already optimized.")
            model = joblib.load("data/model/hyperopt/" + model_save_name + ".pkl")

        else:
            print("Use manuinputted params.") 
            model = makemodel(model_name, max_depth, max_leaf_nodes, n_estimators, min_samples_split, min_samples_leaf, dim, n_mid, activation, solver, lr, epoch)

        print(datetime.datetime.now(), "Regression Start")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_train_real = y_train

        joblib.dump(model, "data/model/" + model_save_name + ".pkl")

        y_train_result = pd.DataFrame([y_train_real.reset_index(drop = True), y_train_pred]).T.set_axis(["y_train_real", "y_train_pred"], axis = 1)
        y_train_result.to_csv("data/model/" + save_name + ".csv", index = False)


    elif run_mode == "Predict":
        model = joblib.load("data/model/" + model_save_name + ".pkl")
        y_test_pred = model.predict(X_test)
        y_test_real = y_test

        y_test_result = pd.DataFrame([y_test_real.reset_index(drop = True), y_test_pred]).T.set_axis(["y_test_real", "y_test_pred"], axis = 1)
        y_test_result.to_csv("data/model/" + save_name + ".csv", index = False)


    elif run_mode == "Hyperopt":

        study = optuna.create_study(study_name = f"{use_X}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}",
                                    storage = "sqlite:///" + storage_name + ".db",
                                        load_if_exists = True,
                                        direction = "minimize")
        study.optimize(objective_variable(model_name, X_train, y_train, valid_ratio_in_train), n_trials = n_trials)


        if model_name == "RF":


            print("Best trial:", study.best_trial)

            optimised_rf = RandomForestRegressor(
                n_estimators = study.best_params["n_estimators"],
                min_samples_split = study.best_params["min_samples_split"],
                min_samples_leaf = study.best_params["min_samples_leaf"],
            )

            joblib.dump(optimised_rf, "data/model/hyperopt/" + model_save_name + ".pkl")


        elif model_name == "GBDT":


            print("Best trial:", study.best_trial)

            optimised_gbdt = GradientBoostingRegressor(
                n_estimators = study.best_params["n_estimators"],
                min_samples_split = study.best_params["min_samples_split"],
                min_samples_leaf = study.best_params["min_samples_leaf"],
                learning_rate = study.best_params["lr"])

            joblib.dump(optimised_gbdt, "data/model/hyperopt/" + model_save_name + ".pkl")

        elif model_name == "NN":


            print("Best trial:", study.best_trial)

            optimised_nn = MLPRegressor(
                hidden_layer_sizes = tuple([study.best_params["dim"] for dim_count in range(study.best_params["n_mid"])]),
                activation = study.best_params["activation"],
                solver = study.best_params["solver"],
                learning_rate_init = study.best_params["lr"],
                max_iter = study.best_params["epoch"])

            joblib.dump(optimised_nn, "data/model/hyperopt/" + model_save_name + ".pkl")



    elif run_mode == 'Interpret':
        if model_name == "RF" or "GBDT":
            model = joblib.load("data/model/"+ model_save_name + ".pkl")
            columns_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy"
            columns = np.load(columns_file_name, allow_pickle = True)
            fti_array = model.feature_importances_
            fti = pd.Series(fti_array, index = list(columns))
            fti_sort = fti.sort_values(ascending = False)
            pd.DataFrame(fti_sort, columns = {"feature_importance"}).to_csv("data/model/fti_" + save_name + ".csv", index = True)
            
            # fti summary
            df = pd.read_csv(raw_file_name)
            if use_X == "all":
                df = df.iloc[:, 0:248] 
            
            x_col = list(df.columns)
            if "perovskite_composition_long_form" in x_col:
                x_col.remove("perovskite_composition_long_form")
            
            fti = []
            for x_name in x_col:
                fti_list = []
                for i in range(len(fti)):
                    if x_name in fti_sort.index[i]:
                        fti_list.append(fti_sort[i])
                fti_sum = sum(fti_list)
                fti.append(fti_sum)
            per_fti = 1 - sum(fti)
            fti.append(per_fti)
            fti_summary = pd.DataFrame(fti, index = x_col+["perovskite_composition_long_form"]) 
            fti_sum_sort = fti_summary.sort_values(by = 0, ascending = False)
            fti_sum_sort.to_csv("data/model/fti_sum_" + save_name + ".csv",index=True)
            
        
            if calc_shap == True:
                X = pd.DataFrame(load_npz(f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz").toarray(),
                 columns = np.load(f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy",
                                 allow_pickle = True))
                X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 0)
                del X, X_test
                explainer = shap.TreeExplainer(model, X_train[0:])
                shap_values = explainer.shap_values(X_train[0:], check_additivity = False)
                vec2csr(shap_values, "data/model/shap_" + model_save_name + "_csr.npz", None)
            
        else:
            pass
            

    save_name = save_name + str(random_state) + ".csv"


    print(datetime.datetime.now(), "Done")

    
    
if __name__ == "__main__":
    params = {
        "run_mode": settings.run_mode,
        "model_name": settings.model_name,
        "random_state": settings.random_state,
        "raw_file_name": settings.raw_file_name,
        "split_way": settings.split_way,
        "per_elem_prop": settings.per_elem_prop,
        "fill_way": settings.fill_way,
        "save_name": settings.save_name,
        "model_save_name": settings.model_save_name,
        "split_way": settings.split_way,
        "num_list": settings.num_list,
        "target": settings.target,
        "use_X": settings.use_X,
        "test_ratio": settings.test_ratio,
        "valid_ratio_in_train": settings.valid_ratio_in_train,
        "n_estimators": settings.n_estimators,
        "max_depth": settings.max_depth,
        "max_leaf_nodes": settings.max_leaf_nodes,
        "min_samples_split": settings.min_samples_split,
        "min_samples_leaf": settings.min_samples_leaf,
        "n_trials": settings.n_trials,
        "dim": settings.dim,
        "n_mid": settings.n_mid,
        "lr": settings.lr,
        "epoch": settings.epoch,
        "solver": settings.solver,
        "activation": settings.activation,
        "storage_name": settings.storage_name,
        "calc_shap": settings.calc_shap,
        
        
}

    main(**params)