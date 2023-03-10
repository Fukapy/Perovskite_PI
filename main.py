# main.py
# editted 2023/2/8

from train import *
from process import *
import settings
import datetime
import joblib
import shap

def main(
    # mode,
    run_mode,
    model_name = None,
    random_state = None,
    raw_file_name = None,
    split_way = None,
    Per_elem_prop = None,
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
    kernel_func = None,
    polydeg = None,
    epsilon = None,
    storage_name = None,
    calc_shap = None,
    
):
    print(datetime.datetime.now(), "Start")
    print(split_way,Per_elem_prop,fill_way)


    if os.path.exists(f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_csr.npz'):

        X = csr2vec(csr_file_name = f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_csr.npz',
                    columns_file_name = f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_columns.npy')

        y = pd.read_csv(raw_file_name)[target]

    else:
        X = file2vector(raw_file_name, split_way, Per_elem_prop, fill_way, num_list, use_X)
        y = pd.read_csv(raw_file_name)[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state = random_state)



    if run_mode == "Train":

        if os.path.exists("data/model/hyperopt/"+ model_save_name + '.pkl'):
            print("Hyperparams are already optimized.")
            model = joblib.load("data/model/hyperopt/"+ model_save_name + '.pkl')

        else:
            print("Use manuinputted params.") 
            model = makemodel(model_name, max_depth,max_leaf_nodes, n_estimators, min_samples_split, min_samples_leaf, dim, n_mid, activation, solver, lr, epoch, kernel_func, polydeg, epsilon)

        print(datetime.datetime.now(), "Regression Start")
        model.fit(X_train,y_train)
        y_train_pred= model.predict(X_train)
        y_train_real=y_train

        joblib.dump(model, "data/model/"+ model_save_name + '.pkl')

        y_train_result = pd.DataFrame([y_train_real.reset_index(drop=True),y_train_pred]).T.set_axis(["y_train_real","y_train_pred"],axis=1)
        y_train_result.to_csv("data/model/"+ save_name + '.csv',index=False)


    elif run_mode == "Predict":
        model = joblib.load("data/model/"+ model_save_name + '.pkl')
        y_test_pred=model.predict(X_test)
        y_test_real=y_test

        y_test_result = pd.DataFrame([y_test_real.reset_index(drop=True),y_test_pred]).T.set_axis(["y_test_real","y_test_pred"],axis=1)
        y_test_result.to_csv("data/model/"+ save_name + '.csv',index=False)


    elif run_mode == 'Hyperopt':

        study = optuna.create_study(study_name = f"{use_X}_{model_name}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}",
                                    storage='sqlite:///'+storage_name+'.db',
                                        load_if_exists=True,
                                        direction='minimize')
        study.optimize(objective_variable(model_name, X_train, y_train, valid_ratio_in_train), n_trials=n_trials)


        if model_name == "RF":


            print("Best trial:", study.best_trial)

            optimised_rf = RandomForestRegressor(
             # max_depth = study.best_params['max_depth'],
             # max_leaf_nodes = study.best_params['max_leaf_nodes'],
                n_estimators = study.best_params['n_estimators'],
             min_samples_split = study.best_params['min_samples_split'],
                min_samples_leaf = study.best_params['min_samples_leaf'],
             # n_jobs=2
            )

            joblib.dump(optimised_rf, "data/model/hyperopt/"+ model_save_name + '.pkl')


        elif model_name == "GBDT":


            print("Best trial:", study.best_trial)

            optimised_gbdt = GradientBoostingRegressor(
             max_depth = None,
             # max_leaf_nodes = study.best_params['max_leaf_nodes'],
                n_estimators = study.best_params['n_estimators'],
             min_samples_split = study.best_params['min_samples_split'],
                min_samples_leaf = study.best_params['min_samples_leaf'],
            learning_rate = study.best_params['lr'])

            joblib.dump(optimised_gbdt, "data/model/hyperopt/"+ model_save_name + '.pkl')

        elif model_name == "NN":


            print("Best trial:", study.best_trial)


            optimised_nn = MLPRegressor(
                hidden_layer_sizes = tuple([study.best_params['dim'] for dim_count in range(study.best_params['n_mid'])]),
                activation = study.best_params['activation'],
                solver = study.best_params['solver'],
                learning_rate_init = study.best_params['lr'],
                max_iter = study.best_params['epoch'])

            joblib.dump(optimised_nn, "data/model/hyperopt/"+ model_save_name + '.pkl')


        elif model_name == "SVM":

            print("Best trial:", study.best_trial)

            optimised_svm = SVR(
                kernel = study.best_params['kernel_func'],
                degree = study.best_params['polydeg'],
                epsilon = study.best_params['epsilon'])

            joblib.dump(optimised_svm, "data/model/hyperopt/"+ model_save_name + '.pkl')



    elif run_mode == 'Interpret':
        if model_name == "RF" or "GBDT":
            model = joblib.load("data/model/"+ model_save_name + '.pkl')
            columns_file_name = f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_columns.npy'
            columns = np.load(columns_file_name, allow_pickle=True)
            fti = model.feature_importances_
            FTI = pd.Series(fti,index=list(columns))
            FTI_sort = FTI.sort_values(ascending=False)
            pd.DataFrame(FTI_sort,columns={"feature_importance"}).to_csv("data/model/fti_"+ save_name + '.csv',index=True)
            
            
            # fti summary
            DF = pd.read_csv(raw_file_name)
            if use_X == "all":
                df = DF.iloc[:,0:248] # 2/14
            else:
                df = DF
                pass
            
            x_col = list(df.columns)
            x_col.remove('Perovskite_composition_long_form')
            
            FTI=[]
            for x_name in x_col:
                fti_list=[]
                for i in range(len(fti)):
                    if x_name in FTI_sort.index[i]:
                        fti_list.append(FTI_sort[i])
                        fti_sum = sum(fti_list)
                FTI.append(fti_sum)
            Per_fti = 1-sum(FTI)
            FTI.append(Per_fti)
            FTI_sum=pd.DataFrame(FTI,index=x_col+["Perovskite_composition_long_form"]) 
            FTI_sum_sort = FTI_sum.sort_values(by=0,ascending=False)
            FTI_sum_sort.to_csv("data/model/fti_sum_"+ save_name + '.csv',index=True)
            

            
            if calc_shap == True:
                X = pd.DataFrame(load_npz(f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_csr.npz').toarray(),
                 columns=np.load(f'data/csr/{use_X}_sp{str(split_way)}_{Per_elem_prop}_{fill_way}_columns.npy',
                                 allow_pickle=True))
                X_train,X_test=train_test_split(X,test_size=0.2, random_state=0)
                del X,X_test
                explainer = shap.TreeExplainer(model, X_train[0:])
                shap_values = explainer.shap_values(X_train[0:],check_additivity=False)
                vec2csr(shap_values, 'data/model/shap_' + model_save_name +'_csr.npz', None)
            
        else:
            pass
            

            
    save_name = save_name + str(random_state) + '.csv'


    print(datetime.datetime.now(), "Done")

    
    
if __name__ == "__main__":
    params = {
        'run_mode': settings.run_mode,
        'model_name': settings.model_name,
        'random_state': settings.random_state,
        'raw_file_name': settings.raw_file_name,
        'split_way': settings.split_way,
        'Per_elem_prop': settings.Per_elem_prop,
        'fill_way': settings.fill_way,
        'save_name': settings.save_name,
        'model_save_name': settings.model_save_name,
        'split_way': settings.split_way,
        'num_list': settings.num_list,
        'target': settings.target,
        'use_X': settings.use_X,
        'test_ratio': settings.test_ratio,
        'valid_ratio_in_train': settings.valid_ratio_in_train,
        'n_estimators': settings.n_estimators,
        'max_depth': settings.max_depth,
        'max_leaf_nodes': settings.max_leaf_nodes,
        'min_samples_split': settings.min_samples_split,
        'min_samples_leaf': settings.min_samples_leaf,
        'n_trials': settings.n_trials,
        'dim': settings.dim,
        'n_mid': settings.n_mid,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'solver': settings.solver,
        'activation': settings.activation,
        'kernel_func': settings.kernel_func,
        'polydeg': settings.polydeg,
        'epsilon': settings.epsilon,
        'storage_name': settings.storage_name,
        'calc_shap':settings.calc_shap,
        
        
}

    main(**params)