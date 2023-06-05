# Perovskite_PI
![messageImage_1678453658472](https://user-images.githubusercontent.com/79046839/224325508-e5f4116d-0061-4d6a-9eda-d0180c956995.jpg)

## Dataset
"Perovskite_database_content_all_data.csv" is the raw data downloaded from "The Perovskite Database Project" as at 31 March 2022. (#=42459)

The "Perovskite_37930data.csv" is the csv file that was saved in the "data_arrangement.ipynb" notebook with unnecessary rows and columns deleted.

The formatted "Perovskite_37930data.csv" was used in the regression analysis.(#=37930)

## Files
Each executable file has the following roles.

`main.py`: Work mainly

`train.py`: Train the machine learning models

`process.py`: Calculate molecular descriptors

`settings.py`: Input the setting

`revised_CBFV` is the file to import in `process.py`.
It is required to correct abbreviations in the composition and to convert the composition into a chemical feature vector.
It is a modified version of the Python open library `CBFV` for ease of use in this project.

## Setting arguments
The types and meanings of the arguments of `settings.py` correspond to the following, respectively.

- raw_file_name (str): "data/raw/Perovskite_37930data.csv"
- run_mode (str): "Hyperopt", "Train", "Predict", "Interpret"
- calc_shap (bool): Whether SHAP value is calculated on the interpretability of the model or not.
- target (str): "JV_default_PCE"
- use_X (str): "all"(Materials and processes for all layers), "per"(Only perovskite compositions), "mat"(Materials for all layers)
- num_list (list): List of column names that can be treated as float type numbers. ["Cell_area_measured","Substrate_thickness","ETL_thickness", "Perovskite_thickness","HTL_thickness_list","Backcontact_thickness_list"]
- fill_way = "zero" # "dummy", "zero", "median"
- per_elem_prop = "oliynyk" # "dummy", "oliynyk", "magpie", "mat2vec" -> for PCE, only oliynyk.
- split_way (int): 0 = onehot, 1 = multihot_1, 2 = multihot_2, 3 = multihot_3

///(editing now)


## Default Output Folder

///(editing now)

## Examples
To run:
`python main.py`

By this code, `main.py` reads the argument of `setting.py` and start the vectorization and training models.

