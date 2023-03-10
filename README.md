# Perovskite_PI

# Dataset
"Perovskite_database_content_all_data.csv" is the raw data downloaded from "The Perovskite Database Project" as at 31 March 2022. (#=42459)

The "Perovskite_37930data.csv" is the csv file that was saved in the "data_arrangement.ipynb" notebook with unnecessary rows and columns deleted.

The formatted "Perovskite_37930data.csv" was used in the regression analysis.(#=37930)

# revised_CBFV
"revised_CBFV" is the file to import in process.py.
It is required to correct abbreviations in the composition and to convert the composition into a chemical feature vector.
It is a modified version of the Python open library "CBFV" for ease of use in this project.

# files
Each executable file has the following roles.

main.py: Work mainly

train.py: Train the machine learning models

process.py: Calculate molecular descriptors

settings.py: Input the setting

"revised_CBFV" is the file to import in process.py.
It is required to correct abbreviations in the composition and to convert the composition into a chemical feature vector.
It is a modified version of the Python open library "CBFV" for ease of use in this project.

# Setting arguments
The types and meanings of the arguments of settings.py correspond to the following, respectively.

///


# Default Output Folder


# Examples
To run:
python main.py

By this code, main.py reads the argument of setting.py and start the vectorization and training models.

