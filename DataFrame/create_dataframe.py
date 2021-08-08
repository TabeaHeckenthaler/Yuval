import pandas as pd
from os import listdir
from trajectory import data_home, solvers, SaverDirectories

df_dir = data_home + 'DataFrame\\data_frame'


def get_filenames(solver):
    if solver == 'ant':
        return [filename for filename in listdir(SaverDirectories[solver]) if 'ant' in filename]
    elif solver == 'human':
        return [filename for filename in listdir(SaverDirectories[solver]) if '_' in filename]
    else:
        return [filename for filename in listdir(SaverDirectories[solver])]


def save_df(df):
    df.to_json(df_dir + '.json')


df = pd.read_json(df_dir + '.json')
