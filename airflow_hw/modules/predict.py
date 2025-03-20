import pandas as pd
import dill
import os
import glob
import json

from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')

def predict():

    directory_model = f'{path}/data/models'
    directory_test = f'{path}/data/test'


    def get_later_file(directory):
        pattern = os.path.join(directory, 'cars_pipe_*.pkl')
        files = glob.glob(pattern)

        if not files:
            return f'Filter not search file for predict. Root: {directory}'

        return max(files, key=os.path.getctime)


    later_file = get_later_file(directory_model)
    df_predict = pd.DataFrame(columns=['id', 'predict'])

    with open(f'{later_file}', 'rb') as file_model:
        model = dill.load(file_model)

    for filename in os.listdir(directory_test):
        file_path = os.path.join(directory_test, filename)

        if os.path.isfile(file_path):
            with open(file_path, 'r') as file_test:
                data = json.load(file_test)

            df = pd.DataFrame(data, index=[0])
            df_predict.loc[len(df_predict)] = [filename, model.predict(df)]

    df_predict.to_csv(f'{path}/data/predictions/predict_test_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
