
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kaggle.api.kaggle_api_extended import KaggleApi


os.environ['KAGGLE_CONFIG_DIR'] = '~/.kaggle'


api = KaggleApi()
api.authenticate()


api.dataset_download_files('titanscijoan/titanic', path='titanic', unzip=True)


df = pd.read_csv('titanic/train.csv')


print("Original Data:\n", df.head())


imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])


scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


normalizer = MinMaxScaler()
df[['Age', 'Fare']] = normalizer.fit_transform(df[['Age', 'Fare']])


print("Preprocessed Data:\n", df.head()
