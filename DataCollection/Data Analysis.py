from tqdm import tqdm

from config import ROOT_DIR
import os
import requests
import pymongo
import pandas as pd
import numpy as np
import json
from bson.json_util import dumps, loads
# import matplotlib.pyplot as plt
# import seaborn as sns

from polyglot.text import Text
import pycountry as country
import seaborn as sns

if __name__ == '__main__':
    # Retrieve database credentials
    print(ROOT_DIR)
    cred_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/connection_creds.txt')
    path_to_data = os.path.join(ROOT_DIR, 'DataCollection/data/data.json')

    # with open(cred_path, 'r') as f:
    #     connection_string = f.read()
    #
    # # Establish database connection
    # client = pymongo.MongoClient(connection_string)
    # collection = client.GitHub.Repos_Exp
    # print(client.server_info())
    #
    # # Save database query result to json
    # data = dumps(collection.find({}))
    # # print(data)
    # with open(path_to_data, 'w') as file:
    #     file.write(data)

    # Load json as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # Make DataFrame from json
    data_frame = pd.DataFrame(data)
    print(data_frame[['repo_full_name', 'readme_text']].iloc[0:10])

    access_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/GitHub_Access_Token.txt')

    # Print column names
    print(data_frame.columns)

    # data_frame['year'] = data_frame['repo_created_at'].dt.year

    for index, row in tqdm(data_frame.iterrows(), total=data_frame.shape[0]):
        try:
            if row['readme_text'] is not ('' or None):
                text = Text(str(row['readme_text']))
                language_code = text.language.code
                if language_code is not None:
                    language_name = country.languages.get(alpha_2=language_code).name
                else:
                    language_name = None
            else:
                language_name = None
                language_code = None
        except AttributeError as ae:
            language_name = None
            language_code = None

        data_frame.at[index, 'language_readme'] = language_name
        data_frame.at[index, 'language_readme_code'] = language_code

    print(data_frame[['readme_text', 'language_readme']])

    data_frame['language_readme'].astype('category')

    print(data_frame['language_readme'].unique())
    print(data_frame.groupby(['language_readme', 'language_readme_code']).size().reset_index(name='Count').sort_values(
        'Count'))

    data_frame.to_excel(r'Output.xlsx', engine='xlsxwriter')

    # repo_count_per_year = data_frame.groupby(['year']).size().reset_index(name='counts of repos')
    # print(repo_count_per_year)
    #
    # has_wiki_count = data_frame.groupby(['has_wiki']).size().reset_index(name='counts of repos')
    # print(has_wiki_count)
    #
    # #has_homepage_count = data_frame.groupby(['homepage']).size().reset_index(name='counts of repos')
    # has_homepage_count = data_frame.iloc[:,3].isna().sum()
    # print(has_homepage_count)
    #
    # owner_count = data_frame.groupby(['repo_owner']).size().reset_index(name='counts of repos').sort_values(by=['counts of repos'], ascending=False)
    # print(owner_count)
    #
    # print(owner_count[owner_count['counts of repos'] > 1].sum())
    #
    # #corr = data_frame.loc[:, data_frame.dtypes == 'int64'].corr()
    # #sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
    #             cmap=sns.diverging_palette(220, 10, as_cmap=True))
    # #plt.show()
