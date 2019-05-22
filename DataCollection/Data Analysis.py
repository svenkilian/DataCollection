from config import ROOT_DIR
import os
import requests
import pymongo
import pandas
import json
from bson.json_util import dumps, loads
# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

if __name__ == '__main__':
    # Retrieve database credentials
    print(ROOT_DIR)
    cred_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/connection_creds.txt')
    path_to_data = os.path.join(ROOT_DIR, 'DataCollection/data/data.json')

    with open(cred_path, 'r') as f:
        connection_string = f.read()

    # # Establish database connection
    # client = pymongo.MongoClient(connection_string)
    # collection = client.GitHub.Repositories
    # print(client.server_info())
    #
    # # Save database query result to json
    # data = dumps(collection.find({}))
    # print(data)
    # with open(path_to_data, 'w') as file:
    #     file.write(data)

    # Load json as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # Make DataFrame from json
    data_frame = pd.DataFrame(data)

    # columnNames = list(data_frame.head(0))
    # print(columnNames)

    # data_frame['year'] = data_frame['repo_created_at'].dt.year

    # try:
    #     data_frame['language_readme'] = data_frame['readme_text'].apply(lambda x: str(detect_langs(x)[:]))
    #     # data_frame['language_readme'] = str(detect_langs('This is such a beautiful Tag dans la cité. Und er wird besser! Très joyeuse, jedes Mal! Das darf aber dann keiner erfahren.')[:])
    # except LangDetectException:
    #     data_frame['language_readme'] = ""

    # print(data_frame['language_readme'])
    print(data_frame['readme_text'][data_frame['readme_text'].str.contains('"message":"Not Found"', regex=True)].iloc[1])

    count = data_frame['readme_text'].str.contains('"message":"Not Found"', regex=True).sum()

    print(count)

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


