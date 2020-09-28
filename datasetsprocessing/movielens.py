from IPython.core.display import display, HTML
import dgl
import torch
import shutil
import pandas as pd
import sklearn
import numpy as np
import os



def clean_movies(df):
    df = df.copy()

    # columns were dropped because 1st isn't relevant and 2nd is completely NaN
    df = df.drop(columns=['IMDb URL', 'video release date'])

    # remove movies with unknown films
    df = df[df["Title"] != "unknown"]

    # parse dates in the dataset
    # using year/month only (day shouldn't be relevant much)
    df.loc[:, 'Release Date'] = pd.to_datetime(df['Release Date'], format='%d-%b-%Y')
    df.insert(1, 'Year', df['Release Date'].dt.year.astype(int))
    df.insert(2, 'Month', df['Release Date'].dt.month.astype(int))
    df = df.drop(columns='Release Date')

    films = df.reset_index(drop=True).sort_index()

    return films


def prepare_movies(df):
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import ColumnTransformer

    df = df.copy()

    # title isn't relevant
    df = df.drop(columns= ['Title' , 'Id'])

    print("@@@",df.shape)

    to_scale = ['Year', 'Month']
    rest = [col for col in df.columns if not col in to_scale]

    transformer = ColumnTransformer(
        [
            ("scaler", StandardScaler(), to_scale),
            ("identity", FunctionTransformer(), rest)
        ]
    )


    movies = transformer.fit_transform(df)

    ret =  pd.DataFrame(movies, columns=to_scale + rest)

    return ret, transformer


def process_movies(DATASET_PATH):
    item_cols_names = "Id | Title | Release Date | video release date |\
                  IMDb URL | unknown | Action | Adventure | Animation |\
                  Children's | Comedy | Crime | Documentary | Drama | Fantasy |\
                  Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |\
                  Thriller | War | Western"
    item_cols_names = [x.strip() for x in item_cols_names.split('|')]

    movies = pd.read_csv(DATASET_PATH + "u.item", sep='|', encoding='latin-1', header=None, names=item_cols_names)
    movies.head(5)

    cleaned_movies = clean_movies(movies)
    prepared_movies, movies_transformer = prepare_movies(cleaned_movies)
    np_prepared_movies = prepared_movies.to_numpy()
    np.save(os.path.join(OUTPUT_PATH, "movies"), np_prepared_movies)

    return cleaned_movies



def process_users(DATASET_PATH):
    user_cols_names = 'Id | Age | Gender | Occupation | Zip'
    user_cols_names = [x.strip() for x in user_cols_names.split('|')]

    users = pd.read_csv(DATASET_PATH + "u.user", sep='|', encoding='latin-1', header=None, names=user_cols_names)
    users.head(5)

    cleaned_users = clean_users(users)
    prepared_users, users_transformer = prepare_users(cleaned_users)
    prepared_users.head(5)
    np_prepared_users = prepared_users.to_numpy()
    np.save(os.path.join(OUTPUT_PATH, "users"), np_prepared_users)

    return cleaned_users


def clean_users(df):
    df = df.copy()

    occupations = pd.get_dummies(df['Occupation'], prefix_sep='')

    df[occupations.columns] = occupations

    df = df.drop(columns=['Occupation', 'Zip']).reset_index(drop=True)

    df.loc[:, 'Gender'] = df['Gender'].map({'M': 0, 'F': 1}).astype(int)

    return df


def prepare_users(df):
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import ColumnTransformer

    df = df.copy()
    df = df.drop(columns = ['Id'])

    to_scale = ['Age']
    rest = [col for col in df.columns if not col in to_scale]

    transformer = ColumnTransformer(
        [
            ("scaler", StandardScaler(), to_scale),
            ("identity", FunctionTransformer(), rest)
        ]
    )

    df = transformer.fit_transform(df)
    return pd.DataFrame(df, columns=to_scale + rest), transformer


def create_graph(full_df , n_users , n_movies):
    graphdic = {}

    groups = full_df.groupby('Rating')
    for rating, df in groups:
        graphdic[("user", str(rating) + "u", "movie")] = (
        torch.tensor(df['uindex'].values), torch.tensor(df['vindex'].values))
        graphdic[("movie", str(rating) + "m", "user")] = (
        torch.tensor(df['vindex'].values), torch.tensor(df['uindex'].values))

    G = dgl.heterograph(graphdic , num_nodes_dict = {"user" : n_users , "movie" : n_movies})
    return G


def get_graph_df(df, cleaned_users , cleaned_movies):
    df = df.drop(columns=['Timestamp'])

    df = df.merge(cleaned_users.reset_index()[['index', 'Id']], left_on='User', right_on=['Id'], validate='many_to_one')
    df = df.drop(columns=['Id']).rename(columns={'index': 'uindex'})

    df = df.merge(cleaned_movies.reset_index()[['index', 'Id']], left_on='Movie', right_on=['Id'],
                  validate='many_to_one')
    df = df.drop(columns=['Id']).rename(columns={'index': 'vindex'})

    return df

def run(DATASET_PATH , OUTPUT_PATH):

    cleaned_users = process_users(DATASET_PATH)
    cleaned_movies = process_movies(DATASET_PATH)

    cleaned_users.to_csv(os.path.join(OUTPUT_PATH , "cleaned_users.csv") , index = False)
    cleaned_movies.to_csv(os.path.join(OUTPUT_PATH , "cleaned_movies.csv") , index = False)

    n_users = cleaned_users.shape[0]
    n_movies = cleaned_movies.shape[0]

    suffixes = ["a"]

    for suff in suffixes:

        fname = "u{}.base".format(suff)
        fpath = DATASET_PATH + fname
        df = pd.read_csv(fpath, sep='\t', header=None, names=["User", "Movie", "Rating", "Timestamp"])
        df = get_graph_df(df , cleaned_users , cleaned_movies)
        g = create_graph(df , n_users , n_movies)

        print(g)

        dgl.save_graphs(os.path.join(OUTPUT_PATH, "u{}_train.graph").format(suff), [g])

    for suff in suffixes:
        fname = "u{}.test".format(suff)
        fpath = DATASET_PATH + fname
        df = pd.read_csv(fpath, sep='\t', header=None, names=["User", "Movie", "Rating", "Timestamp"])
        df = get_graph_df(df, cleaned_users, cleaned_movies)
        g = create_graph(df , n_users , n_movies)

        print(g)

        dgl.save_graphs(os.path.join(OUTPUT_PATH, "u{}_test.graph").format(suff), [g])

if __name__ == '__main__':
    print(os.getcwd())
    DATASET_PATH = "./data/ml-100k/"
    OUTPUT_PATH = "./data/ml-100k_processednew/"

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    os.makedirs(OUTPUT_PATH , exist_ok=True)

    run(DATASET_PATH , OUTPUT_PATH)