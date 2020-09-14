import dgl
import torch
import pandas as pd


item_cols_names = "Id | Title | Release Date | video release date |\
              IMDb URL | unknown | Action | Adventure | Animation |\
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |\
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |\
              Thriller | War | Western"
item_cols_names = [x.strip() for x in item_cols_names.split('|')]


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

    films = df.sort_index()

    return films


def prepare_films(df):
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import ColumnTransformer

    df = df.copy()

    # title isn't relevant
    df = df.drop(columns='Title')

    to_scale = ['Year', 'Month']
    rest = [col for col in df.columns if not col in to_scale]

    transformer = ColumnTransformer(
        [
            ("scaler", StandardScaler(), to_scale),
            ("identity", FunctionTransformer(), rest)
        ]
    )

    films = transformer.fit_transform(df)
    return pd.DataFrame(films, columns=to_scale + rest), transformer




def process_movies(DATASET_PATH):
    item_cols_names = "Id | Title | Release Date | video release date |\
                  IMDb URL | unknown | Action | Adventure | Animation |\
                  Children's | Comedy | Crime | Documentary | Drama | Fantasy |\
                  Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |\
                  Thriller | War | Western"
    item_cols_names = [x.strip() for x in item_cols_names.split('|')]

    movies = pd.read_csv(DATASET_PATH + "u.item", sep='|', encoding='latin-1', header=None, names=item_cols_names)
    movies.set_index('Id', inplace=True)
    movies.head(5)

    cleaned_movies = clean_movies(movies)
    prepared_movies, movies_transformer = prepare_films(cleaned_movies)
    prepared_movies.to_csv("cleanmovies.csv" , index = False)


def process_users(DATASET_PATH):

def process(DATASET_PATH):
    process_movies(DATASET_PATH)
    process_users(DATASET_PATH)



if __name__ == '__main__':
    DATASET_PATH = "./data/ml-100k/"
    process(DATASET_PATH)