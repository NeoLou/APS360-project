
import pandas as pd

def get_data_loaders(path_to_csv):
    df = pd.read_csv(path_to_csv, index_col="index")
    #df.columns = df.iloc[0]
    print(df.columns)
    #df = df[['index', 'id', 'main_picture_large', 'popularity']]
    #print(df)
    #train_size = int(0.8*len(ds))
    #val_size = len(ds) - train_size
    #train, val = random_split(ds, [train_size, val_size])
    #return DataLoader(train, batch_size=32), DataLoader(val, batch_size=32)
    
if __name__ == '__main__':
    get_data_loaders('./data_collection/data/balanced_animes_data_max_rank=5000.csv')