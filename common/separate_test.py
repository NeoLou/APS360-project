import shutil
import os
from pandas import DataFrame

def read_test_ids() -> list[str]:
    """Get all the ids of the elements in our test set

    Returns:
        List: list of ids of the elements in our test set
    """
    cwd = change_cwd_to_repo_level()
    test_ids = open(f'{cwd}/common/test_ids.txt', 'r').read()
    test_ids = test_ids.strip('[]')
    test_ids = test_ids.split(', ')
    return test_ids

def change_cwd_to_repo_level() -> str:
    """changes current working directory to repo level so that we know where we are

    Returns:
        cwd: current working directory path
    """
    # get current directory
    cwd = os.getcwd()
    cwd = cwd.split('/')

    # find out what level we have to go to to get back to repo level
    idx = cwd.index('APS360-project')
    # get absolute path to where we wanna be
    cwd = '/'.join(cwd[:idx+1]) + '/images/test_set'
    os.chdir(cwd)
    return cwd

def separate_test_set_images(path_to_image_folder: str) -> None:
    """separate all the test images out

    Args:
        path_to_image_folder (string): _description_
    """
    test_ids = read_test_ids()
    cur_path = path_to_image_folder
    new_path = os.getcwd() + '/'
    for img_id in test_ids:
        shutil.move(f'{cur_path}/{img_id}', f'{new_path}/{img_id}')

def get_test_dataframe(df_data: DataFrame, test_ids) -> tuple:
    # test_ids = set(read_test_ids())
    if 'id' not in df_data.columns:
        raise Exception("dataframe needs to have an 'id' column")

    df_test = df_data[df_data['id'].isin(test_ids)]
    df_data_new = df_data[~(df_data['id'].isin(test_ids))]
    return df_test, df_data_new