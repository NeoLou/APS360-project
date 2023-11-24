# Description: Script for downloading images from urls in csv file
import gzip
import io
import os
import pandas as pd
import pickle
import PIL
import requests
import requests_cache
import time
from torchvision import transforms
import threading

# Error check for get request
def get_request_and_check(query):
    # Check for exceptions
    try:
        resp = requests.get(query)
        resp.raise_for_status() # Check for exceptions
    # Resolving exceptions
    except requests.exceptions.HTTPError as err:
        print("HTTPError")
        print(err.response.text)
        raise SystemExit(err)
    except requests.exceptions.ConnectionError as err:
        print("ConnectionError:\n")
        print(err.response.text)
        raise SystemExit(err)
    except requests.exceptions.Timeout as err:
        print("Timeout:\n")
        print(err.response.text)
        raise SystemExit(err)
    except requests.exceptions.TooManyRedirects as err:
        print("TooManyRedirects:\n")
        print(err.response.text)
        raise SystemExit(err)
    except requests.exceptions.RequestException as err:
        print("Oops, something else:\n")
        print (err.response.text)
        raise SystemExit(err)
    return resp

# Get the image from the url
def get_img_from_url(url, my_transform):
    img_content = get_request_and_check(url).content
    img = PIL.Image.open(io.BytesIO(img_content))
    img = img.convert('RGB')
    img = my_transform(img)
    return img

# Function for reading the image to dictionary
def read_img(img_dict, img_url, unique_id, my_transform, label):
    try:
        img = get_img_from_url(img_url, my_transform)
        img_dict[unique_id] = pickle.dumps((img, label))
        if len(img_dict) % 100 == 0:
            print(f"Read {len(img_dict)} images")
    except Exception as e:
        # Show the exception
        print(f"Error for image {unique_id}:{e}")

# Read the images from the urls of csv and store them in a dictionary
def read_imgs_to_dict(path_to_csv):
    print("\nReading images from csv...")
    start_time = time.time()
    df = pd.read_csv(path_to_csv) # Get dataframe from csv
    df = df[['unique_id', 'main_picture_large', 'popularity']] # Get only the columns we need
    # Define transformation for images
    my_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_dict = {} # Dictionary to store images with their ids and popularity
    # Install local cache to cache API calls (to avoid repeated calls)
    requests_cache.install_cache('./data_collection/mal_img_cache')
    # Create and start threads for reading
    for row in df.iterrows():
        unique_id = row[1]['unique_id']
        img_url = row[1]['main_picture_large']
        label = row[1]['popularity']
        read_img(img_dict, img_url, unique_id, my_transform, label)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done reading in: {:.2f} seconds".format(elapsed_time))
    return img_dict

# Thread function for writing an image to folder from dictionary
def write_file_thread(img_dict, unique_id, path):
    try:
        with open(path+str(unique_id), 'wb') as f:
            pickle.dump(img_dict[unique_id], f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        # Show the exception
        print(f"Error in thread {unique_id}:{e}")

# Write the images from the dictionary to folder
def write_imgs_from_dict(img_dict, path):
    print("Writing images to folder...")
    start_time = time.time()
    write_threads = [] # List of threads
    # Create and start threads for writing
    for unique_id in img_dict.keys():
        # Create thread
        thread = threading.Thread(target=write_file_thread,
                                  args=(img_dict, unique_id, path))
        write_threads.append(thread) # Add thread to list
        thread.start() # Start thread
    # Wait for all threads to finish
    for i, thread in enumerate(write_threads, 0):
        thread.join()
        if i % 100 == 0:
            print(f"Written {i} images")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done writing in: {:.2f} seconds".format(elapsed_time))

# Compress and zip images from folder
def compress_images(img_dict, zip_path):
    print("\nCompressing and zipping images...")
    start_time = time.time()
    with gzip.open(zip_path, "wb") as f:
        pickle.dump(img_dict, f)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done compressing in: {:.2f} seconds".format(elapsed_time))

# Load dictionary of images from compressed pickle
def load_compressed_pickle(pickle_path):
    print("\nLoading pickle...")
    start_time = time.time()
    with gzip.open(pickle_path, 'rb') as f:
        img_dict = pickle.load(f)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done loading in: {:.2f} seconds".format(elapsed_time))
    return img_dict

if __name__ == '__main__':
    # Define paths
    img_data_dir = './data_collection/img_data/'
    images_dir = './data_collection/img_data/images/'
    img_dict_path = './data_collection/img_data/img_dict_pickle_compressed.gz'
    csv_path = './data_collection/data/balanced_animes_data_max_rank=5000.csv'
    # Check if img_data, images directories exist
    if not os.path.exists(img_data_dir):
        # Create img data dir
        os.makedirs(img_data_dir)
    if not os.path.exists(images_dir):
        # Create images dir
        os.makedirs(images_dir)
    # Check if pickle exists
    if os.path.exists(img_dict_path):
        img_dict = load_compressed_pickle(img_dict_path)
    else:
        # Read images from csv and store them in a dictionary
        img_dict = read_imgs_to_dict(csv_path)
        # Compress and zip the dictionary to a pickle binary file
        compress_images(img_dict, img_dict_path)
    # Checking if images dir is empty
    if len(os.listdir(images_dir)) == 0: 
        # Write the images from the dictionary to folder
        write_imgs_from_dict(img_dict, images_dir)
    
    
    
