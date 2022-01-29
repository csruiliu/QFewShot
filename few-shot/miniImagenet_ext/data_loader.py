import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm


# file_name -> path + name of the file
def load_images(file_name):
    # get file content
    with open(file_name, 'rb') as f:
        info = pickle.load(f)

    img_data = info['image_data']
    class_dict = info['class_dict']

    # create arrays to store x and y of images
    images = []  # x
    labels = []  # y

    # loop over all images and store them
    loading_msg = 'Reading images from %s' % file_name

    # loop over all classes
    for item in tqdm(class_dict.items(), desc=loading_msg):
        # loop over all examples from the class
        for example_num in item[1]:
            # convert image to RGB color channels
            RGB_img = cv2.cvtColor(img_data[example_num], cv2.COLOR_BGR2RGB)

            # store image and corresponding label
            images.append(RGB_img)
            labels.append(item[0])

    # return set of images
    return np.array(images), np.array(labels)


# img_set_x -> images
# img_set_y -> labels
# num_way -> number of classes for episode
# num_shot -> number of examples per class
# num_query -> number of query examples per class
def extract_episode(img_set_x, img_set_y, num_way, num_shot, num_query):
    # get a list of all unique labels (no repetition)
    unique_labels = np.unique(img_set_y)

    # select num_way classes randomly without replacement
    chosen_labels = np.random.choice(unique_labels, num_way, replace=False)
    # number of examples per selected class (label)
    examples_per_label = num_shot + num_query

    # list to store the episode
    episode = []

    # iterate over all selected labels
    for label_l in chosen_labels:
        # get all images with a certain label l
        images_with_label_l = img_set_x[img_set_y == label_l]

        # suffle images with label l
        shuffled_images = np.random.permutation(images_with_label_l)

        # chose examples_per_label images with label l
        chosen_images = shuffled_images[:examples_per_label]

        # add the chosen images to the episode
        episode.append(chosen_images)

    # turn python list into a numpy array
    episode = np.array(episode)

    # convert numpy array to tensor of floats
    episode = torch.from_numpy(episode).float()

    # reshape tensor (required)
    episode = episode.permute(0, 1, 4, 2, 3)

    # get the shape of the images
    img_dim = episode.shape[2:]

    # build a dict with info about the generated episode
    episode_dict = {
        'images': episode, 'num_way': num_way, 'num_shot': num_shot,
        'num_query': num_query, 'img_dim': img_dim}

    return episode_dict
