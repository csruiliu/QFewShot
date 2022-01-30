import os
import numpy as np
import imageio


def create_images_dataset():
    COVID_DATA_DIR = os.path.abspath(os.path.join(os.path.abspath('setup.py'), '../../data/covid_thermal_dataset/data'))
    COVID_DATA_LIST_PATH = os.path.abspath(os.path.join(os.path.abspath('setup.py'),
                                                        '../../data/covid_thermal_dataset/converted_data/description.csv'))

    with open(COVID_DATA_LIST_PATH, 'r') as f:
        next(f)
        for line in f:
            metadata_list = line.strip().split(',')
            image_name = metadata_list[1]
            cat = metadata_list[-1]

            cat_dir = COVID_DATA_DIR + '/' + cat

            if os.path.isdir(cat_dir):
                os.rename(COVID_DATA_DIR + '/' + image_name + '.png', cat_dir + '/' + image_name + '.png')
            else:
                os.mkdir(cat_dir)
                os.rename(COVID_DATA_DIR + '/' + image_name + '.png', cat_dir + '/' + image_name + '.png')


def create_splits():
    COVID_SPLITS_DIR = os.path.abspath(os.path.join(os.path.abspath('setup.py'),
                                                    '../../data/covid_thermal_dataset/splits/rui'))
    COVID_DATA_LIST_PATH = os.path.abspath(os.path.join(os.path.abspath('setup.py'),
                                                        '../../data/covid_thermal_dataset/converted_data/description.csv'))

    with open(COVID_DATA_LIST_PATH, 'r') as f:
        next(f)
        with open(COVID_SPLITS_DIR + '/train.txt', "w") as file:
            for line in f:
                metadata_list = line.strip().split(',')
                image_name = metadata_list[1]
                cat = metadata_list[-1]
                file.write(cat + '/' + image_name + '\n')

    with open(COVID_DATA_LIST_PATH, 'r') as f:
        next(f)
        with open(COVID_SPLITS_DIR + '/val.txt', "w") as file:
            for line in f:
                metadata_list = line.strip().split(',')
                image_name = metadata_list[1]
                cat = metadata_list[-1]
                if np.random.rand() >= 0.5:
                    file.write(cat + '/' + image_name + '\n')

    with open(COVID_DATA_LIST_PATH, 'r') as f:
        next(f)
        with open(COVID_SPLITS_DIR + '/test.txt', "w") as file:
            for line in f:
                metadata_list = line.strip().split(',')
                image_name = metadata_list[1]
                cat = metadata_list[-1]
                if np.random.rand() >= 0.4:
                    file.write(cat + '/' + image_name + '\n')


def main():
    create_images_dataset()
    create_splits()


if __name__ == "__main__":
    main()
