"""
install opencv
https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html

$ sudo apt-get install python3-opencv
"""
from pathlib import Path
import os
import cv2 as cv

# -- Image manipulation

def grayscale(image):
    return image.mean(axis=-1)

def resize(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv.resize(image, dim, interpolation = cv.INTER_AREA)
# --

def first_frame(filename, resize_factor=1):
    """
    Takes the first frame from a video file,
    converts it to grayscale and resizes
    """
    vidcap = cv.VideoCapture(filename)
    success, image = vidcap.read()
    if success:
        # 1. Convert to grayscale
        image = grayscale(image)

        # 2. Downscale image
        if resize_factor != 1:
            image = resize(image, resize_factor)
        return image

def main():
    """ Reads the dataset and converts video to images """
    data_dir = Path('./data/upper-body-thermal-images-and-associated-clinical-data-from-a-pilot-cohort-study-of-covid-19-1.1/')
    video_dir = data_dir / 'termal_mpg_data'
    i = 1
    wildcard = str(video_dir) + '/*/Front/Front.mp4'
    print('Walking directories in', wildcard)
    subdirs = os.listdir(video_dir)
    subdirs.sort()

    for subdir  in subdirs:
        fname = str(video_dir / subdir / 'Front/Front.mp4')
        image = first_frame(fname, resize_factor=0.1)
        ID = f'{subdir}'
        impath = f'./converted_data/images/{ID}.png'
        cv.imwrite(impath, image)
        print('Saved:', impath)
        i+=1


if __name__=="__main__":
    main()
