import cv2
import os
import argparse

import utils

global IMAGES_SOURCE_DIR
IMAGES_SOURCE_DIR=utils.load_config()["IMAGES_SOURCE_DIR"]
print(IMAGES_SOURCE_DIR)

def main():
    image_paths, image_file_names = utils.get_image_paths("original")
    compressed_images = [convert_to_gray_scale(resize(get_image(image_path))) for image_path in image_paths]
    
    for image, image_file_name in iter(zip(compressed_images, image_file_names)):
        print(image_file_name)
        save_image(image, image_file_name)

def get_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return image
    except:
        print("Couldn't find image in specified directory")


def resize(image):
    # resize the image to 400x300 pixels
    return cv2.resize(image, (200, 150), interpolation=cv2.INTER_AREA)

def convert_to_gray_scale(image):
    # convert the resized image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def save_image(image_file_name, image):
    #make compressed directory in images if it doesn't exist
    target_path = os.path.join(IMAGES_SOURCE_DIR, "compressed_test")
    os.makedirs(target_path, exist_ok=True)

    try:
        image_path = os.join.path(target_path, image_file_name)
        print(image_path)
        cv2.imwrite(image_path, image)
        print(f"Image saved to {image_path}")

    except:
        print("Couldn't save compressed image")


if __name__ == "__main__":

    main()



