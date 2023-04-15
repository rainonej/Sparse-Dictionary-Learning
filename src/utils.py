import json
import os

def load_config():
    # Load config values
    with open(r'config.json') as config_file:
        config = json.load(config_file)
    return config

def get_image_paths(dir):
    config = load_config()
    dir_to_search = os.path.join(config["IMAGES_SOURCE_DIR"], dir)
    image_paths = [os.path.join(root, name) for root, _, files in os.walk(dir_to_search) for name in files]
    image_file_names = [name for root, _, files in os.walk(dir_to_search) for name in files]
    return image_paths, image_file_names

if __name__ == "__main__":
    dir = "original"
    image_paths, image_file_names = get_image_paths(dir)
    print(image_paths)
    print(image_file_names)

