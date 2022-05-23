import os
from shutil import rmtree
import random
import xml.etree.ElementTree as ET
from PIL import Image

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def split_dataset(file_path, title_str, split_rate = 0.1):
    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if file.endswith(".jpg")]
    length = len(file_list)
    indeces = random.sample(file_list, int(length * split_rate))
    for index, image in enumerate(indeces):
        image_object = Image.open(os.path.join(file_path, image))
        image_xml = image.replace(".jpg", ".xml")
        image_xml_path = os.path.join(file_path, image_xml)
        tree = ET.parse(image_xml_path)
        root = tree.getroot()
        objects = root.findall("object")
        for object in objects:
            object_name = object.find("name").text
            bndbox = object.find("bndbox")
            xmin = bndbox.find("xmin").text
            ymin = bndbox.find("ymin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text
            image_crop = image_object.crop((int(xmin), int(ymin), int(xmax), int(ymax))).convert('RGB')
            image_crop_name = "{}_{}_{}_{}_{}.jpg".format(image[:-4], xmin, ymin, xmax, ymax)
            image_crop_path = os.path.join(file_path, object_name)
            try:
                image_crop.save(os.path.join(image_crop_path, image_crop_name))
            except:
                print("{} image crop failed.".format(image_crop_name))
        # progress bar
        print("\r{} dataset: {}/{}".format(title_str, index + 1, len(indeces)), end="")


def main():
    # seed random number generator
    random.seed(0)

    # 0.9 for train, 0.1 for validation
    split_rate = 0.1

    # get data set path
    cwd = os.getcwd()
    data_set_path = os.path.join(cwd, "data_set")
    assert os.path.exists(data_set_path), "{} path does not exist.".format(data_set_path)

    # get training and validation set path
    train_path = os.path.join(data_set_path, "train")
    val_path = os.path.join(data_set_path, "val")
    assert os.path.exists(train_path), "{} path does not exist.".format(train_path)
    assert os.path.exists(val_path), "{} path does not exist.".format(val_path)

    # create classifier path for training
    mk_file(os.path.join(train_path, "face"))
    mk_file(os.path.join(train_path, "face_mask"))

    # split training list into face and face_mask
    split_dataset(train_path, "Training", split_rate)
    
    # create classifier path for validation
    mk_file(os.path.join(val_path, "face"))
    mk_file(os.path.join(val_path, "face_mask"))

    # split validation list into face and face_mask
    split_dataset(val_path, "Validation", split_rate)

if __name__ == '__main__':
    main()