import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation


class SortImages:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def sort_images(self):
        if "perfectly" in self.images_path:
            im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
            path_start = "Ears_provided/dataset/"
            koncnica = ".png"
        else:
            im_list = sorted(glob.glob(self.images_path + '/*.jpg', recursive=True))  # yolo nrdi jpg
            path_start = "Ears_yolo/dataset/"
            koncnica = ".jpg"

        cla_d = self.get_annotations(self.annotations_path)

        for im_name in im_list:

            img = cv2.imread(im_name)
            im_name = im_name.replace("\\", "/")
            key = '/'.join(im_name.split('/')[-2:])

            if key not in cla_d:
                if "train" in self.images_path:
                    prvo_uho = key[0: 10:] + key[10 + 1::]
                else:
                    prvo_uho = key[0: 9:] + key[9 + 1::]
                print(key)
                print(prvo_uho)
                ear_class = cla_d[prvo_uho]
            else:
                ear_class = cla_d[key]

            path = path_start + str(ear_class) + "/"

            if not os.path.exists(path):
                os.mkdir(path)

            # print(path)
            img_name = key[5:]
            img_name = img_name[:-4]

            if "train" in self.images_path:
                img_name = img_name + "tr" + koncnica
            else:
                img_name = img_name + koncnica
            filename = path + img_name
            # print(filename)
            cv2.imwrite(filename, img)



if __name__ == '__main__':
    ev = SortImages()
    ev.sort_images()