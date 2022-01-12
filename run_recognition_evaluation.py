import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation

class EvaluateAll:

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

    def run_evaluation(self):
        if "perfectly" in self.images_path:
            im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
            path_start = "Ears_provided/dataset/"
            koncnica = ".png"
        else:
            im_list = sorted(glob.glob(self.images_path + '/*.jpg', recursive=True))  # yolo nrdi jpg
            path_start = "Ears_yolo/dataset/"
            koncnica = ".jpg"
        iou_arr = []
        preprocess = Preprocess()
        evaluation = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        # LBP:
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()

        lbp_features_arr = []
        plain_features_arr = []
        y = []

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            # print(cla_d)
            im_name = im_name.replace("\\", "/")

            key = '/'.join(im_name.split('/')[-2:])
            # moj yolo zazna vec usec in annotationi niso pravilni
            # ce ni not to pomeni da je oznaceno kot drugo uho in mu lahko damo class prvega usesa

            if key not in cla_d:
                if "train" in self.images_path:
                    prvo_uho = key[0: 10:] + key[10 + 1::]
                else:
                    prvo_uho = key[0: 9:] + key[9 + 1::]

                y.append(cla_d[prvo_uho])
            else:
                y.append(cla_d[key])

            # Apply some preprocessing here

            # Run the feature extractors            
            plain_features = pix2pix.extract(img)
            plain_features_arr.append(plain_features)

            # lbp_features = lbp.extract(img)
            # lbp_features_arr.append(lbp_features)

        Y_plain = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')
        # Y_plain = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')

        r1 = evaluation.compute_rank1(Y_plain, y)
        print('Rank-1[%]', r1)
        # r5 = evaluation.compute_rank_n(Y_plain, y, 5)
        # print('Rank-5[%]', r5)

        # evaluation.plot_CMC(Y_plain, y, "CMC plot for LBP with provided ears")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()