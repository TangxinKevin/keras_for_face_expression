import numpy as np
import cv2
import os
from glob import glob
import dlib
import keras.backend as K

class NovaEmotions():
    def __init__(self, target_emotion_map, data_path):
        self.target_emotion_map = target_emotion_map
        self.data_path = data_path


    def load_data(self):
        subdir = os.listdir(self.data_path)
        images = list()
        labels = list()
        emotion_index_map = self.target_emotion_map
        for emotion in subdir:
            emotion_dir = os.path.join(self.data_path, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            if emotion not in emotion_index_map.keys():
                continue
            images_list = glob(join(emotion_dir, '*.png'))
            for i in images_list:
                images.append(i)
                labels.append(emotion)
        vectorized_labels = self._vectorize_label(emotion_index_map, labels)
        return (images, vectorized_labels, emotion_index_map)


    def _vectorize_label(self, emotion_index_map, labels):
        vector_labels = list()
        num_label = len(emotion_index_map.keys())
        for label in labels:
            vector_label = [0] * num_label
            vector_label[emotion_index_map[label]] = 1.0
            vector_labels.append(vector_label)
        return vector_labels

def split_dataset(images,
                  labels,
                  validation_split):
    num_samples = len(images)
    train_split_list = (validation_split, 1)
    train_start, train_stop = int(train_split_list[0] * num_samples), 
                              int(train_split_list[1] * num_samples)
    validation_split_list = (0, validation_split)
    validation_start, validation_stop = int(validation_split_list[0] * num_samples), 
                                        int(validation_split_list[1] * num_samples)
    return (images[train_start:train_stop], labels[train_start:train_stop]),
           (images[validation_start:validation_stop], labels[validation_start:validation_stop])



