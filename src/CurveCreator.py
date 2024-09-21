import cv2 
import os
import pathlib
import easyocr
import sys

import numpy as np

sys.path.append('/home/jessica/reading-kiln-termostat/src')
import InputReader
import PreProcesser


class CurveCreator():

    def __init__(self, first_number, bounding_box):
        self.first_number = first_number
        self.bounding_box = bounding_box
        self.input_reader = InputReader.InputReader()
        self.prep = PreProcesser.PreProcesser()
        self.reader = easyocr.Reader(['en'])


    def define_number(self, previous_number, current_prediction):
        if current_prediction not in np.arange(previous_number - 4, previous_number + 4):
            return previous_number
        else:
            return current_prediction

    def capture_datetime(self, image):
        datetime_results = reader.readtext(f)
        texts = list(map(list, zip(*datetime_results)))[1]
        date = [text for text in texts if '-' in text][0]
        time = [text for text in texts if ':' in text][0]

        return date, time 

    def predict_number(self, image):
        cropped_image = prep.crop_image(image, self.bounding_box)
        if self.prep.is_gray(cropped_image) == True:
            prep_image = self.prep.white_mask(cropped_image)
        else:
            prep_image = self.prep.red_mask(cropped_image)

        predicted_number = self.reader.readtext(prep_image, allowlist='0123456789', paragraph = True)[0][1].replace(" ", "") 

        return predicted_number


    def main_pipeline(self, image):

        
        date, time = capture_datetime(image)
        predicted_number = predict_number(self, image)
        predicted_number = define_number(previous_number, predicted_number)
        



