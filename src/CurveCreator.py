import cv2 
import os
import pathlib
import easyocr
import sys

import numpy as np
import pandas as pd

sys.path.append('/home/jessica/reading-kiln-termostat/src')
import InputReader
import PreProcesser

from datetime import datetime
reader = easyocr.Reader(['en'])


class CurveCreator():

    def __init__(self, dir_path, save_path, first_number, bounding_box):
        self.dir_path = dir_path
        self.first_number = first_number
        self.bounding_box = bounding_box

        self.input_reader = InputReader.InputReader()
        self.prep = PreProcesser.PreProcesser()
        self.reader = easyocr.Reader(['en'])
        self.error = ''

        self.firing_save_path = save_path + pathlib.Path(dir_path).name 
        if not os.path.exists(self.firing_save_path):
            os.makedirs(self.firing_save_path)

    def capture_datetime(self, image):
        datetime_results = self.reader.readtext(image)
        texts = list(map(list, zip(*datetime_results)))[1]
        date = [text for text in texts if '-' in text][0]
        time = [text for text in texts if ':' in text][0]

        return (date.replace(" ", ""), time.replace(" ", ""))

    def predict_number(self, image, previous_number):
        cropped_image = self.prep.crop_image(image, self.bounding_box)
        if self.prep.is_gray(cropped_image) == True:
            prep_image = self.prep.white_mask(cropped_image)
        else:
            prep_image = self.prep.red_mask(cropped_image)

        try:
            predicted_number = int(self.reader.readtext(prep_image, allowlist='0123456789', paragraph = True)[0][1].replace(" ", ""))
        except Exception as e:
            print(e)
            return (cropped_image, previous_number)

        print(predicted_number)
        if abs(predicted_number - previous_number) > 4: 
            return (cropped_image, previous_number)
        else:
            return (cropped_image, predicted_number)

    def create_curve(self):
        curve = []
        png_compression_level = 0
        self.error = ' '

        frames_path = self.firing_save_path + "/frames/"
        # self.input_reader.frame_recorded_firing(self.dir_path, frames_path)

        predictions_path = self.firing_save_path + "/prediction_frames/"
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)

        frames_path_ir = sorted(pathlib.Path(frames_path).glob('**/*'), key=os.path.getmtime)
        frames = [str(f) for f in frames_path_ir if f.is_file()]
        curve_prediction = pd.DataFrame(columns = ['timestamp', 'prediction', 'curve', 'error'])

        image = cv2.imread(frames[0])
        date, time = self.capture_datetime(image)
        date_obj = datetime.strptime(date + ' ' + time, "%Y-%m-%d %H:%M:%S")
        previous_time = date_obj

        previous_number = self.first_number
        cropped_image, predicted_number = self.predict_number(image, previous_number)
        curve = curve + [predicted_number]

        file_name = predictions_path + date + '_' + time + '_' + str(predicted_number) + '.png'
        r = cv2.imwrite(file_name, cropped_image), [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level]

        row = {'timestamp': date + time, 'prediction': predicted_number, 'curve': curve}
        curve_prediction.loc[len(curve_prediction)] = row
        
        previous_number = predicted_number

        for frame in frames[1:]:
            image = cv2.imread(frame)
            date, time = self.capture_datetime(image)
            date_obj = datetime.strptime(date + ' ' + time, "%Y-%m-%d %H:%M:%S")

            if (date_obj - previous_time).total_seconds() < 0:
                self.error = 'Tempo Negativo - leitura de frame anterior'
                continue

            else:
                if(date_obj - previous_time).total_seconds() > 65 :
                    self.error = 'Tempo de leitura entre frames maior que um minuto'
                    delta_t = (date_obj - previous_time).total_seconds() / 60
                    if delta_t > 2:
                        append = [previous_number] * int(np.floor(delta_t - 1))
                        curve = curve + append

            cropped_image, predicted_number = self.predict_number(image, previous_number)
            curve = curve + [predicted_number]

            file_name = predictions_path + date + '_' + time + '_' + str(predicted_number) + '.png'
            r = cv2.imwrite(file_name, cropped_image), [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level]

            row = {'timestamp': date + time, 'prediction': predicted_number, 'curve': curve, 'error': self.error}
            curve_prediction.loc[len(curve_prediction)] = row

            previous_number = predicted_number
            self.error = ' '

        curve_prediction.to_csv(self.firing_save_path + "/curve_predictions.csv")

        return curve


        



