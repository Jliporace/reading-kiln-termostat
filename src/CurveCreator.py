import cv2 
import os
import pathlib
import easyocr
import sys
import re 

import numpy as np
import pandas as pd

sys.path.append('/home/jessica/reading-kiln-termostat/src')
import InputReader
import PreProcesser
import RecognizeSegments

from datetime import datetime
reader = easyocr.Reader(['en'])

PNG_COMPRESSION = 0

class CurveCreator():

    def __init__(self, firing_name, save_path, initial_temp = 0, final_temp = 0, bounding_box = [], video_to_frames = False):

        # self.dir_path = dir_path
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.bounding_box = bounding_box
        self.video_to_frames = video_to_frames
        self.firing_name = firing_name

        self.input_reader = InputReader.InputReader()
        self.prep = PreProcesser.PreProcesser()
        self.reader = easyocr.Reader(['en'])
        
        self.error = ''
        self.bad_predictions = 0
        self.curve = [initial_temp]
        self.prediction = 0
        # self.firing_save_path = save_path 

        self.firing_save_path = save_path + '/' + firing_name
        if not os.path.exists(self.firing_save_path):
            os.makedirs(self.firing_save_path)

    def capture_datetime(self, image):
        datetime_results = self.reader.readtext(image)
        texts = list(map(list, zip(*datetime_results)))[1]
        date = [text for text in texts if '-' in text][0].replace('O', '0')
        time = [text for text in texts if ':' in text][0].replace('O', '0')

        return (date.replace(" ", ""), time.replace(" ", ""))

    def count_consecutive_last_element(self, lst):
        if not lst:  # Check if the list is empty
            return 1

        last_element = lst[-1]  # Get the last element
        count = 0

        # Iterate from the end of the list backward
        for element in reversed(lst):
            if element == last_element:
                count += 1
            else:
                break  # Stop counting when a different element is encountered
        
        return count

    def bad_prediction(self, predicted_number):
        if (predicted_number - self.curve[-1] < 0):
            self.error += "Leitura de número anterior. O forno está esfriando?"
            return True
        else:
            if (predicted_number - (self.curve[-1] + (self.count_consecutive_last_element(self.curve) - 1))) > 4:
                return True
            else:
                if (predicted_number > self.final_temp):
                    self.error += "Número maior que a temperatura final"
                    return True
                else:
                    return False

    def correct_curve(self, predicted_number):
        if(self.bad_predictions >= 2 and (predicted_number - self.curve[-1] > 0)):
            rate = (predicted_number - self.curve[-1])/(self.bad_predictions + 1)
            true_curve = [round(prediction) for prediction in np.arange(self.curve[-1],predicted_number,rate)]
            self.curve[-(self.bad_predictions + 1):] = true_curve
        else:
            return

    def predict_number(self, image, method):
        # cropped_image = self.prep.crop_image(image, self.bounding_box)
        lower_threshold = self.prep.find_best_mask(image)
        prep_image = self.prep.grey_mask(image, lower_threshold)

        try:
            if method == 'easy-ocr':
                predicted_number = int(self.reader.readtext(prep_image, allowlist='0123456789', paragraph = True)[0][1].replace(" ", ""))
            else:
                if method == 'tessaract':
                    pass
                else:
                    rec_seg = RecognizeSegments.RecognizeSegments()
                    
                    countours_path = self.firing_save_path + "/cnts_frames/" 
                    if not os.path.exists(countours_path):
                        os.makedirs(countours_path)

                    predicted_number = rec_seg.pipeline(prep_image, countours_path)

            self.prediction = predicted_number
        except Exception as e:
            self.error += f" Não foi possível extrair leitura. Exceção {e}"
            self.bad_predictions += 1
            self.prediction = 0
            return (prep_image, self.curve[-1])
        
        return self.post_processing(predicted_number, prep_image)

    def post_processing(self, predicted_number, prep_image):

        if self.bad_prediction(predicted_number): 
            original_prediction = predicted_number
            predicted_number = int('11' + str(predicted_number)[2:])
            if self.bad_prediction(predicted_number):
                predicted_number = int('12' + str(original_prediction)[2:])
                if self.bad_prediction(predicted_number):
                    predicted_number = int(str(original_prediction).replace('8','0'))
                    if self.bad_prediction(predicted_number):
                        if len(str(original_prediction)) == 3:
                            predicted_number = int('1' + str(original_prediction))
                            if self.bad_prediction(predicted_number):
                                predicted_number = int(str(original_prediction) + '1')
                                if self.bad_prediction(predicted_number):
                                    self.bad_predictions += 1
                                    self.error = 'Não foi possível achar número plausível'
                                    return (prep_image, self.curve[-1])
                                else:
                                    if(self.bad_predictions >= 2 and (predicted_number - self.curve[-1] > 0)):
                                        self.correct_curve(predicted_number)
                                    self.bad_predictions = 0
                                    return (prep_image, predicted_number)
                            else:
                                self.correct_curve(predicted_number)
                                self.bad_predictions = 0
                                return (prep_image, predicted_number)                 
                        else:
                            self.error = 'Não foi possível achar número plausível'
                            self.bad_predictions += 1
                            return (prep_image, self.curve[-1])
                    else:
                        self.correct_curve(predicted_number)
                        self.bad_predictions = 0
                        return (prep_image, predicted_number)
                else: 
                    self.correct_curve(predicted_number)
                    self.bad_predictions = 0
                    return (prep_image, predicted_number)
            else: 
                self.correct_curve(predicted_number)
                self.bad_predictions = 0
                return (prep_image, predicted_number)
        else:
            self.correct_curve(predicted_number)
            self.bad_predictions = 0
            return (prep_image, predicted_number)

    def frame_firing(self):
        frames_path = self.firing_save_path + "/frames/"
        self.input_reader.frame_recorded_firing('/home/jessica/reading-kiln-termostat/data/recordings/10-10-2023-esmalte/original', frames_path)
        
    def save_cropped_images(self):
        frames_path = self.firing_save_path + "/frames/"
        cropped_path = self.firing_save_path + "/cropped_datetime/"
        if not os.path.exists(cropped_path):
            os.makedirs(cropped_path)

        frames_path_ir = sorted(pathlib.Path(frames_path).glob('**/*'), key=os.path.getmtime)
        frames = [str(f) for f in frames_path_ir if f.is_file()]

        for frame in frames:
            image = cv2.imread(frame)
            date, time = self.capture_datetime(image)
            try: 
                date_obj = datetime.strptime(date + ' ' + time, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                date_obj = datetime.strptime('2024-01-01 01:01:01', "%Y-%m-%d %H:%M:%S")
            
            cropped_image = self.prep.crop_image(image, self.bounding_box)

            file_name = cropped_path + date + '_' + time + '.png'
            r = cv2.imwrite(file_name, cropped_image), [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION]

    def curve_energy(self, signal):
        signal_1160 = [s for s in signal if s >= 1160]
        return np.sum([((s - 1175) ** 2) + 0.3 * s  for s in signal_1160])

    def create_curve(self, test_name, method):
        PNG_COMPRESSION = 0
        self.error = ' '

        # frames_path = self.firing_save_path + "/frames/"
        # if (self.video_to_frames == True):
        #     self.input_reader.frame_recorded_firing(self.dir_path, frames_path)

        predictions_path = self.firing_save_path + "/prediction_frames/" 
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)

        cropped_path = f'/home/jessica/reading-kiln-termostat/data/recordings/{self.firing_name}/cropped_datetime/'
        frames_path_ir = sorted(pathlib.Path(cropped_path).glob('**/*'), key=lambda x: x.name)

        frames = [str(f) for f in frames_path_ir if f.is_file()]

        curve_prediction = pd.DataFrame(columns = ['timestamp', 'prediction', 'original_prediction', 'curve', 'error'])
        error_df =  pd.DataFrame(columns = ['timestamp', 'true_temperature', 'original_prediction_ocr', 'processed_prediction'])

        image = cv2.imread(frames[0])
        pattern = r'(\d{4})_(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})'
        match = re.search(pattern, str(frames[0]))
        if match:
            true_temperature = match.group(1)
            date = match.group(2)
            time = match.group(3)
        else:
            true_temperature = 0
            date, time = ('2024-01-01', '00:00:00')

        previous_time = datetime.strptime(date + ' ' + time, "%Y-%m-%d %H:%M:%S")        
        true_curve = [int(true_temperature)]
        ocr_curve = [self.initial_temp]

        for frame in frames[1:]:
            image = cv2.imread(frame)
            pattern = r'(\d{4})_(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})'
            match = re.search(pattern, str(frame))
            if match:
                true_temperature = match.group(1)
                date = match.group(2)
                time = match.group(3)
            else:
                true_temperature = 0
                date, time = previous_time.date().strftime("%Y-%m-%d"), previous_time.time().strftime("%H:%M:%S")
            try: 
                date_obj = datetime.strptime(date + ' ' + time, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                date_obj = previous_time 
                self.error += 'Não foi possível ler data/hora'

            if (date_obj - previous_time).total_seconds() < -90:
                self.error += ' Tempo Negativo - leitura de frame anterior'
                continue

            else:
                if(date_obj - previous_time).total_seconds() > 90 :
                    self.error += ' Tempo de leitura entre frames maior que um minuto'
                    delta_t = (date_obj - previous_time).total_seconds() / 60
                    if delta_t > 2:
                        append = [self.curve[-1]] * int(np.floor(delta_t - 1))
                        self.curve = self.curve + append

            prediction_image, predicted_number = self.predict_number(image, method)
            self.curve = self.curve + [predicted_number]
            true_curve = true_curve + [int(true_temperature)]
            ocr_curve = ocr_curve + [self.prediction]

            if (self.bad_predictions >=6):
                self.error += ' Últimos 6 números iguais - leitura travada'

            file_name = predictions_path + date + '_' + time + '_' + str(predicted_number) + '_' + str(self.prediction) + '.png'
            r = cv2.imwrite(file_name, prediction_image), [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION]

            row = {'timestamp': date + time, 'prediction': predicted_number, 'original_prediction': self.prediction, 'curve': self.curve, 'error': self.error}
            curve_prediction.loc[len(curve_prediction)] = row

            error_row = {'timestamp': date + time, 'true_temperature': true_temperature, 'original_prediction_ocr': self.prediction, 'processed_prediction': predicted_number}
            error_df.loc[len(error_df)] = error_row

            
            self.error = ' '
            previous_time = date_obj


        accuracy_ocr = (np.array(true_curve) == np.array(ocr_curve)).sum() / len(true_curve)
        accuracy_curve = (np.array(true_curve) == np.array(self.curve)).sum() / len(true_curve)
        energy_diff = self.curve_energy(true_curve) - self.curve_energy(self.curve)
        results_df = pd.DataFrame(columns = ['nome', 'teste', 'ac_ocr', 'acr_previsao', 'erro_energia'])
        results_row = {'nome': self.firing_name , 'teste': test_name, 'ac_ocr': accuracy_ocr, 'acr_previsao' : accuracy_curve, 'erro_energia': energy_diff }
        
        results_df.loc[len(results_df)] = results_row
        curve_prediction.to_csv(self.firing_save_path + "/curve_predictions.csv")
        error_df.to_csv(self.firing_save_path + "/error_df.csv")

        results_df.to_csv(self.firing_save_path + '/results_df.csv')
        return self.curve, results_df


        



