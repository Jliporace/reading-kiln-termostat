import pytesseract

import imutils
from imutils.object_detection import non_max_suppression
import argparse
import numpy as np
from PIL import Image
from IPython.display import display

class TesseractPipeline():

    def __init__(self):
        pass

    def perform_recognition(self, image_path, configs):
        print("Image: ")
        display(Image.open(image_path))
        for config in configs:
            print("config: ")
            print(config)
            print(pytesseract.image_to_string(Image.open(image_path), config=config))
