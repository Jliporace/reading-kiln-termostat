import cv2 
import os
import pathlib
import numpy as np

class PreProcesser():

    def __init__(self):
        pass

    def is_gray(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Re-convert grayscale to BGR for comparison
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Compare the converted image with the original
        if np.array_equal(image, gray_bgr):
            return True
        else:
            return False

    def white_mask(image):
        lower_white = np.array([242, 242, 242], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        # Create a mask to filter out white pixels
        mask = cv2.inRange(image, lower_white, upper_white)

        # Optionally, apply the mask to the original image to extract only white areas
        return cv2.bitwise_and(image, image, mask=mask)


    def red_mask(self, image):
        lower_red = np.array([110, 110, 240], dtype=np.uint8)
        upper_red = np.array([255, 255, 255], dtype=np.uint8)

        # Create a mask to filter out white pixels
        mask = cv2.inRange(image, lower_red, upper_red)

        # Optionally, apply the mask to the original image to extract only white areas
        return cv2.bitwise_and(image, image, mask=mask)

    def bgr_mask(self, image, bgr_lower, bg_upper):
        mask = cv2.inRange(image, lower_red, upper_red)
        return cv2.bitwise_and(image, image, mask=mask)

    def grey_mask(self, image, lower_threshold = 140, upper_threshold = 255):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(image, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
        return threshold_image

    def crop_image(self, image, bounding_box):
        return image[bounding_box[0]: bounding_box[1], bounding_box[2]:bounding_box[3]]

    def upscale(self, image, scale):
        pass