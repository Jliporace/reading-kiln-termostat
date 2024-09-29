import cv2 
import os
import pathlib
import numpy as np

LOWER_BRIGHT_LIMIT = 18
UPPER_BRIGHT_LIMIT = 40
INITIAL_MASK = 235

class PreProcesser():

    def __init__(self):
        pass

    def is_bgr_grey(self, image):
        # Convert the image to greyscale
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Re-convert greyscale to BGR for comparison
        grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        # Compare the converted image with the original
        if np.array_equal(image, grey_bgr):
            return True
        else:
            return False

    def white_mask(self, image):
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

    def find_best_mask(self, image, initial_mask = INITIAL_MASK):
        mask = initial_mask
        while True:
            masked = self.grey_mask(image, mask)
            brightness = self.average_brightness(masked)
            if (brightness >= UPPER_BRIGHT_LIMIT):
                mask = mask + 4
                if mask >= 252:
                    return 252
            else:
                if (brightness <= LOWER_BRIGHT_LIMIT):
                    mask = mask - 4
                    if mask <= 5:
                        return 5
                else:
                    return mask

    def average_brightness(self, image):
        if len(image.shape) == 3:
            
            average_brightness_value = np.mean(greyscale)
            return average_brightness_value 
        # Convert the image to greyscale
        else:
            return np.mean(image)        

    def bgr_mask(self, image, bgr_lower, bg_upper):
        mask = cv2.inRange(image, lower_red, upper_red)
        return cv2.bitwise_and(image, image, mask=mask)

    def grey_mask(self, image, lower_threshold = 235, upper_threshold = 255):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(image, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
        return threshold_image

    def crop_image(self, image, bounding_box):
        return image[bounding_box[0]: bounding_box[1], bounding_box[2]:bounding_box[3]]

    def upscale(self, image, scale):
        pass