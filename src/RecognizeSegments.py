# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}


class RecognizeSegments():

	def __init__(self):
		self.prediction = 0

	def find_countours(self, image):
		cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		digitCnts = []
		# loop over the digit area candidates
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			if w >= 8 and (h >= 15 and h <= 100):
				digitCnts.append(c)

		return digitCnts

	def merge_rects_horizontally(self, rect1, rect2):
		x1, y1, w1, h1 = rect1
		x2, y2, w2, h2 = rect2
		
		# Check if the rectangles have some horizontal overlap or are close enough horizontally
		if not (x1 + w1 < x2 or x2 + w2 < x1):
			# Combine them into a single rectangle that spans both horizontally
			new_x = min(x1, x2)
			new_y = min(y1, y2)
			new_w = max(x1 + w1, x2 + w2) - new_x
			new_h = max(y1 + h1, y2 + h2) - new_y
			return (new_x, new_y, new_w, new_h)
		else:
			return None

	def filter_bad_cnts(self, cnts):
		rectangles = [cv2.boundingRect(c) for c in cnts] 
		rectangles = sorted(rectangles, key=lambda r: r[0])

		# List to store the merged rectangles
		merged_rectangles = []

		# Iterate through the rectangles and merge the ones side by side
		i = 0
		while i < len(rectangles):
			current_rect = rectangles[i]
			for j in range(i + 1, len(rectangles)):
				merged = self.merge_rects_horizontally(current_rect, rectangles[j])
				if merged:
					current_rect = merged  # Update the current rectangle
					i = j  # Move the index to skip the merged rectangle
				else:
					break  # Stop if no more merging is possible
			merged_rectangles.append(current_rect)
			i += 1

		# for rect in merged_rectangles:
		# 	print(rect)

		return merged_rectangles

	def draw_bounding_box(self, image, rectangles, save_path):
		img = image.copy()
		for c in rectangles:
			(x, y, w, h) = c
			# Draw the rectangle on the image
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		
	# Save the image with bounding boxes to a file
		cv2.imwrite(save_path + str(self.prediction)+ '.png', img)

	def find_digits(self, image, cnts):
		digits = []

	# loop over each of the digits
		for c in cnts:
			# extract the digit ROI
			(x, y, w, h) = c
			area = w * h

			if(area < 200):
				continue
			else:
				if (area < 800):
					digits.append(1)
					continue

			roi = image[y:y + h, x:x + w]
			cv2.imwrite(f'roi_{c}.png', roi)

		#     plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
			# compute the width and height of each of the 7 segments
			# we are going to examine
			(roiH, roiW) = roi.shape
			(dW, dH) = (int(roiW * 0.5), int(roiH * 0.15))
			dHC = int(roiH * 0.05)
			# define the set of 7 segments
			segments = [
				((0, 0), (w, dH)),	# top
				((0, 0), (dW, h // 2)),	# top-left
				((w - dW, 0), (w, h // 2)),	# top-right
				((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
				((0, h // 2), (dW, h)),	# bottom-left
				((w - dW, h // 2), (w, h)),	# bottom-right
				((0, h - dH), (w, h))	# bottom
			]
			on = [0] * len(segments)
			
			for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
				# extract the segment ROI, count the total number of
				# thresholded pixels in the segment, and then compute
				# the area of the segment
				segROI = roi[yA:yB, xA:xB]
				total = cv2.countNonZero(segROI)
				area = (xB - xA) * (yB - yA)
				# if the total number of non-zero pixels is greater than
				# 50% of the area, mark the segment as "on"
				try:
					if total / float(area) > 0.5:
						on[i]= 1
				except: 
					on[i] = 0

				

				# lookup the digit and draw it on the image
			try:
				digit = DIGITS_LOOKUP[tuple(on)]
			except:
				digit = 0

			digits.append(digit)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
			cv2.putText(image, str(digit), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

		return digits

	def pipeline(self, image, save_path):
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		cnts = self.find_countours(image)
		cnts = self.filter_bad_cnts(cnts)
		digits = self.find_digits(image, cnts)
		self.prediction = int("".join(map(str, digits)))

		self.draw_bounding_box(image, cnts, save_path)


		return self.prediction