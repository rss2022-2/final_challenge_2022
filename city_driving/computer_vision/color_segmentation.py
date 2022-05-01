import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	COLOR_MIN = np.array([0, 160, 155],np.uint8)
	COLOR_MAX = np.array([25, 255, 255],np.uint8)
	frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
	imgray = frame_threshed
	ret,thresh = cv2.threshold(frame_threshed,127,255,0)
	_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Find the index of the largest contour
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]

	x,y,w,h = cv2.boundingRect(cnt)
	# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	# cv2.imshow("Show",img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return ((x,y),(x+w,y+h))
