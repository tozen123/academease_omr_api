import cv2 
import numpy as np
import os, io
from google.cloud import vision
import pandas as pd



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()


def rectContour(contours): 

	rectCon = []

	for i in contours: 
		area = cv2.contourArea(i)
		#print("Area", area)

		if area>50:
			peri = cv2.arcLength(i, True)
			approx = cv2.approxPolyDP(i, 0.02*peri, True)
			# print("Corner Points", len(approx)

			if len(approx) == 4: 
				rectCon.append(i)

	rectCon = sorted(rectCon, key=cv2.contourArea, reverse = True)

	return rectCon


def getCornerPoints(cont): 
	peri = cv2.arcLength(cont, True)
	approx = cv2.approxPolyDP(cont, 0.02*peri, True)
	return approx

def reorder(myPoints):
	myPoints = myPoints.reshape((4,2))
	myPointsNew = np.zeros((4,1,2), np.int32)
	add = myPoints.sum(1)

	myPointsNew[0] = myPoints[np.argmin(add)] # (0,0)
	myPointsNew[3] = myPoints[np.argmax(add)] # (w, h)

	diff = np.diff(myPoints, axis =1)
	myPointsNew[1] = myPoints[np.argmin(diff)] # (w, 0)
	myPointsNew[2] = myPoints[np.argmax(diff)] # (0, h)
	
	return myPointsNew


def splitBoxes(img, questions, choices): 
	# img = crop_border(img)
	# cv2.imshow("adf", img)

	rows = np.vsplit(img, questions)
	boxes =[]

	for r in rows:
		cols = np.hsplit(r, choices)
		for box in cols:  
			boxes.append(box)
			# cv2.imshow("split", box)
			# cv2.waitKey(0)
	return boxes


def showAnswers(img,myIndex, grading, ans, questions, choices):
	secH = int(img.shape[1]/questions)
	secW = int(img.shape[0]/choices)
	# print(secW, secH)

	for x in range(0,questions): 
		myAns = myIndex[x]
		cX = (myAns*secW) + secW//2
		cY = (x*secH) + secH//2

		if grading[x] == 1: 
			myColor = (255, 0, 0)
		else: 
			myColor = (0, 0, 255)
			correctAns = ans[x]
			cv2.circle(img, ((correctAns*secW)+secW//2, (x*secH)+secH//2), 10, (0, 255, 0), cv2.FILLED)

		cv2.circle(img, (cX,cY), 20, myColor, cv2.FILLED)

	return img


def crop_border(img):
    # Find the coordinates of all non-white (black) pixels
    black_pixels = np.where(img == 0)  # Assuming black pixels are 0 in the binary image
    
    # Get the minimum and maximum coordinates for both dimensions
    x_min, x_max = np.min(black_pixels[1]), np.max(black_pixels[1])
    y_min, y_max = np.min(black_pixels[0]), np.max(black_pixels[0])

    # Crop the image using the bounding box coordinates
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img


def determine_set(img, img_contour, widthImg, heightImg): 

	setVal = ""



	if img_contour.size != 0: 
		cv2.drawContours(img, img_contour, -1, (0, 255, 0), 10)

		img_contour = reorder(img_contour)

		set_pt1 = np.float32(img_contour)
		set_pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

		set_matrix = cv2.getPerspectiveTransform(set_pt1, set_pt2)
		set_imgwarpColored = cv2.warpPerspective(img, set_matrix, (widthImg, heightImg))


		#Apply Threshold
		set_warpGray = cv2.cvtColor(set_imgwarpColored, cv2.COLOR_BGR2GRAY)
		set_Thresh = cv2.threshold(set_warpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]


		set_boxes = splitBoxes(set_Thresh,1, 2)

		#Get the Pixel Values
		set_PixelVal = np.zeros((1, 2))
		set_countC = 0
		set_countR = 0
		for image in set_boxes:
			set_totalPixels = cv2.countNonZero(image)
			set_PixelVal[set_countR][set_countC] = set_totalPixels
			set_countC += 1

			if (set_countC == 2): set_countR +=1 ; set_countC =0
		# print(set_PixelVal)

		#Finding Shaded Answers
		set_Index = []
		for x in range(0, 1): 
			set_arr = set_PixelVal[x]
			set_IndexVal = np.where(set_arr==np.amax(set_arr))
			set_Index.append(set_IndexVal[0][0])
		# print(set_Index)

		set_Index = int(set_Index[0])
		setVal = "A" if set_Index == 0 else "B"
		return setVal
		
	else: 
		return "Set Unidentified"



def digit_recognition(img, points, scale_factor=2):
    
    if points.shape != (4, 1, 2):
        raise ValueError(f"Expected points to have shape (4, 1, 2), but got {points.shape}")

    
    points = np.float32(points)
    
    # Calculate width and height of the region
    widthA = np.linalg.norm(points[0][0] - points[1][0])
    widthB = np.linalg.norm(points[2][0] - points[3][0])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(points[0][0] - points[2][0])
    heightB = np.linalg.norm(points[1][0] - points[3][0])
    maxHeight = max(int(heightA), int(heightB))

    
    destination_points = np.float32([[0, 0], [maxWidth, 0], [0, maxHeight], [maxWidth, maxHeight]])

    
    matrix = cv2.getPerspectiveTransform(points, destination_points)
    digit_cropped_image = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))

    
    newWidth = int(digit_cropped_image.shape[1] * scale_factor)
    newHeight = int(digit_cropped_image.shape[0] * scale_factor)
    digit_cropped_image_scaled = cv2.resize(digit_cropped_image, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)

    
    # cv2.imshow("Scaled Cropped Digit Image", digit_cropped_image_scaled)
    # cv2.waitKey(0)

    
    success, digit_cropped_image_bytes = cv2.imencode('.jpg', digit_cropped_image_scaled)
    if not success:
        raise ValueError("Failed to encode cropped image for digit recognition.")

    
    image_bytes = io.BytesIO(digit_cropped_image_bytes)
    image = vision.Image(content=image_bytes.read())

    response = client.document_text_detection(image=image)
    docText = response.full_text_annotation.text

    return docText

def count_white_black_pixels(imgThresh):
    
    total_white_pixels = cv2.countNonZero(imgThresh)

    # Calculate the total number of pixels in the image
    total_pixels = imgThresh.shape[0] * imgThresh.shape[1]

    # The number of black pixels is the total minus the white pixels
    total_black_pixels = total_pixels - total_white_pixels

    return total_white_pixels, total_black_pixels















