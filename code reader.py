from pyzbar import pyzbar
import cv2
import numpy as np

imageBGR = cv2.imread('8.png')
imageHSV = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2HSV)
image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)

low_thresh = (0, 0, 230)
high_thresh = (255, 255, 255)

filtered = cv2.inRange(imageHSV, low_thresh, high_thresh)
kernel = np.ones((20,20),np.uint8)
filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
conts, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cont in conts:
	rect = cv2.minAreaRect(cont)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	yMin = min(box, key=lambda x: x[0])[0]
	yMax = max(box, key=lambda x: x[0])[0]
	xMin = min(box, key=lambda x: x[1])[1]
	xMax = max(box, key=lambda x: x[1])[1]
	if yMin < 0 or yMax < 0 or xMin < 0 or xMax < 0:
		continue
	a = image[xMin:xMax, yMin:yMax]
	barcodes = pyzbar.decode(a)
	for barcode in barcodes:
		(x, y, w, h) = barcode.rect
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type
		text = "{} ({})".format(barcodeData, barcodeType)
		cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
		print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))