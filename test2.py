import imutils
import cv2
img = cv2.imread("/home/toch/Desktop/nail_detector/data/C7D276EE-D3F5-4135-8BC0-70E479458640_1565262164/C7D276EE-D3F5-4135-8BC0-70E479458640.jpg")
img = imutils.rotate_bound(img, 90)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 600)
cv2.imshow("Image",img)
cv2.waitKey(0)