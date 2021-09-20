import cv2
import numpy as np
from PIL import Image

def remove_hair(image):
  # convert image to grayScale
  grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  
  # kernel for morphologyEx
  kernel = cv2.getStructuringElement(1, (17, 17))
  
  # apply MORPH_BLACKHAT to grayScale image
  blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
  
  # apply thresholding to blackhat
  _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
  
  # inpaint with original image and threshold image
  inpaint = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
  
  final_image = cv2.medianBlur(inpaint, 5)
  
  return final_image

def remove_hair_new(image):
  kernel = np.ones((15,15),np.uint8)

  # Perform closing to remove hair and blur the image
  closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = 2)
  blur = cv2.blur(closing, (15, 15))

  # Binarize the image
  gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # Search for contours and select the biggest one
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  cnt = max(contours, key=cv2.contourArea)

  # Create a new mask for the result image
  h, w = image.shape[:2]
  mask = np.zeros((h, w), np.uint8)

  # Draw the contour on the new mask and perform the bitwise operation
  cv2.drawContours(mask, [cnt],-1, 255, -1)
  final_image = cv2.bitwise_and(image, image, mask=mask)

  # Display the result
  return final_image