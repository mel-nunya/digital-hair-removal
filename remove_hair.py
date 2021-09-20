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