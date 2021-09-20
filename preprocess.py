import os
from segment_lession import segment_lession
from remove_hair import remove_hair

import cv2

input_path = "./input-images"
output_path = "./output-images"

if os.path.isdir(output_path) == 0:
  os.mkdir(output_path)

list_ids = os.listdir(input_path)
if 'LICENSE.txt' in list_ids:
  list_ids.remove('LICENSE.txt')
if 'ATTRIBUTION.txt' in list_ids:
  list_ids.remove('ATTRIBUTION.txt')
if 'ISIC_0016055.jpg' in list_ids:
  list_ids.remove('ISIC_0016055.jpg')

for k in range(len(list_ids)):
  input_image_path = input_path + "/" + list_ids[k]
  output_image_path = output_path + "/" + list_ids[k]
  print(input_image_path, output_image_path)
  input_image = cv2.imread(input_image_path)

  hairless_image = remove_hair(input_image)

  output_image = segment_lession(hairless_image)
  cv2.imwrite(output_image_path, output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

