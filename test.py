from remove_hair import remove_hair, remove_hair_new
import cv2

path = r'newImage.jpg'

src = cv2.imread(path)
cv2.imshow("original Image", src)

dst = remove_hair_new(src)
cv2.imwrite('oldImage.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])