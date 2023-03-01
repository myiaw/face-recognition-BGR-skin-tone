from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import splitlines


def detectSkinTone(img):
    min_HSV = np.array([0, 58, 30])
    max_HSV = np.array([33, 255, 255])
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinHSV = cv2.inRange(img_HSV, min_HSV, max_HSV)
    withMask = cv2.bitwise_and(img, img, mask=skinHSV)

    # B
    img_max_blue = withMask[..., 0].max()
    # G
    img_max_green = withMask[..., 1].max()
    # R
    img_max_red = withMask[..., 2].max()

    minBGR = np.array([32, 32, 77])  # MIN HSV CONVERTED TO BGR
    maxBGR = np.array([img_max_blue, img_max_green, img_max_red])

    return minBGR, maxBGR


pass


def downsizePicture(img):
    return cv2.resize(img, (220, 340), interpolation=cv2.INTER_AREA)


# image coordinates when looping are y,x,z
def processPicture(img, skin_color_down, skin_color_up):
    # width = int(img.shape[1] * 20 / 100)
    # height = int(img.shape[0] * 20 / 100)
    height = int(img.shape[0])
    width = int(img.shape[1])
    depth = int(img.shape[2])
    for i in range(height):
        for j in range(width):
            for c in range(depth):
                print(i, j, c)
    pass


def countSkinPixels(img, skin_color_down, skin_color_up):
    pass


vid = cv2.VideoCapture(0)
first_frame = False
while True:
    _, frame = vid.read()
    resized = downsizePicture(frame)
    print(resized.shape[1])
    if not first_frame:
        min_value, max_value = detectSkinTone(frame)
        first_frame = True
    height, width, _ = frame.shape
    cx = int(width / 2)
    cy = int(height / 2)
    center = frame[cy, cx]
    # print(center)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
