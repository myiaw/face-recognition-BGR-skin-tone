from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def detectSkinTone(img):
    min_HSV = np.array([0, 58, 30])  # Skin tone HSV values found online.
    max_HSV = np.array([33, 255, 255])

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV so we can use mask.
    skinHSV = cv2.inRange(img_HSV, min_HSV,
                          max_HSV)  # Mask the image to only show pixels within the range of np.arrays.
    withMask = cv2.bitwise_and(img, img,
                               mask=skinHSV)  # Apply the mask to the image, only showing pixels that match the mask AND the original image.

    # cv2.cvtColor(withMask, cv2.COLOR_BGR2HSV)
    # cv2.imshow("mask", withMask)

    # B
    img_max_blue = withMask[..., 0].max()  # Get the max value of the blue, green, red channel.
    # G
    img_max_green = withMask[..., 1].max()
    # R
    img_max_red = withMask[..., 2].max()
    minBGR = np.array([32, 32, 77],
                      np.uint8)  # Predefined values for min of skin tone, cant get from withMask because the min is 
    # black there. 
    maxBGR = np.array([img_max_blue, img_max_green, img_max_red],
                      np.uint8)  # Max values of skin tone extracted from withMask.

    return minBGR, maxBGR


def downsizePicture(img):
    return cv2.resize(img, (220, 340))


# image coordinates when looping are y,x,z
def processPicture(img, skin_color_down, skin_color_up):
    global roi, picture_size, face_frame
    picture_size = 0
    counter = 0
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for x in range(5):  # Loop through the 5 rows and 5 columns of the image (20%).
        for y in range(5):
            if counter == 0:
                x2 += 44
                y2 += 68
                roi = img[x1:x2, y1:y2]  # Get the first 20% of the image.
                picture_size = countSkinPixels(roi, skin_color_down,
                                               skin_color_up)  # Get the amount of skin pixels in ROI.
                face_frame = roi  # Set the first ROI as the face frame.
                counter += 1
            else:
                x1 += 44
                x2 += 44
                y1 += 68
                y2 += 68
                roi = img[x1:x2, y1:y2]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue  # If the ROI went out of bounds.
                skin_pixels = countSkinPixels(roi, skin_color_down,
                                              skin_color_up)  # Get the amount of skin pixels in ROI.
                if picture_size < skin_pixels:
                    picture_size = skin_pixels  # If the amount of skin pixels in the ROI is bigger than the face frame swap it.
                    face_frame = roi

    res = cv2.matchTemplate(img, face_frame, cv2.TM_CCOEFF_NORMED)  # Find the face frame in the full frame. #(y, x)
    loc = np.where(res >= 0.9)  # Get the coordinates where the matching score of templates is 0.9 -> accurate match. #

    y_min, x_min = np.min(loc[::-1], axis=1)  # axis 0 RGB values, axis 1 x,y coordinates, axis 2 z values.  #(x, y)
    y_max, x_max = np.max(loc[::-1], axis=1)  # Min top left, max bottom right.

    # If we can, we expand the box to not just include the detected pixels but also the face.
    if y_min > 50:
        y1 = y_min - 50  # Top
    else:
        y1 = 0

    if x_min > 20:  # so we don't go into negatives.
        x1 = x_min - 20  # Left
    else:
        x1 = 0

    y_max += 30  # How wide the face frame is - we expand the box to not just include the detected pixels but also 
    # the face. 
    x_max += 110  # How tall the face frame is.

    return y1, x1, y_max, x_max


def countSkinPixels(img, skin_color_down, skin_color_up):
    range = cv2.inRange(img, skin_color_down, skin_color_up)  # Get array of pixels that are within the skin tone range.
    return cv2.countNonZero(range)  # count the amount of pixels that aren't black.


vid = cv2.VideoCapture(0)
first_frame = False
global min_value, max_value
while True:
    _, frame = vid.read()
    resized = downsizePicture(frame)
    if not first_frame:
        min_value, max_value = detectSkinTone(resized)
        first_frame = True
    y1, x1, y_max, x_max = processPicture(resized, min_value, max_value)
    y1_frame = int(y1 * frame.shape[1] / resized.shape[1])  # Scale y with the percentage of the original image.
    x1_frame = int(x1 * frame.shape[0] / resized.shape[0])  # Scale x with the percentage of the original image.
    y_max_frame = int(y_max * frame.shape[1] / resized.shape[1])
    x_max_frame = int(x_max * frame.shape[0] / resized.shape[0])
    cv2.rectangle(frame, (y1_frame, x1_frame), (y_max_frame, x_max_frame), (0, 255, 0),
                  3)  # Draw the rectangle on the original frame.
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key != -1:
        break

vid.release()
cv2.destroyAllWindows()
