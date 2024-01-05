import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2.aruco as aruco

# Define the range of yellow color in HSV
yellow_lower = np.array([20, 100, 100], dtype="uint8")
yellow_upper = np.array([30, 255, 255], dtype="uint8")

def aruco_detect():
    # Load the image
    image = cv2.imread('../../log/rgb.png')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the predefined dictionary
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Initialize the detector parameters using default values
    parameters = aruco.DetectorParameters()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(image, dictionary, parameters=parameters)

    # Draw the detected marker corners on the image (if at least one was found)
    if markerCorners:
        aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        # Assuming you are looking for one marker, get the pixel coordinates of its center
        if markerIds is not None and len(markerCorners) > 0:
            # Find the center of the first marker
            center = markerCorners[0].mean(axis=1)[0]

            # You can now use the center coordinates as the location of your end effector
            end_effector_position = (int(center[0]), int(center[1]))
            print(f"End effector is at pixel coordinates: {end_effector_position}")

    # Display the image with the detected marker
    cv2.imshow('Detected ArUco marker', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def object_detection(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the flange
    if contours:
        c = max(contours, key=cv2.contourArea)

        # Determine the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the contour and centroid on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)

    else:
        raise Exception("No flange found in the image.")

    # create a rectangular mask below this flange: the object should within this mask

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[cY+20: cY+220, cX-200:cX+200] = 1
    # filter the image with this mask
    image = cv2.bitwise_and(image, image, mask=mask)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray < 100] = 0

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    # Draw the contour on the mask with white color and filled (-1 means a filled contour)
    cv2.drawContours(mask, [c], -1, color=255, thickness=-1)

    return mask
if __name__ == '__main__':
    # Load the image
    image = cv2.imread('../../log/rgb.png')

    mask = object_detection(image)

    # apply the mask to the image to see the result
    # This will leave only the object, making all other pixels black
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show the mask and the result
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()