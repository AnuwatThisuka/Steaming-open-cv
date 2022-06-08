from flask import Flask, render_template, Response
import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def center_point(x, y, w, h):
    ww = int(w/2)
    hh = int(h/2)
    cx = int(x+ww)
    cy = int(y+hh)
    return cx, cy


ret, img = cap.read()
counter_line = 50
each_line = 50
text_color = (0, 255, 0)
object_color = (0, 0, 255)
blurr = 10
bord = 10
error = 100
detect_line = []
counter = 0
state = 0
bol = True


# lower color detection
h_lower = 94
s_lower = 36
v_lower = 134
# upper color detection
h_upper = 179
s_upper = 255
v_upper = 255

# fucntion to detect color


while(1):
    ret, img = cap.read()
    blur = cv2.blur(img, (25, 25))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower = np.array([h_lower, s_lower, v_lower], np.uint8)
    upper = np.array([h_upper, s_upper, v_upper], np.uint8)

    blue = cv2.inRange(hsv, lower, upper)
    kernal = np.ones((5, 5), "uint8")
    blue = cv2.dilate(blue, kernal)
    res_blue = cv2.bitwise_and(img, img, mask=blue)
    # Show line detection
    contours, hierarchy = cv2.findContours(
        blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # functin to detect the object
        if(area > 10000):
            x, y, w, h = cv2.boundingRect(contour)
            center = center_point(x, y, w, h)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            img = cv2.rectangle(
                img, (x, y), (x + w, y + h), object_color, 2)
            detect_line.append(center)
            for x, y in detect_line:
                if (x and state == 0):
                    state = 1
                if (x and x > counter_line and state == 1):
                    if (bol):
                        counter_line += each_line
                        bol = False
                        state = 0
                        counter += 1
                        print("CounterUp")
                detect_line.remove((x, y))
    cv2.putText(img, str(counter), (550, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3, cv2.LINE_AA)
    cv2.imshow("IIOT-B14 Camera Tracking Counter", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
