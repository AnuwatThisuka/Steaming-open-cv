from itertools import count
import re
from flask import Flask, render_template, Response
import cv2
import numpy as np
from matplotlib.pyplot import gray

app = Flask(__name__)

cap = cv2.VideoCapture(0)

# def rescale_frame(img, percent=75):
# width = int(img.shape[1] * percent / 100)
# height = int(img.shape[0] * percent / 100)
# dim = (width, height)
# return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

lower = np.array([94, 36, 134], np.uint8)
upper = np.array([179, 255, 255], np.uint8)


def center_point(x, y, w, h):
    ww = int(w/2)
    hh = int(h/2)
    cx = int(x+ww)
    cy = int(y+hh)
    return cx, cy


def x_line(img, percent):
    posit = int((img.shape[1]*percent)/100)
    left_line = posit - each_line
    right_line = posit + each_line
    return posit, left_line, right_line


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
# lower color detection
h_lower = 94
s_lower = 36
v_lower = 134
# upper color detection
h_upper = 179
s_upper = 255
v_upper = 255
left_line = x_line(img, counter_line)[1] - each_line
count = 0


def gen_frames():
    state = 0

    def countfunc(count):
        global counter
        counter = counter + 1
        return counter
    while(1):
        ret, img = cap.read()
        blur = cv2.blur(img, (25, 25))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower = np.array([h_lower, s_lower, v_lower], np.uint8)
        upper = np.array([h_upper, s_upper, v_upper], np.uint8)

        blue = cv2.inRange(hsv, lower, upper)
        kernal = np.ones((5, 5), "uint8")
        blue = cv2.dilate(blue, kernal)
        res_blue = cv2.bitwise_and(img, img, mask=blue)
        cv2.line(img, (x_line(img, counter_line)[0], bord), (x_line(
            img, counter_line)[0], img.shape[0]-bord), (255, 0, 0), 2)
        cv2.line(img, (x_line(img, counter_line)[1], bord), (x_line(
            img, counter_line)[1], img.shape[0]-bord), (255, 0, 0), 2)
        cv2.line(img, (x_line(img, counter_line)[2], bord), (x_line(
            img, counter_line)[2], img.shape[0]-bord), (255, 0, 0), 2)
        contours, hierarchy = cv2.findContours(
            blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if(area > 10000):
                x, y, w, h = cv2.boundingRect(contour)
                center = center_point(x, y, w, h)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                img = cv2.rectangle(
                    img, (x, y), (x + w, y + h), object_color, 2)
                detect_line.append(center)
                for x, y in detect_line:
                    if (x < x_line(img, counter_line)[2] and x > x_line(img, counter_line)[0] and state == 0):
                        state = 1
                    if (x < x_line(img, counter_line)[2] and x > x_line(img, counter_line)[1]-20):
                        cv2.line(img, (x_line(img, counter_line)[1], bord), (x_line(
                            img, counter_line)[1], img.shape[0]-bord), (0, 255, 0), 2)
                        cv2.line(img, (x_line(img, counter_line)[2], bord), (x_line(
                            img, counter_line)[2], img.shape[0]-bord), (0, 255, 0), 2)
                        if (x < x_line(img, counter_line)[1]) and state == 1:
                            state = 0
                            countfunc(count)
                            #print("Counter: ", counfunc(count))
                            cv2.line(img, (x_line(img, counter_line)[0], bord), (x_line(
                                img, counter_line)[0], img.shape[0]-bord), (0, 250, 0), 5)
                    detect_line.remove((x, y))
        cv2.putText(img, str(counter), (550, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3, cv2.LINE_AA)
        # cv2.imshow("IIOT-B14 Camera Tracking Counter", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--img\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@ app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=img')


@ app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=4001)
