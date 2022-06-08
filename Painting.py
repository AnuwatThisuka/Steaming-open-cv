from flask import Flask, render_template, Response
import cv2
from matplotlib.pyplot import gray
import numpy as np
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(
#    'rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

app = Flask(__name__)


def center_point(x, y, w, h):
    ww = int(w/2)
    hh = int(h/2)
    cx = int(x+ww)
    cy = int(y+hh)
    return cx, cy


def x_line(img, percent):
    #posit = int((img.shape[1]*percent)/100)
    left_line = 20 - each_line
    right_line = 20 + each_line
    return posit, left_line, right_line


ret, img = cap.read()
counter_line = 50
each_line = 50
#left_line_second = x_line(img, counter_line)[1] - each_line
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


def gen_frames():
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
#-------------------------------------------------------------------------------#
        # Show line detection
        # cv2.line(img, (x_line(img, counter_line)[0], bord), (x_line(
        # img, counter_line)[0], img.shape[0]-bord), (255, 0, 0), 2)
        # cv2.line(img, (x_line(img, counter_line)[1], bord), (x_line(
        # img, counter_line)[1], img.shape[0]-bord), (255, 0, 0), 2)
        # cv2.line(img, (x_line(img, counter_line)[2], bord), (x_line(
        # img, counter_line)[2], img.shape[0]-bord), (255, 0, 0), 2)
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
                    if (x < x_line(img, counter_line)[2] and x > x_line(img, counter_line)[0] and state == 0):
                        state = 1
                    if (x < x_line(img, counter_line)[2] and x > x_line(img, counter_line)[1]-20):
                        if (x < x_line(img, counter_line)[1]) and state == 1:
                            state = 0
                            counter += 1
                            print("CounterUp")


@ app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@ app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
