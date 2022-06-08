from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([94, 36, 134])
upper = np.array([179, 255, 255])  # (These ranges will detect Yellow)

camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(
#    'rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert to hsv format
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(frame, lower, upper)
        # Bitwise-AND mask and original image
        mask_contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in mask image
        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > 500:
                    x, y, w, h = cv2.boundingRect(mask_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 3)  # drawing rectangle
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


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
