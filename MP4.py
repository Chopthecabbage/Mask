from imutils import face_utils
import cv2
import dlib

# Load video
vid_location = "samples/g_v2.mp4"
cap = cv2.VideoCapture(vid_location)

# Load our classifiers
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("loading dlib.get_frontal_face_detector()...")
detector = dlib.get_frontal_face_detector()

'''
만약 비디오 프레임의 폭과 높이를 알고 싶다면, cap.get(3), cap.get(4)의 값을 확인하면 됩니다.
만약 새로운 폭과 높이로 설정하기 원한다면 cap.set()함수로 설정하면 됩니다.
예를 들어 프레임의 크기를 320x240 으로 설정하고 싶다면
ret = cap.set(3, 320)
ret = cap.set(4, 240)
'''
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
# Grab the width and height of the input video
# CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = cap.get(5)
print("frame_rate: ", frame_rate)

# Create the video writer for MP4 format
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('effect_v4.mp4', fourcc, frame_rate, (frame_width, frame_height))

face_mask = cv2.imread('samples/mask.png')
h_mask, w_mask = face_mask.shape[:2]

print("기다려주세요......")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame -> gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    # no faces
    if len(faces) == 0:
        print('얼굴 -> X')

    try:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            if h > 0 and w > 0:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                h, w = int(1.4 * h), int(1.2 * w)
                y += 25
                x -= 17

                # face -> frame_roi
                frame_roi = frame[y:y + h, x:x + w]

                # jpg, png -> cv2.resize
                face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
                # face_mask_small -> gray
                gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)

                # background -> black,,, // mask -> white,,,
                ret, mask = cv2.threshold(gray_mask, 244, 255, cv2.THRESH_BINARY_INV)
                # 그 반대 이미지 생성
                mask_inv = cv2.bitwise_not(mask)
                # cv2.threshold + cv2.resize() -> masked_face
                masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
                # frame_roi + mask_inv
                masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
                # add
                frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

                # http://datahacker.rs/003-opencv-projects-how-to-create-instagram-like-filters-mustaches-glasses-and-masks/
    except: { }

    out.write(frame)

print("upgrade complete")
cap.release()
out.release()
cv2.destroyAllWindows()
