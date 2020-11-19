from imutils import face_utils
import time
import cv2
import dlib

# scaler = 1.0

# 인식력이 안좋음... -> shape_predictor_68_face_landmarks.dat -> 학습된 모델
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# dlib -> 얼굴을 가져 오는 코드가 매우 간단하기 때문에 얼굴 감지 작업
print("loading dlib.get_frontal_face_detector()...")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("camera sensor warming up...")
cap = cv2.VideoCapture(1) # webcame video
# 동영상 가능 -> scaler -> 조정 -> 프레임 속도 문제는 어쩔 수 없나 보다...
# cap = cv2.VideoCapture('samples/g_v2.mp4')
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(2.0)

# cv2.IMREAD_UNCHANGED == -1
mask_image_original = cv2.imread('samples/mask_v4.png', -1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# frame_width = int(cap.get(3) * scaler)
# frame_height = int(cap.get(4) * scaler)
frame_rate = cap.get(5)
print("frame_rate:", frame_rate)

# Create the video writer for MP4 format
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('effect_v3.mp4', fourcc, frame_rate, (frame_width, frame_height))

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

print("기다려주세요......")
while 1:
    ret, img = cap.read()
    # 동영상일 경우
    if not ret:
        break
    # resize
    # img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5) // default
    # faces = face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350)) // 검출 최소, 검출 최대
    faces = detector(gray, 0)

    # no faces
    if len(faces) == 0:
        print('얼굴 -> X')

    try:
        # for (x, y, w, h) in faces:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            if h > 0 and w > 0:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # 마스크 위치 설정
                mask_position_min = int(y + 1.7 * h / 5)
                mask_position_max = int(y + 5.7 * h / 5)
                mask_height = mask_position_max - mask_position_min

                mask_roi = img[mask_position_min:mask_position_max, x:x + w]
                '''
                interplolation: 리사이징을 수행할 때 적용할 interpolation 방법
                INTER_NEAREST: nearest-neighbor interpolation
                INTER_LINEAR: bilinear interpolation (디폴트 값)
                INTER_AREA: 픽셀 영역 관계를 이용한 resampling 방법으로 이미지 축소에 있어 선호되는 방법. 이미지를 확대하는 경우에는 INTER_NEAREST와 비슷한 효과를 보임
                INTER_CUBIC: 4x4 픽셀에 적용되는 bicubic interpolation
                INTER_LANCZOS4 : 8x8 픽셀에 적용되는 Lanczos interpolation
                '''
                specs = cv2.resize(mask_image_original, (w, mask_height), interpolation=cv2.INTER_CUBIC)

                transparentOverlay(mask_roi, specs)
    except: {}

    out.write(img)

    cv2.imshow('effect_v3', img)

    # 64bit -> & 0xff
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        cv2.imwrite('effect_v3.jpg', img)
        break

print("upgrade complete")
# cap 객체를 해제
cap.release()
# out 객체를 해제
out.release()
# 생성한 모든 윈도우 제거
cv2.destroyAllWindows()