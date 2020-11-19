import cv2
import numpy as np
# Dlib, unlike all the other models, works on grayscale images.
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
mask = cv2.imread("data/mask_v4.png")
# ret, frame = cap.read() : 비디오의 한 프레임씩 읽습니다. 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타납니다. fram에 읽은 프레임이 나옵니다
_, frame = cap.read()
# return Y축, X축, 채널의 수
cols, rows, _ = frame.shape
# np.zeros(shape, dtype, order)
mask_overlay = np.zeros((cols, rows), np.uint8)

# dlib에 있는 정면 얼굴 검출기로 입력 사진 img에서 얼굴을 검출 하여 detector로 
detector = dlib.get_frontal_face_detector()
# 랜드마크 찾기 위해서 -> 딥러닝 되어 있는... -> 찾은 얼굴에서 랜드마크를 예측하는 방식으로 되어 있다.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    # == ignored, frame = cap.read()
    # 프레임 별 캡쳐
    _, frame = cap.read()
    mask_overlay.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지 여러개
    faces = detector(frame)
    try:
        for face in faces:
            # left, top // right, bottom
            # print("face:", face)
            landmarks = predictor(gray_frame, face)

            # mask coordinates
            center_mask = (landmarks.part(66).x, landmarks.part(66).y)
            left_mask = (landmarks.part(2).x, landmarks.part(2).y)
            right_mask = (landmarks.part(14).x, landmarks.part(14).y)

            # cv2.circle(img, center, radian, color, thickness)
            # cv2.circle(frame, bottom_mask, 3, (255, 0, 0), -1)

            # double hypot(double x, double y); 직각 삼각형의 빗변의 길이 계산
            mask_width = int(hypot(left_mask[0] - right_mask[0], left_mask[1] - right_mask[1]))
            # .png -> ratio
            mask_height = int(mask_width * 0.9)

            # print("mask_width: ", mask_width)
            # print("mask_height: ", mask_height)

            # New mask position
            top_left_mask = (int(center_mask[0] - mask_width / 2), int(center_mask[1] - mask_height / 2))
            top_right_mask = (int(center_mask[0] + mask_width / 2), int(center_mask[1] + mask_height / 2))

            '''
            cv2.rectangle(img, start, end, color, thickness)
            cv2.rectangle(frame, (int(center_mask[0] - mask_width / 2), int(center_mask[1] - mask_height / 2)),
                                 (int(center_mask[0] + mask_width / 2), int(center_mask[1] + mask_height / 2)),
                                 (0, 255, 0), 2)
            '''
            # Adding the new mask
            # mask_v5.png -> w: 1540, h: 800 -> 800 % 1540 = 0.52
            mask_resize = cv2.resize(mask, (mask_width, mask_height))
            mask_resize_gray = cv2.cvtColor(mask_resize, cv2.COLOR_BGR2GRAY)
            # 이미지 임계처리 -> 문턱 값 이상이면 어떤 값으로 바꿔주고 낮으면 0으로 바꿔주는 기능
            # cv2.threshold(img, threshold_value, value, flag) -> img -> grayScale // threshold_value -> 픽셀 문턱값 // value <- 문턱값 이상이면
            '''
            cv2.THRESH_BINARY: threshold보다 크면 value이고 아니면 0으로 바꾸어 줍니다. 
            cv2.THRESH_BINARY_INV: threshold보다 크면 0이고 아니면 value로 바꾸어 줍니다.   
            cv2.THRESH_TRUNC: threshold보다 크면 value로 지정하고 작으면 기존의 값 그대로 사용한다. 
            cv2.THRESH_TOZERO: treshold_value보다 크면 픽셀 값 그대로 작으면 0으로 할당한다. 
            cv2.THRESH_TOZERO_INV: threshold_value보다 크면 0으로 작으면 그대로 할당해준다. 
            '''
            _, mask_overlay = cv2.threshold(mask_resize_gray, 25, 255, cv2.THRESH_BINARY_INV)

            # 슬라이싱
            mask_area = frame[top_left_mask[1]: top_left_mask[1] + mask_height, top_left_mask[0]: top_left_mask[0] + mask_width]
            mask_area_else = cv2.bitwise_and(mask_area, mask_area, mask=mask_overlay)
            final_mask = cv2.add(mask_area_else, mask_resize)
            frame[top_left_mask[1]: top_left_mask[1] + mask_height, top_left_mask[0]: top_left_mask[0] + mask_width] = final_mask
            cv2.imshow("mask_area_else", mask_area_else)
            cv2.imshow("mask_area", mask_area)

            # cv2.imshow("mask", mask)
            cv2.imshow("mask_resize", mask_resize)
            cv2.imshow("mask_overlay", mask_overlay)
            cv2.imshow("final mask", final_mask)
    except: {}

    cv2.imshow("overlaid", frame)

    key = cv2.waitKey(1)
    # key -> Esc
    if key == 27:
        break

'''        
# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))
'''
