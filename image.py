from imutils import face_utils
import cv2
import dlib

# 인식력이 안좋음... -> shape_predictor_68_face_landmarks.dat -> 학습된 모델
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# dlib -> 얼굴을 가져 오는 코드가 매우 간단하기 때문에 얼굴 감지 작업
print("loading dlib.get_frontal_face_detector()...")
# dlib에 있는 정면 얼굴 검출기로 입력 사진 img에서 얼굴을 검출하여 detector로 반환
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# cv2.IMREAD_UNCHANGED == -1
mask_image_original = cv2.imread('samples/mask_v2.png', -1)

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

# img = dlib.load_rgb_image('samples/twice.jpg')
img = cv2.imread('samples/twice.jpg')
faces = detector(img, 1)
cv2.imshow('effect_v5', img)

for face in faces:
    (x, y, w, h) = face_utils.rect_to_bb(face)
    if h > 0 and w > 0:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # 마스크 위치 설정
        mask_position_min = int(y + 1.6 * h / 5)
        mask_position_max = int(y + 5.3 * h / 5)
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

k = cv2.waitKey(0)  # 키보드 눌림 대기
if k == 27: # ESC키
    cv2.destroyAllWindows()
elif k == ord('s'): # 저장하기 버튼
    cv2.imwrite("effect_v5.jpg", img)

print("upgrade complete")
# 생성한 모든 윈도우 제거
cv2.destroyAllWindows()
