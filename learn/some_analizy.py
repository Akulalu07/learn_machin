import cv2
import mediapipe as mp
import numpy as np

# Инициализация MediaPipe для рук, лица и тела
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Открытие камеры
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

# Параметры камеры
w, h = 640, 480


def main():
    mode = 'hands'  # Начальный режим

    # Инициализация распознавания рук, лица и тела
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands, \
            mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.8,
                                  min_tracking_confidence=0.7) as face_mesh, \
            mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.7) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: не удалось захватить кадр.")
                break

            # Преобразование изображения в RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Улучшение качества изображения (например, повышение резкости)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.addWeighted(frame, 1.5, frame, 0, -100)

            # Обработка рук
            if mode == 'hands':
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=6),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
                        )

            # Обработка лица
            elif mode == 'face':
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Преобразуем координаты точек лица в массив для рисования маски
                        points = []
                        for landmark in face_landmarks.landmark:
                            points.append((int(landmark.x * w), int(landmark.y * h)))

                        # Рисуем черную маску на лице
                        points = np.array(points, dtype=np.int32)
                        cv2.fillPoly(frame, [points], (0, 0, 0))  # Черная маска

                        mp_drawing.draw_landmarks(
                            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=0),
                            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1)
                        )

            # Обработка тела
            if mode == 'body':
                results = pose.process(image)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=6),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
                    )

            # Восстановление флага записи
            image.flags.writeable = True

            # Отображение изображения с наложенной разметкой
            cv2.imshow('MediaPipe Detection', frame)

            # Обработка нажатий клавиш
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                mode = 'hands'
                print("Переключено на режим: Руки")
            elif key == ord('f'):
                mode = 'face'
                print("Переключено на режим: Лицо")
            elif key == ord('b'):
                mode = 'body'
                print("Переключено на режим: Тело")

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
