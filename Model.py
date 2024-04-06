#model
import cv2
import numpy as np
import mediapipe as mp
import Controller as controller
class Model:
    def __init__(self,path):
        self.path=path
        self.facing=[]
        self.faces=[]
        
        
    def cheking_function(self):
        frame_count = 1
        cap = cv2.VideoCapture(self.path)
        lst, grp, a, em = [], [], 0, []
        lasting_frames = 5
        prev_gaze = None
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results_mesh = face_mesh.process(image)
            image.flags.writeable = True

            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_face = face_detection.process(frame_rgb)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            face_2d.append([x, y])

                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -10:
                        text = "Looking Down"
                    elif x > 10:
                        text = "Looking Up"
                    else:
                        text = "Forward"

                    if prev_gaze != text:
                        if text != "Forward":
                            controller.image_screenshot(image,frame_count)
                            frame=controller.frame2time(frame_count)
                            self.facing.append(f"not facing the cam:, {frame} ")  # not facing cam

                    prev_gaze = text

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

                # ------------------------------------------------------------------------------------------------- #

                if results_face.detections:
                    for detection in results_face.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = image.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(image, bbox, (0, 0, 255), 2)

                    if len(results_face.detections) < 2:
                        lst.append(0)
                    else:
                        lst.append(1)
                        controller.image_screenshot(image,frame_count)
                        frame=controller.frame2time(frame_count)
                        self.faces.append(f"multiple-faces: , {frame}")  # multi

            cv2.imshow('Head Pose Estimation', image)
            frame_count += 1
            # pr_tm = tm
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        for i in range(len(lst)):
            if lst[i] == lst[i - 1] and lst[i] == 1:
                a += 1
            if lst[i] != lst[i - 1]:
                if a > lasting_frames:
                    grp.append(1)
                else:
                    grp.append(0)
                a = 0
        return grp,grp.count(1),self.faces,self.facing
