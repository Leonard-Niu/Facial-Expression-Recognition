#-------------------------------
# written by Leonard Niu
# HIT
#-------------------------------
import cv2

CASC_PATH = './utils/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def face_d(image):

    faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
    )
    if not len(faces) > 0:
        print ("-----------------------------------")
        print ("Can not detect any face information")
        print ("-----------------------------------")
        return None, None
    else:
        max_face = faces[0]
        for face in faces:
            if face[2] * face[3] > max_face[2] * max_face[3]:
                max_face = face
        face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]

    return face_image, max_face