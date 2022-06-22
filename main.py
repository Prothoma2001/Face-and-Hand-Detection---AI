import cv2
import glob
import os
import shutil
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)  # to open Camera

currentframe = 0000
condition_success = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# accessing pretrained model
pretrained_model = cv2.CascadeClassifier("face_detector.xml.txt")
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    boolean, frame = capture.read()

    cv2.imwrite(str(currentframe) + ".jpg", frame)
    currentframe += 1
    out.write(frame)

    if boolean == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # drawing rectangle in frame
        for (x, y, w, h) in coordinate_list:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display detected face
        cv2.imshow("Live Face Detection", frame)
        cv2.imshow("Image", img)

    if cv2.waitKey(20) == ord('x'):
        break

capture.release()
out.release()
cv2.destroyAllWindows()


#keeping .jpg and .avi files in separate folders

path = r"C:\PyCharm Projects (main)\Face and Hand Detection\Videos and Frames\Frames\Prothoma" + str(condition_success) + "\Ami" + str(condition_success) + "\g" + str(condition_success) + "\\"
os.makedirs(path + "Video " + str(condition_success))

path2 = r"C:\PyCharm Projects (main)\Face and Hand Detection\Videos and Frames\Videos\Prothoma" + str(condition_success) + "\Ami" + str(condition_success) + "\\"
os.makedirs(path2 + "bg" + str(condition_success))

src_folder = r"C:\PyCharm Projects (main)\Face and Hand Detection"
dst_folder = r"C:\PyCharm Projects (main)\Face and Hand Detection\Videos and Frames\Frames\Prothoma" + str(condition_success) + "\Ami" + str(condition_success)+ "\g" + str(condition_success)
dst_folder2 = r"C:\PyCharm Projects (main)\Face and Hand Detection\Videos and Frames\Videos\Prothoma" + str(condition_success) + "\Ami" + str(condition_success)

pattern = "\*.jpg"
files = glob.glob(src_folder + pattern)

pattern2 = "\*.avi"
files2 = glob.glob(src_folder + pattern2)

for file in files:
    shutil.move(file, os.path.join(dst_folder, "Video " + str(condition_success)))
    print('Moved:', file)

for file in files2:
    shutil.move(file, os.path.join(dst_folder2, "bg" + str(condition_success)))
    print('Moved:', file)



