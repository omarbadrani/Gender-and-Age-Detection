import sys
import cv2
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QCheckBox, QSlider, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer

# ----- Modèles -----
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
padding = 20

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300,300), [104,117,123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),2)
    return frameOpencvDnn, faceBoxes

# ----- Interface améliorée -----
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Age & Gender Detection PRO")
        self.setGeometry(100,100,1000,700)

        # Onglets
        self.tabs = QTabWidget()
        self.tabDetect = QWidget()
        self.tabOptions = QWidget()
        self.tabStats = QWidget()
        self.tabs.addTab(self.tabDetect, "Détection")
        self.tabs.addTab(self.tabOptions, "Options")
        self.tabs.addTab(self.tabStats, "Statistiques")

        # --- Onglet Détection ---
        self.imageLabel = QLabel()
        self.startButton = QPushButton("Webcam")
        self.loadButton = QPushButton("Charger Image")
        self.startButton.setStyleSheet("background-color: #4CAF50; color: white; font-weight:bold")
        self.loadButton.setStyleSheet("background-color: #2196F3; color: white; font-weight:bold")

        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.startButton)
        btnLayout.addWidget(self.loadButton)
        layoutDetect = QVBoxLayout()
        layoutDetect.addWidget(self.imageLabel)
        layoutDetect.addLayout(btnLayout)
        self.tabDetect.setLayout(layoutDetect)

        # --- Onglet Options ---
        self.blurCheck = QCheckBox("Flouter visages")
        self.pixelCheck = QCheckBox("Pixeliser visages")
        self.fpsCheck = QCheckBox("Afficher FPS")
        self.saveCheck = QCheckBox("Sauvegarder résultat")
        self.confSlider = QSlider()
        self.confSlider.setMinimum(50)
        self.confSlider.setMaximum(100)
        self.confSlider.setValue(70)
        self.confSlider.setOrientation(1)
        layoutOptions = QVBoxLayout()
        layoutOptions.addWidget(self.blurCheck)
        layoutOptions.addWidget(self.pixelCheck)
        layoutOptions.addWidget(self.fpsCheck)
        layoutOptions.addWidget(self.saveCheck)
        layoutOptions.addWidget(QLabel("Seuil confiance"))
        layoutOptions.addWidget(self.confSlider)
        self.tabOptions.setLayout(layoutOptions)

        # --- Onglet Statistiques ---
        self.statsLabel = QLabel("Statistiques s'afficheront ici")
        self.statsLabel.setFont(QFont("Arial", 12))
        self.statsLabel.setStyleSheet("color: #FF5722;")
        layoutStats = QVBoxLayout()
        layoutStats.addWidget(self.statsLabel)
        self.tabStats.setLayout(layoutStats)

        # Layout principal
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.tabs)
        self.setLayout(mainLayout)

        # Connexions
        self.startButton.clicked.connect(self.startWebcam)
        self.loadButton.clicked.connect(self.loadImage)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

        self.frame_count = 0
        self.start_time = time.time()
        self.current_stats = ""

    # --- Fonctions ---
    def startWebcam(self):
        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        self.start_time = time.time()
        self.timer.start(30)

    def loadImage(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.cap = None
            self.image = cv2.imread(path)
            self.frame_count = 0
            self.start_time = time.time()
            self.timer.start(30)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return
        elif hasattr(self, 'image'):
            frame = self.image.copy()
        else:
            return

        conf_threshold = self.confSlider.value()/100
        resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold)

        # Statistiques
        age_counts = {age:0 for age in ageList}
        gender_counts = {g:0 for g in genderList}

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]
            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

            if self.blurCheck.isChecked():
                resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]] = cv2.GaussianBlur(
                    resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]], (99,99), 30)
            elif self.pixelCheck.isChecked():
                face_region = resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]]
                h, w = face_region.shape[:2]
                temp = cv2.resize(face_region, (16,16), interpolation=cv2.INTER_LINEAR)
                resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]] = cv2.resize(temp, (w,h), interpolation=cv2.INTER_NEAREST)

            age_counts[age] += 1
            gender_counts[gender] += 1

        # Affichage statistiques
        stats_text = ""
        for age, count in age_counts.items():
            if count>0:
                stats_text += f"{age}: {count}  "
        for gender, count in gender_counts.items():
            if count>0:
                stats_text += f"{gender}: {count}  "
        self.statsLabel.setText(stats_text)

        # FPS
        self.frame_count += 1
        if self.fpsCheck.isChecked():
            fps = self.frame_count / (time.time() - self.start_time + 0.001)
            cv2.putText(resultImg, f"FPS: {fps:.2f}", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Convertir pour PyQt
        rgbImage = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qtImg = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.imageLabel.setPixmap(QPixmap.fromImage(qtImg))

        if self.saveCheck.isChecked():
            cv2.imwrite("result.jpg", resultImg)

# ----- Lancer l'application -----
app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec_())
