import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QCheckBox, QSlider, QTabWidget,
                             QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
                             QTextEdit, QSplitter, QMessageBox, QStyleFactory)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ----- Modèles -----
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Charger les réseaux de neurones
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Configurer pour utiliser GPU si disponible
try:
    faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU acceleration")
except:
    print("Using CPU")

padding = 20


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class AdvancedApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Age & Gender Detection PRO")
        self.setGeometry(100, 100, 1400, 900)

        # Initialiser le timer avant setup_ui
        self.timer = QTimer()
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()

        # Variables pour les statistiques
        self.age_stats = {age: 0 for age in ageList}
        self.gender_stats = {gender: 0 for gender in genderList}
        self.detection_history = []

        # Configuration de l'interface
        self.setup_ui()
        self.apply_styles()

        # Initialiser le graphique
        self.update_charts()

    def setup_ui(self):
        # Configuration des onglets
        self.tabs = QTabWidget()
        self.tabDetect = QWidget()
        self.tabOptions = QWidget()
        self.tabStats = QWidget()
        self.tabAbout = QWidget()

        self.tabs.addTab(self.tabDetect, "Détection")
        self.tabs.addTab(self.tabOptions, "Options")
        self.tabs.addTab(self.tabStats, "Statistiques")
        self.tabs.addTab(self.tabAbout, "À propos")

        # --- Onglet Détection ---
        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(640, 480)
        self.imageLabel.setText("Charger une image ou démarrer la webcam")
        self.imageLabel.setStyleSheet("border: 2px dashed #ccc; padding: 20px;")

        # Boutons de contrôle
        self.startButton = QPushButton("Démarrer Webcam")
        self.stopButton = QPushButton("Arrêter Webcam")
        self.loadButton = QPushButton("Charger Image")
        self.processButton = QPushButton("Traiter Image")
        self.saveButton = QPushButton("Sauvegarder")

        # Barre de progression
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)

        # Disposition des boutons
        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.startButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(self.loadButton)
        controlLayout.addWidget(self.processButton)
        controlLayout.addWidget(self.saveButton)

        # Layout principal de l'onglet Détection
        layoutDetect = QVBoxLayout()
        layoutDetect.addWidget(self.imageLabel, 4)
        layoutDetect.addLayout(controlLayout, 1)
        layoutDetect.addWidget(self.progressBar)
        self.tabDetect.setLayout(layoutDetect)

        # --- Onglet Options ---
        optionsLayout = QVBoxLayout()

        # Groupe de paramètres de détection
        detectionGroup = QGroupBox("Paramètres de Détection")
        detectionLayout = QVBoxLayout()

        self.confSlider = QSlider(Qt.Horizontal)
        self.confSlider.setMinimum(50)
        self.confSlider.setMaximum(100)
        self.confSlider.setValue(70)
        self.confLabel = QLabel("Seuil de confiance: 0.7")

        self.paddingSpin = QSpinBox()
        self.paddingSpin.setMinimum(0)
        self.paddingSpin.setMaximum(100)
        self.paddingSpin.setValue(20)
        self.paddingSpin.setSuffix(" px")

        detectionLayout.addWidget(QLabel("Seuil de confiance:"))
        detectionLayout.addWidget(self.confSlider)
        detectionLayout.addWidget(self.confLabel)
        detectionLayout.addWidget(QLabel("Padding autour des visages:"))
        detectionLayout.addWidget(self.paddingSpin)
        detectionGroup.setLayout(detectionLayout)

        # Groupe d'effets
        effectsGroup = QGroupBox("Effets et Affichage")
        effectsLayout = QVBoxLayout()

        self.blurCheck = QCheckBox("Flouter visages")
        self.pixelCheck = QCheckBox("Pixeliser visages")
        self.fpsCheck = QCheckBox("Afficher FPS")
        self.saveCheck = QCheckBox("Sauvegarder automatiquement")
        self.rectStyleCombo = QComboBox()
        self.rectStyleCombo.addItems(["Rectangle plein", "Rectangle contour", "Rectangle arrondi"])

        effectsLayout.addWidget(self.blurCheck)
        effectsLayout.addWidget(self.pixelCheck)
        effectsLayout.addWidget(self.fpsCheck)
        effectsLayout.addWidget(self.saveCheck)
        effectsLayout.addWidget(QLabel("Style de rectangle:"))
        effectsLayout.addWidget(self.rectStyleCombo)
        effectsGroup.setLayout(effectsLayout)

        # Groupe de performances
        perfGroup = QGroupBox("Performances")
        perfLayout = QVBoxLayout()

        self.resizeSpin = QSpinBox()
        self.resizeSpin.setMinimum(100)
        self.resizeSpin.setMaximum(1000)
        self.resizeSpin.setValue(500)
        self.resizeSpin.setSuffix(" px")
        self.resizeSpin.setToolTip("Redimensionner l'image pour un traitement plus rapide")

        self.gpuCheck = QCheckBox("Utiliser l'accélération GPU")
        self.gpuCheck.setChecked(True)

        perfLayout.addWidget(QLabel("Largeur max pour le traitement:"))
        perfLayout.addWidget(self.resizeSpin)
        perfLayout.addWidget(self.gpuCheck)
        perfGroup.setLayout(perfLayout)

        optionsLayout.addWidget(detectionGroup)
        optionsLayout.addWidget(effectsGroup)
        optionsLayout.addWidget(perfGroup)
        optionsLayout.addStretch()
        self.tabOptions.setLayout(optionsLayout)

        # --- Onglet Statistiques ---
        statsLayout = QHBoxLayout()

        # Graphiques
        self.canvasAge = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvasGender = MplCanvas(self, width=5, height=4, dpi=100)

        # Historique des détections
        self.historyText = QTextEdit()
        self.historyText.setReadOnly(True)

        # Splitter pour diviser l'espace
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvasAge)
        splitter.addWidget(self.canvasGender)
        splitter.addWidget(self.historyText)
        splitter.setSizes([300, 300, 400])

        statsLayout.addWidget(splitter)
        self.tabStats.setLayout(statsLayout)

        # --- Onglet À propos ---
        aboutLayout = QVBoxLayout()
        aboutText = QTextEdit()
        aboutText.setReadOnly(True)
        aboutText.setHtml("""
        <h1>Age & Gender Detection PRO</h1>
        <p>Application avancée de détection d'âge et de genre utilisant l'apprentissage profond.</p>
        <h2>Fonctionnalités:</h2>
        <ul>
            <li>Détection de visages en temps réel</li>
            <li>Estimation de l'âge et du genre</li>
            <li>Options avancées de traitement d'image</li>
            <li>Statistiques et visualisation des données</li>
            <li>Support GPU pour l'accélération</li>
        </ul>
        <h2>Modèles utilisés:</h2>
        <ul>
            <li>Détection de visages: OpenCV Face Detector</li>
            <li>Estimation d'âge: Caffe Age Network</li>
            <li>Estimation de genre: Caffe Gender Network</li>
        </ul>
        <p>Version: 2.0.0</p>
        """)
        aboutLayout.addWidget(aboutText)
        self.tabAbout.setLayout(aboutLayout)

        # Layout principal
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.tabs)
        self.setLayout(mainLayout)

        # Connexions
        self.setup_connections()

        # Initialisation
        self.stopButton.setEnabled(False)
        self.processButton.setEnabled(False)
        self.saveButton.setEnabled(False)

    def setup_connections(self):
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.loadButton.clicked.connect(self.load_image)
        self.processButton.clicked.connect(self.process_image)
        self.saveButton.clicked.connect(self.save_result)
        self.timer.timeout.connect(self.update_frame)
        self.confSlider.valueChanged.connect(self.update_conf_label)
        self.gpuCheck.stateChanged.connect(self.toggle_gpu)

    def apply_styles(self):
        # Application d'un style moderne
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Palette de couleurs moderne
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

        # Style des boutons
        button_style = """
        QPushButton {
            background-color: #2b2b2b;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3a3a3a;
        }
        QPushButton:pressed {
            background-color: #4a4a4a;
        }
        QPushButton:disabled {
            background-color: #2b2b2b;
            color: #6a6a6a;
        }
        """

        self.startButton.setStyleSheet(button_style + "color: #4CAF50;")
        self.stopButton.setStyleSheet(button_style + "color: #F44336;")
        self.loadButton.setStyleSheet(button_style + "color: #2196F3;")
        self.processButton.setStyleSheet(button_style + "color: #FF9800;")
        self.saveButton.setStyleSheet(button_style + "color: #E91E63;")

        # Style des onglets
        self.tabs.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #444;
            top: -1px;
            background: #2b2b2b;
        }
        QTabBar::tab {
            background: #353535;
            color: #ddd;
            padding: 8px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            border: 1px solid #444;
        }
        QTabBar::tab:selected {
            background: #2b2b2b;
            border-bottom-color: #2b2b2b;
        }
        QTabBar::tab:hover:!selected {
            background: #3a3a3a;
        }
        """)

        # Style des groupes
        group_style = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #444;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 15px;
            color: #ddd;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        """

        for group in self.tabOptions.findChildren(QGroupBox):
            group.setStyleSheet(group_style)

    def update_conf_label(self):
        self.confLabel.setText(f"Seuil de confiance: {self.confSlider.value() / 100:.2f}")

    def toggle_gpu(self):
        if self.gpuCheck.isChecked():
            try:
                faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using GPU acceleration")
            except:
                print("GPU not available, using CPU")
                self.gpuCheck.setChecked(False)
        else:
            faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU")

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Erreur", "Impossible d'accéder à la webcam")
            return

        self.frame_count = 0
        self.start_time = time.time()
        self.timer.start(30)
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.processButton.setEnabled(False)
        self.imageLabel.setText("Webcam active...")

    def stop_webcam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.processButton.setEnabled(True)
        self.imageLabel.setText("Webcam arrêtée")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.current_image_path = path
            self.original_image = cv2.imread(path)
            self.display_image(self.original_image)
            self.processButton.setEnabled(True)
            self.saveButton.setEnabled(False)

    def process_image(self):
        if hasattr(self, 'original_image'):
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)

            # Simuler une progression (dans une application réelle, cela serait basé sur l'avancement réel)
            for i in range(101):
                time.sleep(0.02)  # Simulation de traitement
                self.progressBar.setValue(i)
                QApplication.processEvents()

            self.update_frame(process_static=True)
            self.progressBar.setVisible(False)
            self.saveButton.setEnabled(True)

    def save_result(self):
        if hasattr(self, 'processed_image'):
            path, _ = QFileDialog.getSaveFileName(
                self, "Sauvegarder l'image", "",
                "Images (*.png *.jpg *.jpeg)"
            )
            if path:
                cv2.imwrite(path, self.processed_image)
                QMessageBox.information(self, "Succès", "Image sauvegardée avec succès")

    def update_frame(self, process_static=False):
        if self.cap or process_static:
            if process_static and hasattr(self, 'original_image'):
                frame = self.original_image.copy()
            elif self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_webcam()
                    return
            else:
                return

            # Redimensionner pour les performances si nécessaire
            max_width = self.resizeSpin.value()
            if frame.shape[1] > max_width:
                ratio = max_width / frame.shape[1]
                new_height = int(frame.shape[0] * ratio)
                frame = cv2.resize(frame, (max_width, new_height))

            conf_threshold = self.confSlider.value() / 100
            padding = self.paddingSpin.value()

            resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold)

            # Statistiques pour cette frame
            age_counts = {age: 0 for age in ageList}
            gender_counts = {gender: 0 for gender in genderList}
            detection_time = time.strftime("%Y-%m-%d %H:%M:%S")

            for faceBox in faceBoxes:
                face = frame[
                       max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                       max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)
                       ]

                if face.size == 0:
                    continue

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]

                # Appliquer le style de rectangle sélectionné
                rect_style = self.rectStyleCombo.currentIndex()
                if rect_style == 0:  # Plein
                    cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), -1)
                elif rect_style == 1:  # Contour
                    cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), 2)
                else:  # Arrondi
                    # Implémentation simplifiée de rectangles arrondis
                    radius = 20
                    color = (0, 255, 0)
                    thickness = 2
                    # Points pour dessiner des arcs
                    cv2.ellipse(resultImg, (faceBox[0] + radius, faceBox[1] + radius), (radius, radius), 180, 0, 90,
                                color, thickness)
                    cv2.ellipse(resultImg, (faceBox[2] - radius, faceBox[1] + radius), (radius, radius), 270, 0, 90,
                                color, thickness)
                    cv2.ellipse(resultImg, (faceBox[0] + radius, faceBox[3] - radius), (radius, radius), 90, 0, 90,
                                color, thickness)
                    cv2.ellipse(resultImg, (faceBox[2] - radius, faceBox[3] - radius), (radius, radius), 0, 0, 90,
                                color, thickness)
                    # Lignes droites
                    cv2.line(resultImg, (faceBox[0] + radius, faceBox[1]), (faceBox[2] - radius, faceBox[1]), color,
                             thickness)
                    cv2.line(resultImg, (faceBox[0] + radius, faceBox[3]), (faceBox[2] - radius, faceBox[3]), color,
                             thickness)
                    cv2.line(resultImg, (faceBox[0], faceBox[1] + radius), (faceBox[0], faceBox[3] - radius), color,
                             thickness)
                    cv2.line(resultImg, (faceBox[2], faceBox[1] + radius), (faceBox[2], faceBox[3] - radius), color,
                             thickness)

                label = f"{gender}, {age}"
                cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Appliquer les effets si sélectionnés
                if self.blurCheck.isChecked():
                    resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]] = cv2.GaussianBlur(
                        resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]], (99, 99), 30)
                elif self.pixelCheck.isChecked():
                    face_region = resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]]
                    h, w = face_region.shape[:2]
                    temp = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
                    resultImg[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]] = cv2.resize(temp, (w, h),
                                                                                         interpolation=cv2.INTER_NEAREST)

                # Mettre à jour les statistiques
                age_counts[age] += 1
                gender_counts[gender] += 1

                # Ajouter à l'historique
                self.detection_history.append({
                    'time': detection_time,
                    'gender': gender,
                    'age': age,
                    'confidence': max(genderPreds[0].max(), agePreds[0].max())
                })

            # Mettre à jour les statistiques globales
            for age, count in age_counts.items():
                if count > 0:
                    self.age_stats[age] += count

            for gender, count in gender_counts.items():
                if count > 0:
                    self.gender_stats[gender] += count

            # Afficher les FPS si demandé
            if self.fpsCheck.isChecked() and not process_static:
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(resultImg, f"FPS: {fps:.2f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Sauvegarder automatiquement si demandé
            if self.saveCheck.isChecked() and not process_static:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"detection_{timestamp}.jpg", resultImg)

            # Stocker l'image traitée
            self.processed_image = resultImg.copy()

            # Afficher l'image
            self.display_image(resultImg)

            # Mettre à jour les graphiques
            self.update_charts()

            # Mettre à jour l'historique texte
            self.update_history_text()

    def display_image(self, image):
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qtImg = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.imageLabel.setPixmap(QPixmap.fromImage(qtImg).scaled(
            self.imageLabel.width(),
            self.imageLabel.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def update_charts(self):
        # Graphique des âges
        self.canvasAge.axes.clear()
        ages = list(self.age_stats.keys())
        counts = list(self.age_stats.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(ages)))
        self.canvasAge.axes.bar(ages, counts, color=colors)
        self.canvasAge.axes.set_title('Répartition par Âge')
        self.canvasAge.axes.tick_params(axis='x', rotation=45)
        self.canvasAge.draw()

        # Graphique des genres
        self.canvasGender.axes.clear()
        genders = list(self.gender_stats.keys())
        counts = list(self.gender_stats.values())
        colors = ['lightblue', 'lightpink']
        self.canvasGender.axes.pie(counts, labels=genders, autopct='%1.1f%%', colors=colors)
        self.canvasGender.axes.set_title('Répartition par Genre')
        self.canvasGender.draw()

    def update_history_text(self):
        text = "Historique des détections:\n\n"
        for detection in self.detection_history[-10:]:  # Afficher les 10 dernières détections
            text += f"{detection['time']} - {detection['gender']}, {detection['age']} (conf: {detection['confidence']:.2f})\n"
        self.historyText.setPlainText(text)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


# ----- Lancer l'application -----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Age & Gender Detection PRO")
    app.setApplicationVersion("2.0.0")

    window = AdvancedApp()
    window.show()
    sys.exit(app.exec_())