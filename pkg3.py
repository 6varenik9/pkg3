import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QWidget, QComboBox, QSpinBox, QFrame, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1000, 700)

        self.original_image = None
        self.processed_image = None
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        
        self.images_layout = QHBoxLayout()
        self.layout.addLayout(self.images_layout)
        
        self.original_frame = QFrame()
        self.original_frame.setFrameShape(QFrame.Box)
        self.original_frame.setLineWidth(2)
        self.original_image_label = QLabel("No image loaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_image_label)
        self.original_frame.setLayout(original_layout)
        self.images_layout.addWidget(self.original_frame)

        
        self.processed_frame = QFrame()
        self.processed_frame.setFrameShape(QFrame.Box)
        self.processed_frame.setLineWidth(2)
        self.processed_image_label = QLabel("No processed image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        processed_layout = QVBoxLayout()
        processed_layout.addWidget(self.processed_image_label)
        self.processed_frame.setLayout(processed_layout)
        self.images_layout.addWidget(self.processed_frame)

        self.controls_layout = QHBoxLayout()
        self.layout.addLayout(self.controls_layout)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.controls_layout.addWidget(self.load_button)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Global Thresholding",
            "Adaptive Thresholding",
            "Edge Detection",
            "Line Detection",
            "Point Detection",
            "Brightness Gradient Detection"
        ])
        self.controls_layout.addWidget(self.method_combo)

        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 255)
        self.threshold_spinbox.setValue(128)
        self.threshold_spinbox.setPrefix("Threshold: ")
        self.controls_layout.addWidget(self.threshold_spinbox)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_processing)
        self.controls_layout.addWidget(self.apply_button)

    def load_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
            if file_name:
                pil_image = Image.open(file_name)
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
                self.processed_image = None  
                self.display_image(self.original_image, self.original_image_label)
                self.processed_image_label.clear()
        except Exception as e:
            self.show_error_message("Error", f"Failed to load image: {str(e)}")

    def display_image(self, img, label):
        try:
            if len(img.shape) == 2: 
                height, width = img.shape
                bytes_per_line = width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif len(img.shape) == 3: 
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                raise ValueError("Unsupported image format.")
            
            label.setPixmap(QPixmap.fromImage(q_img).scaled(
                label.width(), label.height(), Qt.KeepAspectRatio
            ))
        except Exception as e:
            self.show_error_message("Error", f"Failed to display image: {str(e)}")

    def apply_processing(self):
        if self.original_image is None:
            self.show_error_message("Error", "No image loaded. Please load an image first.")
            return

        try:
            img = self.original_image.copy()
            method = self.method_combo.currentText()
            threshold = self.threshold_spinbox.value()  

            if method == "Global Thresholding":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, self.processed_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            elif method == "Adaptive Thresholding":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.processed_image = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY,
                    11,  
                    threshold  
                )
            elif method == "Edge Detection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.processed_image = cv2.Canny(gray, 50, 150)
            elif method == "Line Detection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
                self.processed_image = img.copy()
                if lines is not None:
                    for rho, theta in lines[:, 0]:
                        a, b = np.cos(theta), np.sin(theta)
                        x0, y0 = a * rho, b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(self.processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif method == "Point Detection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                corners = cv2.cornerHarris(gray, 2, 3, 0.04)
                self.processed_image = img.copy()
                self.processed_image[corners > 0.01 * corners.max()] = [0, 0, 255]
            elif method == "Brightness Gradient Detection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = cv2.magnitude(sobel_x, sobel_y)
                self.processed_image = cv2.convertScaleAbs(sobel_combined)

            self.display_image(self.processed_image, self.processed_image_label)
        except Exception as e:
            self.show_error_message("Error", f"Processing failed: {str(e)}")

    def show_error_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
