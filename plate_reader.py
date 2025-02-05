import time
import requests
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import easyocr
import sys
import pytesseract

from plate_ocr import recognize_plate

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class PlateReader:
    OPENALPR_PATH = "openalpr_64\\alpr.exe"

    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.yolo11 = YOLO('yolo11n.pt', task='detect')
        self.plate_detector_model = YOLO('license_plate_detector.pt')
        self.plate_runner_action = None

    def set_plate_runner_action(self, method):
        self.plate_runner_action = method

    def read_aplr_plate(self, frame_path):
        output = subprocess.run([self.OPENALPR_PATH, "-", "-c", "eu", frame_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = output.stdout.decode('utf-8').splitlines()[1:]
        return output

    def test_contours(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            if w < 50 or h < 50:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("tessssst", frame)

    def keep_black_pixels_only(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("binary mask", binary_mask)
        output_image = np.ones_like(gray_image) * 255
        output_image = np.where(binary_mask == 0, binary_mask, output_image)
        contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("mid output", output_image)
        mask = np.zeros_like(output_image)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        cv2.imshow("mask2", mask)
        result = np.where(mask == 255, output_image, 255)
        return result

    def read_license_plate(self, license_plate_crop):
        return recognize_plate(license_plate_crop), 0.9

    def extract_license_plates_area(self, frame, coordinates):
        if coordinates is None:
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        else:
            x1, y1, x2, y2 = map(int, coordinates.tolist())
        r0 = max(0, x1)
        r1 = max(0, y1)
        r2 = min(frame.shape[1], x2)
        r3 = min(frame.shape[0], y2)
        car_region = frame[r1:r3, r0:r2]
        for license_plate in (self.plate_detector_model(car_region)[0]).boxes.data.tolist():
            px1, py1, px2, py2, pscore, pclass_id = license_plate
            print(f"Plate: {pscore=}, ")
            yield (x1+int(px1), y1+int(py1), x1+int(px2), y1+int(py2)), pscore

    def detect_plates_and_run_model(self, frame, model=None):
        if model is None:
            model = self.yolo11
        results = model.predict(source=frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                if model.names[class_id].lower() not in ["car", "bus", "truck", "motorcycle", "person"]:
                    continue
                confidence = box.conf.item()
                coordinates = box.xyxy[0]
                print(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f},")
                for coords, pscore in self.extract_license_plates_area(frame, None):
                    px1, py1, px2, py2 = coords
                    license_plate_crop = frame[(py1):(py2), (px1): (px2), :]
                    cv2.rectangle(frame, ((px1), (py1)), ((px2), (py2)), (0, 255, 0), 2)
                    plate, _ = self.read_license_plate(license_plate_crop)
                    print(f"{plate=}, {pscore=}")
                    if self.plate_runner_action:
                        self.plate_runner_action(plate, pscore)
                    cv2.putText(
                        img=frame,
                        text=f"Plate: {plate} ({pscore:.2f})",
                        org=(px1, py1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(255, 0, 0),
                        thickness=2
                    )
                x1, y1, x2, y2 = map(int, coordinates.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    img=frame,
                    text=f"{model.names[class_id]} ({confidence:.2f})",
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(255, 0, 0),
                    thickness=2
                )

    def detect_plates(self, frame):
        for coords, pscore in self.extract_license_plates_area(frame, None):
            px1, py1, px2, py2 = coords
            license_plate_crop = frame[(py1):(py2), (px1): (px2), :]
            cv2.rectangle(frame, ((px1), (py1)), ((px2), (py2)), (0, 255, 0), 2)
            plate, _ = self.read_license_plate(license_plate_crop)
            print(f"{plate=}, {pscore=}")
            cv2.putText(
                img=frame,
                text=f"Plate: {plate} ({pscore:.2f})",
                org=(px1, py1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 0, 0),
                thickness=2
            )

class VideoFeed:
    def __init__(self, ip):
        self.ip = ip

    def read_frame(self):
        url = f"http://{self.ip}/capture"
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            response.raise_for_status()

    def display_frame(self, frame):
        np_arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imshow('Frame', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_video_feed(self, plate_reader: PlateReader):
        while True:
            frame = self.read_frame()
            time.sleep(0.1)
            np_arr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            plate_reader.detect_plates(img)
            cv2.imshow('Video Feed', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def display_local_video_feed(self, idx_or_url: int, plate_reader: PlateReader):
        print(idx_or_url, type(idx_or_url))
        cap = cv2.VideoCapture(idx_or_url)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        except:
            pass
        while True:
            _, frame = cap.read()
            plate_reader.detect_plates(frame)
            cv2.imshow('Video Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def map_contours_only(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    result = np.where(mask == 255, frame, 255)
    cv2.imshow("result", result)
    return result


def main():
    if len(sys.argv) <= 1:
        return

    frame_path = sys.argv[1]
    frame = cv2.imread(frame_path)
    plate_reader = PlateReader()
    plate_reader.detect_plates(frame)
    video_feed = VideoFeed("")
    video_feed.display_frame(cv2.imencode('.jpg', frame)[1].tobytes())


if __name__ == "__main__":
    main()
