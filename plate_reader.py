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

reader = easyocr.Reader(['en'], gpu=False)
yolo11 = YOLO('yolo11n.pt', task='detect')
plate_detector_model = YOLO('license_plate_detector.pt')
plate_runner_action = None

IP = "192.168.4.1"
ALPR_PATH = "openalpr_64\\alpr.exe"


def run_plate_db_action(number, conf):
    return True


def read_aplr_plate(frame_path):
    output = subprocess.run([ALPR_PATH, "-", "-c", "eu", frame_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output =  output.stdout.decode('utf-8').splitlines()[1:]
    return output


def test_contours(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        if w < 50 or h < 50:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0),2)
    cv2.imshow("tessssst", frame)


def keep_black_pixels_only(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where black pixels are 1 and all other pixels are 0
    _, binary_mask = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("binary mask", binary_mask)
    # Create an output image where all pixels are white
    output_image = np.ones_like(gray_image) * 255
    
    # Use np.where to copy the black pixels from the original image to the output image
    output_image = np.where(binary_mask == 0, binary_mask, output_image)
    contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("mid output", output_image)
    mask = np.zeros_like(output_image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    cv2.imshow("mask2", mask)
    result = np.where(mask == 255, output_image, 255)
    return (result)


def read_license_plate(license_plate_crop):
    # license_plate_crop = cv2.resize(license_plate_crop, None, fx=3, fy=3)
    return recognize_plate(license_plate_crop), 0.9
    # frame = map_contours_only(license_plate_crop)
    # license_plate_crop = frame
    
    
    
    # test = keep_black_pixels_only(license_plate_crop)
    # cv2.imshow("test", test)
    cv2.imshow("Plate only", license_plate_crop)
    # save image crop
    timestamp = int(time.time())
    crop_path = f"plates\\license_plate_crop_{timestamp}.png"
    text = ""
    # cv2.imwrite(crop_path, license_plate_crop)

    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(crop_path, license_plate_crop_gray)
    
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 60, 255, cv2.THRESH_BINARY)
    _, license_plate_crop_thresh_inv = cv2.threshold(license_plate_crop_gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    # cv2.imwrite(crop_path, license_plate_crop_thresh)
    # print(pytesseract.image_to_data(license_plate_crop_thresh))
    raw_text = pytesseract.image_to_string(license_plate_crop_thresh, config="-c tessedit_char_whitelist=0123456789 --psm 6")
    if not raw_text:
        # cv2.imwrite(crop_path, license_plate_crop_thresh_inv)
        raw_text = pytesseract.image_to_string(license_plate_crop_thresh_inv, config="-c tessedit_char_whitelist=0123456789 --psm 6")
    print("<<< raw >>>", raw_text)
    for c in raw_text:
        if c.isdigit():
            text += c
    return (text, 0.9)
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if True:
            return (text, score)
    return None, None


def extract_license_plates_area(frame, coordinates):
    if coordinates is None:
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
    else:
        x1, y1, x2, y2 = map(int, coordinates.tolist())
    r0 = max(0, x1)
    r1 = max(0, y1)
    r2 = min(frame.shape[1], x2)
    r3 = min(frame.shape[0], y2)
    car_region = frame[r1:r3, r0:r2]
    for license_plate in (plate_detector_model(car_region)[0]).boxes.data.tolist():
            px1, py1, px2, py2, pscore, pclass_id = license_plate
            print(f"Plate: {pscore=}, ")
            yield (x1+int(px1), y1+int(py1), x1+int(px2), y1+int(py2)), pscore
    

def dark_magic_function(frame, model=yolo11):
    global plate_runner_action
    results = model.predict(source=frame)

    # Iterate over results and draw predictions
    for result in results:
        boxes = result.boxes  # Get the boxes predicted by the model
        for box in boxes:
            class_id = int(box.cls)  # Get the class ID
            if model.names[class_id].lower() not in ["car", "bus", "truck", "motorcycle", "person"]:
                continue
            confidence = box.conf.item()  # Get confidence score
            coordinates = box.xyxy[0]  # Get box coordinates as a tensor
            print(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f},")
            # Extract and convert box coordinates to integers
            for coords, pscore in extract_license_plates_area(frame, None):
                px1, py1, px2, py2 = coords
                license_plate_crop = frame[(py1):(py2), (px1): (px2), :]
                cv2.rectangle(frame, ((px1), (py1)), ((px2), (py2)), (0, 255, 0), 2)
                plate, _ = read_license_plate(license_plate_crop)
                print(f"{plate=}, {pscore=}")

                if plate_runner_action:
                    plate_runner_action(plate, pscore)
                
                cv2.putText(
                    img=frame,
                    text=f"Plate: {plate} ({pscore:.2f})",
                    org=(px1, py1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(255, 0, 0),
                    thickness=2
                )
            
            x1, y1, x2, y2 = map(int, coordinates.tolist())  # Convert tensor to list and then to int

            # Draw the box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
            cv2.putText(
                    img=frame,
                    text=f"{model.names[class_id]} ({confidence:.2f})",
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(255, 0, 0),
                    thickness=2
                )
            

def dark_magic_function2(frame, model=yolo11):
    for coords, pscore in extract_license_plates_area(frame, None):
        px1, py1, px2, py2 = coords
        license_plate_crop = frame[(py1):(py2), (px1): (px2), :]
        cv2.rectangle(frame, ((px1), (py1)), ((px2), (py2)), (0, 255, 0), 2)
        plate, _ = read_license_plate(license_plate_crop)
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


def read_frame(ip):
    url = f"http://{ip}/capture"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        response.raise_for_status()


def display_frame(frame):
    np_arr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow('Frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_video_feed(ip):
    while True:
        frame = read_frame(ip)
        time.sleep(0.1)
        np_arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        dark_magic_function(img)
        cv2.imshow('Video Feed', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def display_local_video_feed(idx):
    cap = cv2.VideoCapture(index=idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        _, frame = cap.read()
        dark_magic_function(frame)
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
    # cv2.imshow("mask", mask)
    result = np.where(mask == 255, frame, 255)
    cv2.imshow("result", result)
    return result


def main():
    if len(sys.argv) <= 1:
        return

    frame_path = sys.argv[1]
    frame = cv2.imread(frame_path)
    dark_magic_function2(frame)
    display_frame(cv2.imencode('.jpg', frame)[1].tobytes())


if __name__ == "__main__":
    main()
