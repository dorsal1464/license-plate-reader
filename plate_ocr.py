import cv2
import pytesseract
import numpy as np
import re
import math
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


MIN_AREA = 90
MIN_RATIO = 1.45
pad = 40  # breaks detections sometimes WTF
MIN_HEIGHT_RATIO = 0.25
MIN_WIDTH_RATIO = 1.0 / 30


def recognize_digit(roi):
    # currently uses tesseract, which is apparently a piece of trash
    # if i had more time i would train a model myself to read the digits
    psms = [8, 6, 10, 9, 7]
    valid_answers = {}
    text = ""
    for psm in psms:
        _text = pytesseract.image_to_string(roi, config=f'-c tessedit_char_whitelist=0123456789 --psm {psm}')
        if _text:
            text = _text[0]
            valid_answers[psm] = text
    print(valid_answers)
    if len(valid_answers) > 1 and len(set(valid_answers.values())) > 1:
        return list(valid_answers.values())[0]
    return text


def recognize_plate(img):
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    box = img
    # print(box.shape[:2])
    scalar = int(math.ceil(800.0 / box.shape[1]))
    print(f"Scaling to {scalar}")
    # grayscale region within bounding box
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = scalar, fy = scalar, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    # cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    im_height, im_width = im2.shape
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(im2, (x,y), (x+w, y+h), (0,100,0),1)
        # if height of box is not tall enough relative to total height then skip
        if  float(h) / im_height < MIN_HEIGHT_RATIO: continue

        ratio = h / float(w)

        if ratio < MIN_RATIO: continue

        # if width is not wide enough relative to total width then skip
        if  float(w) / im_width < MIN_WIDTH_RATIO: continue

        area = h * w

        if area < MIN_AREA: continue

        # draw the rectangle
        cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-0:y+h+0, x-0:x+w+0]
        # perform bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        # roi = cv2.medianBlur(roi, 5)
        # expand with white in all directions
        roi = cv2.copyMakeBorder(roi, pad // 2, pad // 2, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        try:
            # print("try:")
            # cv2.imshow(f"try_{x}", roi)
            # print("", pytesseract.image_to_data(roi, config='-c tessedit_char_whitelist=0123456789 --psm 10'))
            text = recognize_digit(roi)
            # clean tesseract text by removing any unwanted blank spaces
            print(">>> ", text)
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except Exception as e:
            raise (e)
        
    if plate_num != None:
        print("License Plate #: ", plate_num)
    
    cv2.imshow("Character's Segmented", im2)
    # cv2.waitKey(0)
    return plate_num


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    print(recognize_plate(img))
