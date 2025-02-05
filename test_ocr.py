import easyocr
import sys

# Create an OCR reader object
reader = easyocr.Reader(['en'], gpu=False)

# Read text from an image
result = reader.readtext(sys.argv[1])
print("res: ", result)

# Print the extracted text
for detection in result:
    print(detection[1])
