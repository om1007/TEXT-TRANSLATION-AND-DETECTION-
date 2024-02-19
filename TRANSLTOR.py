import cv2
import matplotlib.pyplot as plt
import easyocr
from deep_translator import GoogleTranslator

# Read the image (ensure correct format)
img = cv2.imread(r"C:\Users\om\Desktop\CV\kor.jpeg", cv2.IMREAD_COLOR)

# Detect text using EasyOCR

reader = easyocr.Reader(['hi', 'mr','ne'], gpu=False)
reader = easyocr.Reader(['ar', 'fa','ur'], gpu=False)
reader = easyocr.Reader(['fr','es','ga','de'], gpu=False)
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
reader = easyocr.Reader(['ru', 'rs_cyrillic','bg','uk','mn','be'], gpu=False)
reader = easyocr.Reader(['te', 'en'], gpu=False)
reader = easyocr.Reader(['ta', 'en'], gpu=False)
reader = easyocr.Reader(['ko', 'en'], gpu=False)
reader = easyocr.Reader(['ja', 'en'], gpu=False)
# print(easyocr.Reader.get_available_languages())

text_ = reader.readtext(img)



# Create an instance of GoogleTranslator from deep-translator

translator = GoogleTranslator(source='auto', target='en')



# Initialize an empty list to store translated text

translated_text = []



# Translate each detected text block

for t in text_:

    bbox, text, score = t



    # Translate using deep-translator

    translated = translator.translate(text)



    # Append translated text and metadata

    translated_text.append((bbox, translated, score))



# Draw bounding boxes and text (translated or original)
# here text_to_display means the translated text 
for bbox, text_to_display, score in translated_text:
    # Extract coordinates and ensure correct data types
    x1, y1 = int(bbox[0][0]), int(bbox[0][1])  # Top-left corner
    x2, y2 = int(bbox[2][0]), int(bbox[2][1])  # Bottom-right corner

    # Print coordinates for debugging (optional)
    #print("Coordinates:", x1, y1, x2, y2)

    text_origin =  (x1, y1)

    # Draw the rectangle with proper coordinates
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Add text above the rectangle
    cv2.putText(img, text_to_display,  text_origin, cv2.FONT_HERSHEY_COMPLEX,1.5 , (255, 0, 0), 3)

# Display the image with detected and translated text
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
