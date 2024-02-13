import cv2
import matplotlib.pyplot as plt
import easyocr
from deep_translator import GoogleTranslator

# Read the image (ensure correct format)
img = cv2.imread(r"C:\Users\om\Desktop\CV\gg.jpg", cv2.IMREAD_COLOR)

# Detect text using EasyOCR

reader = easyocr.Reader(['ru', 'bg'], gpu=False)

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
    cv2.putText(img, text_to_display,  text_origin, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with detected and translated text
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
