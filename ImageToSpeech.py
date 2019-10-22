import pytesseract
import pyttsx3
from PIL import Image

def convertImageToString(image):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    image = Image.open(image)
    text = pytesseract.image_to_string(image)
    print(text)
    textEngine = pyttsx3.init()
    textEngine.say(text)
    textEngine.runAndWait()
    textEngine.stop()