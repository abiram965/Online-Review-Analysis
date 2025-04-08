from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            return translated_text, lang
        return text, "en"
    except:
        return text, "unknown"

# Example Usage
review = "El servicio de Google, ofrecido de forma gratuita"  # Spanish review
translated_review, original_language = detect_and_translate(review)
print(f"Original Language: {original_language}")
print(f"Translated Review: {translated_review}")
