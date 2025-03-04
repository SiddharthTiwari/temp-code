import requests
import json

def translate_arabic_to_english(arabic_text, url="http://localhost:8000/v1/completions"):
    """
    Sends a POST request to vLLM model to translate Arabic text to English
    
    Args:
        arabic_text (str): Arabic text to translate
        url (str): URL of the vLLM API endpoint
    
    Returns:
        str: English translation
    """
    
    # Prepare the prompt with clear instructions
    prompt = f"""Translate the following Arabic text to English:

Arabic: {arabic_text}

English translation:"""
    
    payload = {
        "model": "your_model_name",  # Replace with your actual model name if needed
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.1,  # Lower temperature for more deterministic translations
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the translated text from the response
        # This may need adjustment based on your actual response format
        translation = result["choices"][0]["text"].strip()
        
        return translation
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

# Example usage
if __name__ == "__main__":
    arabic_text = "مرحبا بالعالم. كيف حالك اليوم؟"  # "Hello world. How are you today?"
    translation = translate_arabic_to_english(arabic_text)
    
    if translation:
        print(f"Arabic: {arabic_text}")
        print(f"English: {translation}")
    else:
        print("Translation failed.")
