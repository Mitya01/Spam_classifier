from deep_translator import GoogleTranslator


class TextTranslator:
    def __init__(self):
        pass

    @staticmethod
    def translate_text(text):
        """
        Переводит текст с помощью Google Translator.
        :param text: Текст для перевода.
        :return: Переведенный текст.
        """
        try:
            translator = GoogleTranslator(source='en', target='ru')
            return translator.translate(text)
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            return None

# Пример использования:
if __name__ == "__main__":
    translator = TextTranslator()
    # original_text = "Hi, how are you?"
    original_text = 'ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."'
    translated_text = translator.translate_text(original_text)
    print(f"Перевод: {translated_text}")
