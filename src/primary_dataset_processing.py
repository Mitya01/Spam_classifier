import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

# загружаются стоп-слова
nltk.download('stopwords')


class TextProcessor:
    """
    Класс для обработки и нормализации русского текста.
    """
    def __init__(self):
        """
        Инициализация класса
        """
        self.morph = MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))

    @staticmethod
    def clean_special_chars(text):
        """
        Очистка текста от спецсимволов, URL-адресов, HTML-тегов и чисел.
        :param text: исходный текст
        :return:
        """
        text = re.sub(r'http\S+|<.*?>|[^\w\s]|\d+', '', text)
        return text

    @staticmethod
    def tokenize_text(text):
        """
        Токенизация текста на отдельные слова
        :param text: Исходный текст
        :return:
        """
        words = [word for word in text.split()]
        return words

    def remove_stop_words(self, words):
        """
        Удаление стоп-слов из списка слов.
        :param words: Список слов
        :return:
        """
        return [word for word in words if word not in self.stop_words]

    def lemmatizate_words(self, words):
        """
        Лемматизация слов до их базовой формы.
        :param words: Список слов
        :return:
        """
        lemmatizated_words = []
        for word in words:
            parses = self.morph.parse(word)
            if parses:
                # Используем первую нормальную форму
                lemmatizated_word = parses[0].normal_form
                lemmatizated_words.append(lemmatizated_word)
            else:
                # Если слово не найдено, оставляем его как есть
                lemmatizated_words.append(word)
        return lemmatizated_words

    def process_text(self, text):
        """
        Полная обработка текста: очистка, токенизация, удаление стоп-слов и лемматизация.
        :param text: Исходный текст
        :return:
        """
        # Очистка от спецсимволов
        cleaned_text = self.clean_special_chars(text)

        # Токенизация
        tokens = self.tokenize_text(cleaned_text)

        # Лемматизация
        lemmatized_tokens = self.lemmatizate_words(tokens)

        # Удаление стоп-слов
        filtered_tokens = self.remove_stop_words(lemmatized_tokens)

        # Объединение слов обратно в текст
        processed_text = " ".join(filtered_tokens)

        return processed_text


# Пример использования
if __name__ == "__main__":
    processor = TextProcessor()

    text = "Это пример текста с числами 123 и знаками препинания: !, ?."
    processed_text = processor.process_text(text)
    print("Обработанный текст:", processed_text)
