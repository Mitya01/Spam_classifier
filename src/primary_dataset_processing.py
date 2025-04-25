import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import pandas as pd

# Downloading stop words, 1 time is enough
nltk.download('stopwords')


class TextProcessor:
    """
    A class for processing and normalizing Russian text.
    """
    def __init__(self):
        """
        Initializing a class
        """
        self.morph = MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))

    @staticmethod
    def clean_special_chars(text):
        """
        Clearing text from special characters, URLs, HTML tags, and numbers.
        :param text: Source text.
        :return: text: Text consisting of words only.
        """
        if pd.isna(text) or not isinstance(text, (str, bytes)):
            return ""
        text = re.sub(r'http\S+|<.*?>|[^\w\s]|\d+', '', text)
        return text

    @staticmethod
    def tokenize_text(text):
        """
        Tokenization of text into individual words
        :param text: Source text.
        :return: words: Space-separated text.
        """
        words = [word for word in text.split()]
        return words

    def remove_stop_words(self, words):
        """
        Removing stop words from the word list.
        :param words: A list of words.
        :return: The list in which the stop words are deleted.
        """
        return [word for word in words if word not in self.stop_words]

    def lemmatizate_words(self, words):
        """
        Lemmatization of words to their basic form.
        :param words: A list of words.
        :return: lemmatizated_words: Words after lemmatization.
        """
        lemmatizated_words = []
        for word in words:
            parses = self.morph.parse(word)
            if parses:
                # Using the first normal form
                lemmatizated_word = parses[0].normal_form
                lemmatizated_words.append(lemmatizated_word)
            else:
                # If the word is not found, leave it as it is.
                lemmatizated_words.append(word)
        return lemmatizated_words

    def process_text(self, text):
        """
        Full text processing: cleaning of special characters, tokenization, removal of stop words and lemmatization.
        :param text: Source text.
        :return: processed_text: Processed text.
        """

        # Cleaning of special characters
        cleaned_text = self.clean_special_chars(text)

        # Tokenization
        tokens = self.tokenize_text(cleaned_text)

        # Lemmatization
        lemmatized_tokens = self.lemmatizate_words(tokens)

        # Removal of stop words
        filtered_tokens = self.remove_stop_words(lemmatized_tokens)

        # Combining words back into text
        processed_text = " ".join(filtered_tokens)

        return processed_text


# Usage example
if __name__ == "__main__":
    processor = TextProcessor()

    text = "Это пример текста с числами 123 и знаками препинания: !, ?."
    processed_text = processor.process_text(text)
    print("Обработанный текст:", processed_text)
