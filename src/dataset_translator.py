import pandas as pd
from src.text_translator import TextTranslator


class DataProcessor:
    def __init__(self, csv_input_path, csv_output_path):
        """
        Инициализация объекта DataProcessor.
        :param csv_input_path: Путь к входному csv-файлу.
        :param csv_output_path: Путь к выходному csv-файлу.
        """
        self.csv_input_path = csv_input_path
        self.csv_output_path = csv_output_path

    def load_data(self):
        """
        Загрузка данных из CSV-файла.
        :return: DataFrame с загруженными данными.
        """
        return pd.read_csv(self.csv_input_path)

    @staticmethod
    def process_categorical_data(df):
        """
        Обработка категориальных данных в первом столбце.
        :param df: DataFrame с данными для обработки.
        :return: DataFrame с обработанными категориальными данными.
        """
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: 0 if x == 'ham' else 1)
        return df

    @staticmethod
    def process_text_data(df):
        """
        Обработка текстовых данных во втором столбце (английский текст).
        :param df: DataFrame с данными для обработки.
        :return: DataFrame с обработанными текстовыми данными.
        """
        translator = TextTranslator()
        df.iloc[:, 1] = df.iloc[:, 1].apply(translator.translate_text)
        return df

    def save_data(self, df):
        """
        Сохранение обработанных данных в новый CSV-файл.
        :param df: DataFrame с обработанными данными.
        :return: None
        """
        df.to_csv(self.csv_output_path, index=False)

    def run(self):
        """
        Основной метод для запуска обработки данных.
        """
        df = self.load_data()
        df = self.process_categorical_data(df)
        df = self.process_text_data(df)
        self.save_data(df)

if __name__ == "__main__":
    csv_input_path = '../datasets/english_dataset.csv'
    csv_output_path = '../datasets/russian_dataset.csv'
    processor = DataProcessor(csv_input_path, csv_output_path)
    processor.run()
