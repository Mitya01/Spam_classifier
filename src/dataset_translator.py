import pandas as pd
from src.text_translator import TextTranslator


class DataProcessor:
    def __init__(self, csv_input_path, csv_output_path):
        """
        Initialization of the Data Processor object.
        :param csv_input_path: Path to the input csv file.
        :param csv_output_path: Path to the output csv file.
        """
        self.csv_input_path = csv_input_path
        self.csv_output_path = csv_output_path

    def load_data(self):
        """
        Uploading data from a CSV file.
        :return: Data Frame with uploaded data.
        """
        return pd.read_csv(self.csv_input_path)

    @staticmethod
    def process_categorical_data(df):
        """
        Processing the categorical data in the first column.
        :parameter df: A data frame with data to process.
        :return: A data frame with processed categorical data.
        """
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: 0 if x == 'ham' else 1)
        return df

    @staticmethod
    def process_text_data(df):
        """
        Processing of text data in the second column (English text).
        :param df: Data Frame with data to process.
        :return: Data Frame with processed text data.
        """
        translator = TextTranslator()
        df.iloc[:, 1] = df.iloc[:, 1].apply(translator.translate_text)
        return df

    def save_data(self, df):
        """
        Saving the processed data to a new CSV file.
        ::param df: Data Frame with processed data.
        :return: None
        """
        df.to_csv(self.csv_output_path, index=False)

    def run(self):
        """
        The main method for starting data processing.
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
