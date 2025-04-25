import pandas as pd
from src.primary_dataset_processing import TextProcessor


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
    def process_text_data(df):
        """
        Processing of text data in the second column (cleaning of special characters, tokenization,
        removal of stop words and lemmatization).
        :param df: Data Frame with data to process.
        :return: Data Frame with processed text data.
        """
        processor = TextProcessor()
        df.iloc[:, 1] = df.iloc[:, 1].apply(processor.process_text)
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
        df = self.process_text_data(df)
        self.save_data(df)

if __name__ == "__main__":
    csv_input_path = '../datasets/russian_dataset.csv'
    csv_output_path = '../datasets/processed_dataset.csv'
    processor = DataProcessor(csv_input_path, csv_output_path)
    processor.run()
