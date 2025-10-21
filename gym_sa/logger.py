import csv
import os


class CSVLogger:
    def __init__(self, filename, fieldnames):
        """
        Initialize the CSV logger. Writes headers if the file does not exist.
        Args:
            filename (str): Path to the CSV file.
            fieldnames (list): List of column headers.
        """
        self.filename = filename
        self.fieldnames = fieldnames
        file_exists = os.path.isfile(filename)

        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row_dict):
        """
        Append a row of data to the CSV file.
        Args:
            row_dict (dict): Dictionary mapping column headers to values.
        """
        with open(self.filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row_dict)
