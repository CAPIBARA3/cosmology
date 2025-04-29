import pandas as pd
import numpy as np

class read_data:
    def __init__(self, file_path="./dataset.csv"):
        try:
            self.df = pd.read_csv(file_path)
            self.hubble = self.df['H0']
            self.hubble_sigma = self.df['sigmaH0']
            self.Om = self.df['Om']
            self.Om_sigma = self.df['sigmaOm']
            self.Ode = self.df['Ode']
            self.Ode_sigma = self.df['sigmaOde']
            self.dataset = self.df['Dataset']
            self.year = self.df['Year']
            self.detections = self.df['Direct/Indirect']
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            self.df = None
        except Exception as e:
            print(f"An error occurred: {e}")
            self.df = None

    def get_hubble(self):
        return (list(self.hubble), list(self.hubble_sigma))

    def get_matter(self):
        return (list(self.Om), list(self.Om_sigma))

    def get_lambda(self):
        return (list(self.Ode), list(self.Ode_sigma))

    def get_detection(self):
        return list(self.detections)

    def get_dataset(self):
        return self.dataset

    def get_year(self):
        return self.year


rd = read_data()