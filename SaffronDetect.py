import os
from PIL import Image
from skimage.feature import hog


class SaffronDetect:

    def __init__(self, feature_configuration, max_iterations=1000):
        self.MaxIteration = max_iterations
        self.data = []
        self.labels = []
        self.feature_configuration = feature_configuration

    def load_dataset(self, positive_samples_path, negative_samples_path):
        positive_samples = os.listdir(positive_samples_path)
        negative_samples = os.listdir(negative_samples_path)

        for sample in positive_samples:
            img = Image.open(positive_samples_path + '\\' + sample)
            gray = img.convert('L')
            feature_descriptor = hog(gray,
                                     self.feature_configuration.orientations,
                                     self.feature_configuration.cell_size,
                                     self.feature_configuration.normalization_block_size,
                                     block_norm='L2', feature_vector=True)
            self.data.append(feature_descriptor)
            self.labels.append(1)

        for sample in negative_samples:
            img = Image.open(negative_samples_path + '\\' + sample)
            gray = img.convert('L')
            feature_descriptor = hog(gray,
                                     self.feature_configuration.orientations,
                                     self.feature_configuration.cell_size,
                                     self.feature_configuration.normalization_block_size,
                                     block_norm='L2', feature_vector=True)
            self.data.append(feature_descriptor)
            self.labels.append(0)

    def save_models(self):
        pass

    def load_models(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def classification_report(self):
        pass

    def plot_roc_curves(self):
        pass








