import torch.utils.data


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, y_true, y_true_one_hot, x_value):
        self.y_true = y_true
        self.y_true_one_hot = y_true_one_hot
        self.x_value = x_value

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, index):
        return self.y_true_one_hot[index], self.x_value[index]


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, x_value):
        self.x_value = x_value

    def __len__(self):
        return len(self.x_value)

    def __getitem__(self, index):
        return self.x_value[index]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, generator):
        self.generator = generator
        self.extract = generator.extract()

    def __len__(self):
        return self.generator.num_of_images

    def __getitem__(self, item):
        data = next(self.extract)
        if self.generator.sample_weight is None:
            return data[0], data[1]  # X, Y
        else:
            return data[0], data[1], data[2]  # X, Y, Weights
