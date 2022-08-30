import os
import pickle
import zipfile

import numpy as np
import torch
from Bio import SeqIO

from ecpick.model import PredictionModel
from ecpick.util import DataLoader, PredictDataset
from ecpick.util import Logger, SeqEncoder, StopWatch
from ecpick.util import download_file


class ECPICK:
    """
    ECPICK

    Parameters
    ----------
    dropout_rate: Dropout Rate (default 0.8)
    beta: Beta value (default 0.6)
    num_of_neural: Number of Neural in Hidden Layer (default 384)
    cuda: CUDA Devices (default cpu)

    Examples
    --------
    >>> from ecpick import ECPICK
    >>> ecpick = ECPICK(dropout_rate=0.8, beta=0.6, num_of_neural=384, cuda='0')
    >>> ecpick.predict_fasta(fasta_path='sample.fasta')
    """

    def __init__(self, dropout_rate=0.8, beta=0.6, num_of_neural=384, cuda='cpu'):
        self.__dropout_rate = dropout_rate
        self.__beta = beta
        self.__num_of_neural = num_of_neural
        self.__cuda = cuda

        # region [ CUDA Setting ]
        Logger.info("#### CUDA Settings ####")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        if cuda != 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)

        self.__cuda_available = torch.cuda.is_available()
        Logger.info(f"> Is CUDA available? {self.__cuda_available}")
        if self.__cuda_available:
            cuda_device_count = torch.cuda.device_count()
            Logger.info(f"> CUDA Device Count: {cuda_device_count}")

            for i in range(cuda_device_count):
                Logger.info(f"> CUDA #{i}: {torch.cuda.get_device_name(i)}")

            cuda_device = torch.cuda.current_device()
            Logger.info(f"> Selected CUDA Device: #{cuda_device} ({torch.cuda.get_device_name(cuda_device)})")
        else:
            cuda_device = torch.device("cpu")

        self.__cuda_device = cuda_device
        # endregion

        # region [ Load Predictable Classes ]
        Logger.info('#### Load Predictable Classes ####')
        Logger.info('> Load output classes')
        base_path = f'assets'
        os.makedirs(base_path, exist_ok=True)
        class_path = os.path.join(base_path, f'output_classes.pkl')

        if not os.path.exists(class_path):
            Logger.warn("> Not Exist Classes file. Start download")
            url = "https://download.sharenshare.kr/ecpick/output_classes.pkl"
            download_file(url, output_path=class_path)
            Logger.warn("> Download Complete")

        with open(class_path, 'rb') as f:
            self.__output_classes = pickle.load(f)

        cursor = []
        for i in range(len(self.__output_classes)):
            if i == 0:
                cursor.append([0, len(self.__output_classes[i])])
            else:
                cursor.append([cursor[-1][1], cursor[-1][1] + len(self.__output_classes[i])])
        Logger.info(f"> Cursor: {cursor}")

        # endregion

        # region [ Load Model ]
        Logger.info("#### Load Model ####")

        base_path = 'assets/models'
        if not os.path.exists(base_path):
            Logger.warn("> Not Exist Models file. Start download")
            os.makedirs(base_path, exist_ok=True)
            url = "https://download.sharenshare.kr/ecpick/models.zip"
            download_file(url, output_path='assets/models.zip')
            Logger.warn("> Download Complete, Start Extracting...")
            with zipfile.ZipFile(r'assets/models.zip', 'r') as zip_file:
                zip_file.extractall('assets/models')
            Logger.warn("> Extract Complete")
            os.remove(r'assets/models.zip')

        model_path = [
            os.path.join(base_path, 'model-0.pth'),
            os.path.join(base_path, 'model-1.pth'),
            os.path.join(base_path, 'model-2.pth'),
            os.path.join(base_path, 'model-3.pth'),
            os.path.join(base_path, 'model-4.pth'),
            os.path.join(base_path, 'model-5.pth'),
            os.path.join(base_path, 'model-6.pth'),
            os.path.join(base_path, 'model-7.pth'),
            os.path.join(base_path, 'model-8.pth'),
            os.path.join(base_path, 'model-9.pth')
        ]

        self.__models = []
        for i in range(10):
            Logger.info(f"> Load Model {i + 1}")
            model = PredictionModel(self.__output_classes,
                                    cuda_support=self.__cuda_available,
                                    cuda_device=self.__cuda_device,
                                    dropout_rate=self.__dropout_rate,
                                    beta=self.__beta,
                                    relu_size=self.__num_of_neural,
                                    model_path=model_path[i])
            model.eval()
            self.__models.append(model)
        # endregion

    def predict_fasta(self, fasta_path):

        # region [ Load Sequence ]
        Logger.info("#### Load Sequence ####")

        seq_encoder = SeqEncoder()

        y_id = []
        y_true_seq = []
        y_predict_seq = []
        x_value = []

        for record in SeqIO.parse(fasta_path, 'fasta'):
            y_id.append(record.id)
            seq = str(record.seq)
            if len(seq) > 1000:
                seq = seq[:1000]
            y_true_seq.append(seq)
            seq = seq + (' ' * (1000 - len(seq)))
            y_predict_seq.append(seq)
            x_value.append(seq_encoder.seq_to_one_hot_encoding(seq))

        x_value = np.reshape(np.array(x_value), (len(x_value), 1, 1000, 21))
        dataset = PredictDataset(x_value)
        data_loader = DataLoader(dataset, batch_size=32)
        Logger.info(f"> Number of Sequence: {x_value.shape}")
        # endregion

        # region [ Predict ]
        Logger.info(f"#### Start Predict Process ####")

        stop_watch = StopWatch()
        stop_watch.start()

        y_pred = []
        y_pred_prob = []
        y_prob = []

        c = 0
        with torch.no_grad():

            for batch_idx, data in enumerate(data_loader):
                if self.__cuda_available:
                    X_value = data.to(self.__cuda_device).float()
                else:
                    X_value = data.float()

                result_p = None
                for i in range(len(self.__models)):
                    output = self.__models[i](X_value)
                    result_p = output['final_output'] if result_p is None else result_p + output['final_output']
                result_p /= 10

                result_prob = result_p.cpu().numpy()
                prob = np.argwhere(result_prob >= 0.19)

                for i in range(len(X_value)):
                    pred = np.argwhere(prob[:, 0] == i).tolist()

                    if len(pred) == 0:
                        y_prob.append(np.zeros(len(self.__output_classes[-1]), dtype=np.uint8).tolist())
                        y_pred.append([])
                        y_pred_prob.append([])
                        c += 1
                        continue

                    pred = self.__output_classes[-1][prob[np.reshape(np.argwhere(prob[:, 0] == i).tolist(), -1)][:, 1]].tolist()
                    pred_prob = result_prob[i][prob[np.reshape(np.argwhere(prob[:, 0] == i).tolist(), -1)][:, 1]] * 100
                    pred_prob = pred_prob.tolist()

                    y_prob.append(result_prob[i].tolist())
                    y_pred.append(pred)
                    y_pred_prob.append(pred_prob)
        # endregion

        # region [ Show Result ]
        for i, idx in enumerate(y_id):
            Logger.info(f"Result: ID={idx}, EC={y_pred[i]}, Prob={y_pred_prob[i]}")
        # endregion
