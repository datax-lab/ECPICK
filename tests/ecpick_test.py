import argparse
import os
from datetime import datetime

from ecpick import ECPICK
from ecpick.util import Logger

if __name__ == '__main__':

    # region [ Parameters ]
    dropout_rate = 0.8
    beta = 0.6
    num_of_neural = 384
    cuda = 0
    output_format = 'csv'

    # region [ Argument Parser ]
    parser = argparse.ArgumentParser(description="PredictEnzyme")
    parser.add_argument('--dropout-rate', '-D', type=float, default=dropout_rate)
    parser.add_argument('--beta', '-B', type=float, default=beta)
    parser.add_argument('--num-of-neural', '-n', type=int, default=num_of_neural)
    parser.add_argument('--cuda', '-c', type=int, default=cuda)
    parser.add_argument('--fasta-file', '-f', type=str, required=True)
    parser.add_argument('--output-path', '-o', type=str, required=True)
    parser.add_argument('--verbose', '-v', type=bool, default=False)
    parser.add_argument('--format', type=str, default='csv')
    args = parser.parse_args()

    dropout_rate = args.dropout_rate
    beta = args.beta
    num_of_neural = args.num_of_neural
    cuda = args.cuda
    fasta_file = args.fasta_file
    output_path = args.output_path
    verbose = args.verbose
    output_format = args.format
    # endregion

    Logger.CONSOLE = verbose

    Logger.info("#### Hyper Parameters ####")
    Logger.info(f"> DropoutRate: {dropout_rate}")
    Logger.info(f"> Beta: {beta}")
    Logger.info(f"> Num of Neural: {num_of_neural}")
    Logger.info(f"> Cuda: #{cuda}")
    Logger.info(f"> Fasta File: {fasta_file}")
    Logger.info(f"> Output Format: {output_format}")
    # endregion

    # region [ Create Result Folder ]
    Logger.info("#### Create Result Folder ####")
    os.makedirs("result", exist_ok=True)

    result_path = os.path.join("result", str(round(datetime.utcnow().timestamp() * 1000)))
    os.makedirs(result_path)
    Logger.info(f"> Folder '{result_path}' Check")

    model_path = os.path.join(result_path, "model")
    os.makedirs(model_path, exist_ok=True)
    Logger.info(f"> Folder '{model_path}' Check")

    img_path = os.path.join(result_path, "img")
    os.makedirs(img_path, exist_ok=True)
    Logger.info(f"> Folder '{img_path}' Check")

    log_path = os.path.join(result_path, "result.log")
    Logger.init(log_path, verbose)
    Logger.info("#### Start ECPICK Process ####")
    # endregion

    ecpick = ECPICK(dropout_rate, beta, num_of_neural, cuda)
    ecpick.predict_fasta(fasta_path=fasta_file, output_path=output_path, output_format=output_format)
