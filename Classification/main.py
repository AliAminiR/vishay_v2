from __future__ import print_function
from __future__ import division

import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from Network import VishayNN
from Network import DataAugVishay
import logging

now = datetime.datetime.now()
log_filename = "vishay_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/{log_filename}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # logging.info("started data aug")
    # input_path = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_dataset\val\nio"
    # output_path = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_aug\val\nio"
    # data_aug = DataAugVishay(input_path, output_path, 20)
    # data_aug.data_aug_vishay()
    # logging.info("ended data aug")

    # TODO: calculate the mean and std and add the values to Nomalization
    lr = 0.005
    weight_decay = 0
    batch_size = 4
    num_epochs = 1

    logging.info("started vishay")
    path_to_dataset = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_final_ds"
    vishay = VishayNN(batch_size,lr,weight_decay, num_epochs)
    logging.info("created vishay object")
    vishay.create_vishay_network()
    logging.info("created NN")
    vishay.create_dataloader(path_to_dataset)
    logging.info("created data loader")
    vishay.create_optimizer()
    logging.info("created optimizer")
    vishay.train_model()
    logging.info("finished training!")
    vishay.test_model()
    logging.info("finished testing!")
