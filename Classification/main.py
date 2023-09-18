from __future__ import print_function
from __future__ import division

from datetime import datetime

from Network import VishayNN
from Network import DataAugVishay
import logging

now = datetime.now()
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

    lr = 0.0001
    weight_decay = 0.1
    batch_size = 4
    num_epochs = 50
    just_test = False
    data_aug_flag = False
    model_name_to_load = "model_2023-09-12_11-37-56_acc_0.52.pth"

    if data_aug_flag:
        logging.info("started data aug")
        input_path = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_dataset\val\nio"
        output_path = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_aug_v2\val\nio"
        data_aug = DataAugVishay(input_path, output_path, 10)
        data_aug.data_aug_vishay()
        logging.info("ended data aug")
    else:
        logging.info("started vishay")
        path_to_dataset = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_aug_v2"
        # path = r"C:\Users\A750290\Projects\Vishay\Data\139DM15C16_for_normalization"

        vishay = VishayNN(batch_size, lr, weight_decay, num_epochs)
        logging.info("created vishay object")
        vishay.create_vishay_network()
        logging.info("created NN")
        vishay.create_dataloader(path_to_dataset)
        # mean, std = vishay.get_mean_std(path)
        logging.info("created data loader")
        vishay.create_optimizer()
        logging.info("created optimizer")

        if just_test:
            vishay.load_model(f"./models/{model_name_to_load}")
            logging.info(f"{model_name_to_load} Loaded!")
        else:
            vishay.train_model()
            logging.info("finished training!")
            vishay.save_model(f"./models/model_{now.strftime('%Y-%m-%d_%H-%M')}.pth")
            logging.info("Model saved!")

        vishay.test_model()
        logging.info("finished testing!")
