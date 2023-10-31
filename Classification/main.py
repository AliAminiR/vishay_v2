from __future__ import print_function
from __future__ import division

from datetime import datetime

from Network import VishayNN
from Network import DataAugVishay
import logging

# import mlflow

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
    # Data Aug Parameters
    data_aug_flag = False
    input_path_for_data_aug = r"C:\Users\A750290_old\Projects\Vishay\Data\052SB18A12RS\052SB18A12RS_dataset\val\nio"
    output_path_for_data_aug = r"C:\Users\A750290_old\Projects\Vishay\Data\052SB18A12RS\052SB18A12RS_aug\val\nio"

    # Training Parameters
    lr = 0.00001
    weight_decay = 0.1
    batch_size = 1
    num_epochs = 1
    path_to_dataset = r"C:\Users\a750290\Projects\Vishay\Data\139DM15C16\139DM15C16_aug_v2"

    # Testing Parameters
    just_test = True
    save_images = True
    test_with_cam_flag = True
    model_name_to_load = "model_139DM15C16_2023-09-18_10-45_acc_0.92.pth"
    # model_name_to_load = "model_052SB18A12RS_2023-10-08_22-51_acc_0.88.pth"
    semi_conduct_name = "052SB18A12RS"

    # mlflow.pytorch.autolog()

    if data_aug_flag:
        logging.info("started data aug")
        data_aug = DataAugVishay(input_path_for_data_aug, output_path_for_data_aug, 10)
        data_aug.data_aug_vishay()
        logging.info("ended data aug")
    else:
        logging.info("started vishay")

        if test_with_cam_flag:
            batch_size = 1

        if just_test:
            model_name = model_name_to_load
        else:
            model_name = f"model_{semi_conduct_name}_{now.strftime('%Y-%m-%d_%H-%M')}"

        vishay = VishayNN(batch_size, lr, weight_decay, num_epochs, semi_conduct_name, save_images, model_name)
        logging.info("created vishay object")

        vishay.create_vishay_network()
        logging.info("created NN")

        vishay.create_dataloader(path_to_dataset)
        logging.info("created data loader")

        vishay.create_optimizer()
        logging.info("created optimizer")

        if just_test:
            vishay.load_model()
            logging.info(f"{model_name} Loaded!")
        else:
            vishay.train_model()
            logging.info("finished training!")
            vishay.save_model()
            logging.info("Model saved!")

        if test_with_cam_flag:
            vishay.test_with_CAM()
        else:
            vishay.test_model()

        vishay.confusion_matrix()
        logging.info("finished testing!")
