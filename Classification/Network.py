from __future__ import print_function
from __future__ import division

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import gc
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
# from torchcam.methods import SmoothGradCAMpp
from torchcam.methods import CAM
from PIL import Image
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as f
from datetime import datetime

import imgaug.augmenters as iaa
import imageio.v2 as imageio

import logging

from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image

logger = logging.getLogger(__name__)


class VishayNN:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes = ("iO", "NiO")
    writer = None

    model = None
    dataloaders_dict = None
    batch_size = None
    lr = None
    weight_decay = None
    optimizer = None
    criterion = None
    num_epochs = None
    best_acc_after_training = 0.0
    semi_conduct_name = None
    image_num = 0
    save_test_images_flag = True
    model_name: str = None

    def __init__(self, batch_size, lr, weight_decay, num_epochs, semi_conduct_name, save_images, model_name):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.semi_conduct_name = semi_conduct_name

        assert model_name is not None, logger.error("Model name is not provided!")
        self.model_name = model_name

        self.save_test_images_flag = save_images
        if save_images:
            self.path_to_save = f"./models/{self.model_name.split('.')[0]}"
            if not os.path.exists(self.path_to_save):
                os.mkdir(self.path_to_save)

        self.writer = SummaryWriter(
            log_dir=f"./runs/runs_current/{semi_conduct_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            comment=f"LR_{self.lr}_BS_{self.batch_size}_WC_{self.weight_decay}")
        logger.info(f"Device: {self.device}")

    def __del__(self):
        self.writer.close()

    def create_vishay_network(self):
        # model = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        resnet50_flag = False

        if resnet50_flag:
            model = resnet18(weights="ResNet18_Weights.DEFAULT")
            logger.info("Network: Resnet50")
        else:
            model = resnet18(weights="ResNet18_Weights.DEFAULT")
            logger.info("Network: Resnet18")

        model.eval()
        # print(model.fc)

        model.fc = torch.nn.Linear(512, 2)
        if torch.cuda.is_available():
            model = model.cuda()

        self.model = model

    def create_dataloader(self, path_to_dataset):

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((512, 512)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((512, 512)),
                # transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((512, 512)),
                # transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        logger.info("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: ImageFolderWithPaths(os.path.join(path_to_dataset, x), data_transforms[x]) for x in
                          ['train', 'val', 'test']}
        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'val', 'test']}

        # test_datasets = ImageFolderWithPaths( os.path.join(r"C:\Users\A750290\Projects\Vishay\Data\cut\Input_cleansed", "test"), data_transforms["test"])
        # test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=4, shuffle=True, num_workers=4)

        self.dataloaders_dict = dataloaders_dict

    def load_model(self):
        if self.model is not None:
            self.model.load_state_dict(torch.load(f"./models/{self.model_name}"))
        else:
            logger.error("nothing to Load. Model is empty!")

    def save_model(self):
        target_path_model = f"./models/{self.model_name}"
        if self.model is not None:
            torch.save(self.model.state_dict(),
                       target_path_model.split(".pth")[0] + f"_acc_{self.best_acc_after_training:.2f}.pth")
        else:
            logger.error("nothing to Save. Model is empty!")

    def get_mean_std(self, path):
        # Compute the mean and standard deviation of all pixels in the dataset
        num_pixels = 0
        mean = 0.0
        std = 0.0

        for images, _ in self.dataloaders_dict['train']:
            batch_size, num_channels, height, width = images.shape
            num_pixels += batch_size * height * width
            mean += images.mean(axis=(0, 2, 3)).sum()
            std += images.std(axis=(0, 2, 3)).sum()

        mean /= num_pixels
        std /= num_pixels

        return mean, std

    def create_optimizer(self):
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            loss_dic = {}
            acc_dic = {}
            logger.info(f"Epoch {epoch}/{self.num_epochs - 1}")
            logger.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels, _paths in self.dataloaders_dict[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            # loss *= 3

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders_dict[phase].dataset)

                if phase == "train":
                    loss_dic["train"] = epoch_loss
                    acc_dic["train"] = epoch_acc.item()
                    # self.writer.add_scalar("Loss", loss_dic["train"], epoch)
                    # self.writer.add_scalar("Accuracy", acc_dic["train"], epoch)

                elif phase == "val":
                    loss_dic["val"] = epoch_loss
                    acc_dic["val"] = epoch_acc.item()
                    # self.writer.add_scalar("Loss", loss_dic["val"], epoch)
                    # self.writer.add_scalar("Accuracy", acc_dic["val"], epoch)

                logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            self.writer.add_scalars("Loss", loss_dic, epoch)
            self.writer.add_scalars("Accuracy", acc_dic, epoch)
            # self.writer.flush()

        time_elapsed = time.time() - since
        logger.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        logger.info(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.writer.flush()
        self.best_acc_after_training = best_acc

        return val_acc_history

    def test_model(self):
        running_accuracy = 0
        total = 0
        step = 0
        self.image_num = 0

        if self.model is None:
            logger.error("Load the model first. model is empty!")
        else:
            with torch.no_grad():
                for inputs, outputs, paths in self.dataloaders_dict['test']:

                    inputs = inputs.to(self.device)
                    outputs = outputs.to(self.device)
                    outputs = outputs.to(torch.float32)

                    predicted_outputs = self.model(inputs)
                    _, predicted = torch.max(predicted_outputs, 1)

                    total += outputs.size(0)
                    running_accuracy += (predicted == outputs).sum().item()

                    probs = [f.softmax(el, dim=0)[i].item() for i, el in zip(predicted, predicted_outputs)]

                    for idx in range(self.batch_size):
                        self.writer.add_figure('predictions vs. actuals',
                                               self.plot_classes_predictions(outputs[idx].item(),
                                                                             paths[idx],
                                                                             predicted[idx].item(),
                                                                             probs[idx],
                                                                             image_with_CAM=False,
                                                                             cam_image=None),
                                               global_step=step)
                        step += 1
                logger.info(f"Accuracy of the model on test data is: %{(100 * running_accuracy / total)} %%")

    def test_with_CAM(self):

        running_accuracy = 0
        total = 0
        step = 0
        self.image_num = 0

        if self.model is None:
            logger.error("Load the model first. model is empty!")
        else:
            cam = CAM(self.model, 'layer4', 'fc')
            for inputs, outputs, paths in self.dataloaders_dict['test']:

                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                outputs = outputs.to(torch.float32)
                with torch.no_grad():
                    predicted_outputs = self.model(inputs)

                _, predicted = torch.max(predicted_outputs, 1)
                activation_map = cam(predicted_outputs.squeeze(0).argmax().item(), predicted_outputs)

                # Resize the CAM and overlay it
                cam_image = overlay_mask(to_pil_image(read_image(paths[0])),
                                         to_pil_image(activation_map[0].squeeze(0), mode='F'),
                                         alpha=0.5)

                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

                probs = [f.softmax(el, dim=0)[i].item() for i, el in zip(predicted, predicted_outputs)]

                for idx in range(self.batch_size):
                    self.writer.add_figure('predictions vs. actuals',
                                           self.plot_classes_predictions(outputs[idx].item(),
                                                                         paths[idx],
                                                                         predicted[idx].item(),
                                                                         probs[idx],
                                                                         image_with_CAM=True,
                                                                         cam_image=cam_image),
                                           global_step=step)
                    step += 1
            logger.info(f"Accuracy of the model on test data is: %{(100 * running_accuracy / total)} %%")

    def confusion_matrix(self):

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels, paths in self.dataloaders_dict['test']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            labels = labels.to(torch.float32)

            predicted_outputs = self.model(inputs)

            output = (torch.max(torch.exp(predicted_outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        if len(self.classes) == 2:
            logging.info(f"TP: {cf_matrix[0, 0]}")
            logging.info(f"TN: {cf_matrix[1, 1]}")
            logging.info(f"FP: {cf_matrix[1, 0]}")
            logging.info(f"FN: {cf_matrix[0, 1]}")
            logging.info(f"Precision: {cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[1, 0])}")
            logging.info(f"Recall: {cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[0, 1])}")

        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in self.classes],
                             columns=[i for i in self.classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('output.png')

    def plot_classes_predictions(self, label, path, pred, prob, image_with_CAM, cam_image):
        """
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        """
        file_name = Path(path).stem
        if image_with_CAM:
            fig = plt.figure()
            fig.set_figheight(5)
            fig.set_figwidth(10)
            ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
            plt.imshow(plt.imread(path), interpolation='nearest')
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[pred],
                prob * 100.0,
                self.classes[int(label)]),
                color=("green" if pred == label else "red"))

            ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
            plt.imshow(cam_image, interpolation='nearest')
            ax.axis('off')
            ax.set_title(f"image name: \n{file_name}")

        else:
            fig = plt.figure()
            fig.set_figheight(15)
            fig.set_figwidth(15)
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
            plt.imshow(plt.imread(path), interpolation='nearest')
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[pred],
                prob * 100.0,
                self.classes[int(label)]),
                color=("green" if pred == label.item() else "red"))

        self.image_num += 1
        if self.save_test_images_flag:
            fig.savefig(os.path.join(self.path_to_save, f"IMAGE_{self.image_num}.png"))
        return fig


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


class DataAugVishay:
    # for image aug
    input_dir = None
    output_dir = None
    num_of_aug_img = None

    def __init__(self, input_path, output_path, num_of_aug_img: int = 20):
        logger.info("data aug started")
        self.input_dir = input_path
        self.output_dir = output_path
        self.num_of_aug_img = num_of_aug_img

    @staticmethod
    def divide_images(path):

        files = set(os.listdir(path))

        num_total = len(files)
        num_train = int(((num_total * 60) / 100))
        num_val = int(((num_total * 20) / 100))
        num_test = int(((num_total * 20) / 100))

        import random
        random.seed(3)
        random_train = set(random.sample(files, k=num_train))
        remained_files = files - random_train
        random_val = set(random.sample(remained_files, k=num_val))
        remained_files = remained_files - random_val
        random_test = set(random.sample(remained_files, k=num_test))

        dataset_dic = {
            "train": list(random_train),
            "val": list(random_val),
            "test": list(random_test)
        }
        for dataset in dataset_dic.keys():
            dataset_path = os.path.join(path, dataset)
            if not os.path.isdir(dataset_path):
                os.mkdir(dataset_path)
            for file in dataset_dic[dataset]:
                shutil.move(os.path.join(path, file), dataset_path)

    def data_aug_vishay(self):
        if self.input_dir and self.output_dir is None:
            logger.error("Input and output folder paths are None!")
        else:
            for i in range(0, self.num_of_aug_img):
                seq = iaa.Sequential([
                    iaa.Crop(px=(0, 50, 0, 50), keep_size=False),
                    iaa.Resize({"height": 512, "width": "keep-aspect-ratio"}),
                    # iaa.Pad(px=((50, 50), (0, 0), (50, 50), (0, 0)), pad_mode="edge"),
                    iaa.HorizontalFlip(0.5),
                    iaa.VerticalFlip(0.5),
                    iaa.TranslateX(px=(-20, 20), mode="edge"),
                    # iaa.TranslateY(px=(-20, 20), mode="edge")
                ], random_order=True)

                image_names = os.listdir(self.input_dir)
                images = self.read_images(image_names)
                images_aug = seq(images=images)
                self.save_images(images_aug, image_names, i)

                logger.info(f"{i + 1} is finished")

            del images
            del images_aug
            gc.collect()

    def read_images(self, image_names):
        images = []
        for image_name in image_names:
            img = imageio.imread(os.path.join(self.input_dir, image_name))
            images.append(img)
        return images

    def save_images(self, images, image_names, i):
        for u in range(len(images)):
            parts = image_names[u].split(".")
            imageio.imwrite(os.path.join(self.output_dir, f"{parts[0]}_aug_{str(i)}.{parts[1]}"), images[u])
