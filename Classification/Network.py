from __future__ import print_function
from __future__ import division

import gc
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50

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

    def __init__(self, batch_size, lr, weight_decay, num_epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(log_dir=f"./runs/runs_current/{datetime.now().strftime('%Y-%m-%d_%H-%M')}", comment=f"LR_{self.lr}_BS_{self.batch_size}_WC_{self.weight_decay}")
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
        image_datasets = {x: ImageFolderWithPaths(os.path.join(path_to_dataset, x), data_transforms[x]) for x in ['train', 'val', 'test']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

        # test_datasets = ImageFolderWithPaths( os.path.join(r"C:\Users\A750290\Projects\Vishay\Data\cut\Input_cleansed", "test"), data_transforms["test"])
        # test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=4, shuffle=True, num_workers=4)

        self.dataloaders_dict = dataloaders_dict

    def load_model(self, path_to_saved_model):
        # torch.save(model.state_dict(), path)
        if self.model is not None:
            self.model.load_state_dict(torch.load(path_to_saved_model))
        else:
            logger.error("nothing to Load. Model is empty!")

    def save_model(self, target_path_model: str):
        if self.model is not None:
            torch.save(self.model.state_dict(), target_path_model.split(".pth")[0] + f"_acc_{self.best_acc_after_training:.2f}.pth")
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
        if self.model is None:
            logger.error("Load the model first. model is empty!")
        else:
            with torch.no_grad():
                for inputs, outputs, paths in self.dataloaders_dict['test']:
                    # now_before = datetime.utcnow()

                    inputs = inputs.to(self.device)
                    outputs = outputs.to(self.device)
                    outputs = outputs.to(torch.float32)

                    predicted_outputs = self.model(inputs)
                    _, predicted = torch.max(predicted_outputs, 1)

                    # now_after = datetime.utcnow()

                    # logger.info(f"time before: {now_before}")
                    # logger.info(f"time after: {now_after}")
                    # logger.info(f"time diff: {now_after - now_before}")

                    total += outputs.size(0)
                    running_accuracy += (predicted == outputs).sum().item()

                    self.writer.add_figure('predictions vs. actuals', self.plot_classes_preds(inputs.to("cpu"), outputs.to("cpu"), paths), global_step=step)

                    step += 1
                logger.info(f"Accuracy of the model on test data is: %{(100 * running_accuracy / total)} %%")

    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
    @staticmethod
    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def images_to_probs(self, images):
        """
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        """
        output = self.model(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        if preds_tensor.shape != 1:
            preds = np.squeeze(preds_tensor.numpy())

        return preds, [f.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, images, labels, paths):
        """
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        """
        self.model.to("cpu")
        preds, probs = self.images_to_probs(images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure()
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            # matplotlib_imshow(images[idx], one_channel=False)
            plt.imshow(plt.imread(paths[idx]), interpolation='nearest')
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[int(labels[idx].item())]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))

        self.model.to(self.device)
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
                    iaa.Crop(px=(0, 80, 0, 80), keep_size=False),
                    iaa.Resize({"height": 512, "width": "keep-aspect-ratio"}),
                    # iaa.Pad(px=((50, 50), (0, 0), (50, 50), (0, 0)), pad_mode="edge"),
                    iaa.HorizontalFlip(0.5),
                    iaa.VerticalFlip(0.5),
                    iaa.TranslateX(px=(-20, 20), mode="edge"),
                    iaa.TranslateY(px=(-20, 20), mode="edge")
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
