from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from datetime import datetime

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ("NiO", "iO")


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper functions
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    if preds_tensor.shape != (1):
        preds = np.squeeze(preds_tensor.numpy())

    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, paths):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        # matplotlib_imshow(images[idx], one_channel=False)
        plt.imshow(plt.imread(paths[idx]), interpolation='nearest')
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[int(labels[idx].item())]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def train_model(model, t_writer, dataloaders, criterion, optimizer, num_epochs=1, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        loss_dic = {}
        acc_dic = {}
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == "train":
                loss_dic["train"] = epoch_loss
                acc_dic["train"] = epoch_acc.item()
            elif phase == "val":
                loss_dic["val"] = epoch_loss
                acc_dic["val"] = epoch_acc.item()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        t_writer.add_scalars("Loss", loss_dic, epoch)
        t_writer.add_scalars("Accuracy", acc_dic, epoch)
        t_writer.flush()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    t_writer.flush()

    return model, val_acc_history


def test_model(tester_model, t_writer, test_loader_data):
    running_accuracy = 0
    total = 0
    step = 0
    with torch.no_grad():
        for data in test_loader_data:
            now_before = datetime.utcnow()
            inputs, outputs, paths = data
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            outputs = outputs.to(torch.float32)
            predicted_outputs = tester_model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            now_after = datetime.utcnow()
            print("time before: ", now_before)
            print("time after: ", now_after)
            print("time diff: ", now_after - now_before)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

            t_writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(model.to("cpu"), inputs.to("cpu"), outputs.to("cpu"), paths),
                                global_step=step)
            step += 1
            model.to(device)
            inputs.to(device)
            outputs.to(device)
        print('Accuracy of the model on test data is: %d %%' % (100 * running_accuracy / total))


if __name__ == "__main__":
    # print(torch.hub.list('pytorch/vision', force_reload=True))
    writer = SummaryWriter()

    # model = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    model = resnet18(weights="ResNet18_Weights.DEFAULT")

    model.eval()
    print(model.fc)

    model.fc = torch.nn.Linear(512, 2)
    model = model.cuda()

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

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join("input", x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for
                        x in ['train', 'val']}

    optimizer_ft = optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    # model, hist = train_model(model, writer, dataloaders_dict, criterion, optimizer_ft, num_epochs=30)

    test_datasets = ImageFolderWithPaths(
        os.path.join(r"C:\Users\A750290\Projects\Vishay\Data\cut\Input_cleansed", "test"), data_transforms["test"])
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=4, shuffle=True, num_workers=4)

    path = "./NetModel_first best.pth"
    # torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load("NetModel_first best.pth"))
    test_model(model, writer, test_loader)
    writer.close()