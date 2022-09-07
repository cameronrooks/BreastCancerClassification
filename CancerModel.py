import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
import pandas as pd
import numpy as np
import CancerModel as cm
import config
import seaborn as sn
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

class CancerModel(nn.Module):

    def __init__(self):
        super(CancerModel, self).__init__()

        #convolutional layer 1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding = "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(.3)
        )

        #convolutional layer 2
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding = "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding = "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(.3)
        )

        #convolutional layer 3
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding = "same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, padding = "same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(.3)
        )

        #fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(4608, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(.3)
        )

        #output layer
        self.output = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.output(x)

        output = x.reshape(-1)
  
        return output


#class that interacts with the CancerModel
class ModelDriver():
    def __init__(self, batch_size, learning_rate, model_name):

        #initialize hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        #initialize necessary directories
        self.model_dir = "./models/" + model_name
        self.epochs_dir = self.model_dir + "/epochs"
        self.plots_dir = self.model_dir + "/plots"
        self.checkpoint_dir = self.model_dir + "/checkpoint"
        self.checkpoint_path = self.checkpoint_dir + "/checkpoint.pt"


        #create all directories if they don't exist
        if (not os.path.exists("./models")):
            os.mkdir("models")

        if (not os.path.exists(self.model_dir)):
            os.mkdir(self.model_dir)

        if (not os.path.exists(self.epochs_dir)):
            os.mkdir(self.epochs_dir)

        if (not os.path.exists(self.plots_dir)):
            os.mkdir(self.plots_dir)

        if (not os.path.exists(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)


        #set up model and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = cm.CancerModel()
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = .7)
        self.epoch = 0

        #load optimizer and model state dicts from checkpoint if it exists
        if (len(os.listdir(self.checkpoint_dir)) > 0):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']

        self.test_loader = None
        self.train_loader = None
        self.val_loader = None

        self.n_val = 0
        self.n_test = 0
        self.n_train = 0


    def load_train_data(self):
        transform = tf.Compose([
            tf.Resize((50, 50)),
            tf.ToTensor()
        ])
        train_data = torchvision.datasets.ImageFolder(config.TRAIN_PATH, transform)
        self.n_train = len(train_data)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size, shuffle = True, num_workers = 0)

        val_data = torchvision.datasets.ImageFolder(config.VAL_PATH, transform)
        self.n_val = len(val_data)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size = self.batch_size, shuffle = True, num_workers = 0)

        return 0

    def load_test_data(self):
        transform = tf.Compose([
            tf.Resize((50, 50)),
            tf.ToTensor()
        ])
        test_data = torchvision.datasets.ImageFolder(config.TEST_PATH, transform)
        self.n_test = len(test_data)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size = self.batch_size, shuffle = True, num_workers = 0)

        return 0


    def train(self, num_epochs):

        #open text files to log the validation and training losses
        loss_text_file = open(self.model_dir + "/train_losses.txt", 'a')
        val_loss_file = open(self.model_dir + "/validation_losses.txt", 'a')

        optimizer = self.optimizer

        if (self.train_loader == None):
            self.load_train_data()

        #initialize loss funtion to Binary Cross Entropy Loss
        loss_fn = torch.nn.BCELoss()


        for x in range(1, num_epochs + 1):
            running_loss = 0
            progress = 0


            #training loop
            for i, data, in enumerate(self.train_loader):

                #print current epoch progress
                if (((i * self.batch_size) / self.n_train) * 100 >= progress):
                    print(str(progress) + "%", flush = True)
                    progress += 25

                #get inputs and labels
                inputs, labels = data
                labels = labels.float()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                #zero the gradient
                optimizer.zero_grad()

                #get predictions from model
                pred = self.model(inputs)


                #calculate loss and perform backward pass
                loss = loss_fn(pred, labels)
                loss.backward()

                running_loss += loss.item() * inputs.shape[0]
                optimizer.step()

            #validation loop
            print("running validation...")
            running_val_loss = 0
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                labels = labels.float()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                pred = self.model(inputs)
                loss = loss_fn(pred, labels)

                running_val_loss += loss.item() * inputs.shape[0]

            #create new checkpoint for current epoch
            checkpoint = {
                'epoch': x + self.epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            running_loss /= self.n_train
            running_val_loss /= self.n_val

            print("epoch " + str(x + self.epoch) + ": training loss = " + str(running_loss))
            print("validation loss: " + str(running_val_loss))

            #save epoch and checkpoint to respective directories
            torch.save(self.model.state_dict(), self.epochs_dir + "/epoch" + str(x + self.epoch))
            torch.save(checkpoint, self.checkpoint_path)

            val_loss_file.write(str(running_val_loss))
            val_loss_file.write('\n')

            loss_text_file.write(str(running_loss))
            loss_text_file.write('\n')


    def test(self, epoch = 'checkpoint'):

        test_model = cm.CancerModel()
        test_model.to(self.device)
        if (epoch == 'checkpoint'):
            checkpoint = torch.load(self.checkpoint_path)
            test_model.load_state_dict(checkpoint['state_dict'])
        else:

            test_model.load_state_dict(torch.load(self.epochs_dir + "/" + epoch))

        test_model.eval()
        loss_fn = torch.nn.BCELoss()

        running_loss = 0

        correct_pos = 0
        correct_neg = 0
        false_pos = 0
        false_neg = 0
        
        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            labels = labels.float()
            inputs, labels = inputs.to(self.device), labels.to(self.device)


            pred = test_model(inputs)
            loss = loss_fn(pred, labels)

            running_loss += loss.item() * inputs.shape[0]

            

        return running_loss/self.n_test

        
    def generate_confusion_matrix(self, epoch = 'checkpoint'):
        if (self.test_loader == None):
            self.load_test_data()

        test_model = cm.CancerModel()
        test_model.to(self.device)
        if (epoch == 'checkpoint'):
            checkpoint = torch.load(self.checkpoint_path)
            test_model.load_state_dict(checkpoint['state_dict'])
        else:

            test_model.load_state_dict(torch.load(self.epochs_dir + "/" + epoch))

        test_model.eval()

        y_pred = []
        y_true = []

        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            labels = labels.float()
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            pred = test_model(inputs)
            pred = torch.round(pred)
            pred = pred.data.cpu().numpy()
            y_pred.extend(pred)

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

        classes = ('Negative', ' Positive')

        matrix = confusion_matrix(y_true, y_pred)


        df_cm = pd.DataFrame(matrix/matrix.astype(np.float).sum(axis=1)[:,None], index = [i for i in classes], columns = [i for i in classes])
        #df_cm = pd.DataFrame(matrix, index = classes, columns = classes)


        plt.figure(figsize = (12, 7))
        ax = sn.heatmap(df_cm, annot=True)
        ax.set_xlabel('Predicted values')
        ax.set_ylabel('Actual values')

        plt.savefig(self.plots_dir + '/confusion_matrix.png')
        plt.show()
            

    def generate_loss_plots(self):
        if (self.test_loader == None):
            self.load_test_data()

        epochs = os.listdir(self.epochs_dir)

        test_losses = [None] * len(epochs)
        train_losses = []
        validation_losses = []

        train_file = open(self.model_dir + '/train_losses.txt', 'r')
        val_file = open(self.model_dir + '/validation_losses.txt', 'r')

        for line in train_file:
            line = line.strip()
            train_losses.append(float(line))

        for line in val_file:
            line = line.strip()
            validation_losses.append(float(line))

        count = 0
        for epoch in epochs:
            epoch_num = int(epoch[5:]) -1
            loss = self.test(epoch)
            test_losses[epoch_num] = loss

            count += 1
            print(count)


        x_vals = range(1, self.epoch + 1)

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        ax1.plot(x_vals, train_losses, color = 'blue', label = 'training loss')
        ax1.plot(x_vals, validation_losses, color = 'orange', label = 'validation loss')
        ax1.set_title("Training and Validation Loss vs Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.figure.savefig(self.plots_dir + '/train_val_loss.png')


        ax2.plot(x_vals, train_losses, color = 'blue', label = 'training loss')
        ax2.plot(x_vals, test_losses, color = 'orange', label = 'test loss')
        ax2.set_title("Training and Test Loss vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.figure.savefig(self.plots_dir + '/train_test_loss.png')

        ax3.plot(x_vals, test_losses, color = 'blue', label = 'test loss')
        ax3.set_title("Test Loss vs Epoch")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.figure.savefig(self.plots_dir + '/train_loss.png')

        plt.show()
        

        return None