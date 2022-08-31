import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
import CancerModel as cm
import config
import math
import os

class CancerModel(nn.Module):

    def __init__(self):
        super(CancerModel, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = cm.CancerModel()
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = .7)
        self.epoch = 0

        if (len(os.listdir(self.checkpoint_dir)) > 0):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']

        


    def train(self, num_epochs):
        loss_text_file = open(self.model_dir + "/train_losses.txt", 'a')
        val_loss_file = open(self.model_dir + "/validation_losses.txt", 'a')


        optimizer = self.optimizer
        transform = tf.Compose([
            tf.Resize((50, 50)),
            tf.ToTensor()
        ])
        train_data = torchvision.datasets.ImageFolder(config.TRAIN_PATH, transform)
        n_train = len(train_data)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size, shuffle = True, num_workers = 0)
        num_train_batches = math.ceil(n_train/self.batch_size)


        val_data = torchvision.datasets.ImageFolder(config.VAL_PATH, transform)
        n_val = len(val_data)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = self.batch_size, shuffle = True, num_workers = 0)
        num_val_batches = math.ceil(n_val/self.batch_size)

        loss_fn = torch.nn.BCELoss()


        for x in range(1, num_epochs + 1):
            running_loss = 0
            progress = 0


            #training loop
            for i, data, in enumerate(train_loader):
                if (((i * self.batch_size) / n_train) * 100 >= progress):
                    print(str(progress) + "%", flush = True)
                    progress += 25


                inputs, labels = data
                labels = labels.float()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                

                optimizer.zero_grad()

                pred = self.model(inputs)

                loss = loss_fn(pred, labels)
                #print(loss.item())

                loss.backward()

                running_loss += loss.item() * inputs.shape[0]
                optimizer.step()

            #validation loop
            print("running validation...")
            running_val_loss = 0
            for i, data in enumerate(val_loader):
                inputs, labels = data
                labels = labels.float()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                pred = self.model(inputs)
                loss = loss_fn(pred, labels)

                running_val_loss += loss.item() * inputs.shape[0]

            checkpoint = {
                'epoch': x + self.epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            running_loss /= n_train
            running_val_loss /= n_val

            print("epoch " + str(x + self.epoch) + ": training loss = " + str(running_loss))
            print("validation loss: " + str(running_val_loss))

            torch.save(self.model.state_dict(), self.epochs_dir + "/epoch" + str(x + self.epoch))
            torch.save(checkpoint, self.checkpoint_path)

            val_loss_file.write(str(running_val_loss))
            val_loss_file.write('\n')

            loss_text_file.write(str(running_loss))
            loss_text_file.write('\n')


    def test(self, epoch = 'checkpoint'):
        transform = tf.Compose([
            tf.Resize((50, 50)),
            tf.ToTensor()
        ])

        test_model = cm.CancerModel()
        if (epoch == 'checkpoint'):
            checkpoint = torch.load(self.checkpoint_path)
            test_model.load_state_dict(checkpoint['state_dict'])
        else:
            test_model.load_state_dict(torch.load(self.epochs_dir + "/" + epoch))


        test_data = torchvision.datasets.ImageFolder(config.TEST_PATH, transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = self.batch_size, shuffle = True, num_workers = 0)
        n_test = len(test_data)
        loss_fn = torch.nn.BCELoss()

        running_loss = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            labels = labels.float()
            inputs, labels = inputs.to(self.device), labels.to(self.device)


            pred = self.model(inputs)
            loss = loss_fn(pred, labels)

            running_loss += loss.item() * inputs.shape[0]

            

        return running_loss/n_test

        
    def generate_confusion_matrix(self):
        return None

    def generate_loss_plots(self):
        return None