import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
import CancerModel as cm
import config
import os

class CancerModel(nn.Module):

    def __init__(self):
        super(CancerModel, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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


        optimizer = self.optimizer
        transform = tf.Compose([
            tf.Resize((50, 50)),
            tf.ToTensor()
        ])
        data = torchvision.datasets.ImageFolder(config.TRAIN_PATH, transform)
        n = len(data)
        train_loader = torch.utils.data.DataLoader(data, batch_size = self.batch_size, shuffle = True, num_workers = 0)
        loss_fn = torch.nn.BCELoss()

        for x in range(1, num_epochs + 1):
            running_loss = 0
            progress = 0

            for i, data, in enumerate(train_loader):
                if (((i * self.batch_size) / n) * 100 >= progress):
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

                running_loss += loss.item()
                optimizer.step()

            checkpoint = {
                'epoch': x,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            print("epoch " + str(x + self.epoch) + ": loss = " + str(running_loss))
            torch.save(self.model.state_dict(), self.epochs_dir + "/epoch" + str(x + self.epoch))
            torch.save(checkpoint, self.checkpoint_path)

            loss_text_file.write(str(running_loss))
            loss_text_file.write('\n')