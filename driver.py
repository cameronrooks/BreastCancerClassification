import torchvision
import torchvision.transforms as tf
import CancerModel as cm

transform = tf.ToTensor()
data = torchvision.datasets.ImageFolder("./data/processed/train", transform)


model = cm.CancerModel(None, None, None)

model(data[0][0])

