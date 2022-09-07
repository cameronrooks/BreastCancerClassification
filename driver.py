from CancerModel import ModelDriver
import sys
import config

driver = ModelDriver(batch_size = 512, learning_rate = .01, model_name = "model-01512")

if (sys.argv[1] == "train"):
    driver.train(int(sys.argv[2]))
elif(sys.argv[1] == "plot"):
    driver.generate_loss_plots()
    driver.generate_confusion_matrix()
elif(sys.argv[1] == "test"):
    if (len(sys.argv) > 2):
        print(driver.test(sys.argv[2]))
    else:
        print(driver.test())