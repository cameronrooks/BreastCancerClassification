from imutils import paths
import random, shutil, os
import config

input_paths = list(paths.list_images('data/original'))
random.seed(12)
random.shuffle(input_paths)

num_files = len(input_paths)
split_index = int(config.TRAIN_SPLIT * num_files)

train_paths = input_paths[:split_index]
test_paths = input_paths[split_index:]

print(len(train_paths))
print(len(test_paths))

val_split_index = int((1 -config.VAL_SPLIT) * split_index)
val_paths = train_paths[val_split_index:]
train_paths = train_paths[:val_split_index]

print(len(train_paths))
print(len(val_paths))

datasets = [("train", train_paths, config.TRAIN_PATH), ("validation", val_paths, config.VAL_PATH), ("test", test_paths, config.TEST_PATH)]

for (name, paths, dir_path) in datasets:
    if (not os.path.exists(dir_path)):
        os.mkdir(dir_path)

    count = 0
    for path in paths:
        #get the index of the class number from the file path
        class_index = len(path) - 5
        class_ = int(path[class_index])

        new_dir_path = dir_path + "/" + str(class_)

        if (not os.path.exists(new_dir_path)):
            os.mkdir(new_dir_path)

        new_path = new_dir_path + "/" + str(count) + ".png"
        count += 1

        shutil.copy2(path, new_path)





