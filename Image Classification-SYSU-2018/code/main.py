import csv, os, shutil, time, copy, torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image


def mymovefile(srcfile, dstfile, type):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        if type == "move":
            shutil.move(srcfile, dstfile)
            print("move %s -> %s" % (srcfile, dstfile))
        else:
            shutil.copy(srcfile, dstfile)
            print("copy %s -> %s" % (srcfile, dstfile))


def ClassifyImage():
    csv_reader = csv.reader(open('../data/train.csv', encoding='utf-8'))
    for row in csv_reader:
        JpgFilePath = row[0]
        Label = row[1]
        srcfile = '../data/image/train/' + JpgFilePath
        dstfile = '../data/image-/train/' + Label + '/' + JpgFilePath
        mymovefile(srcfile, dstfile, "move")


def SplitTrainAndTest():
    SrcDir = "../data/image-/train"
    SplitTrainDir = "../data/image-/train1"
    SplitTestDir = "../data/image-/test1"
    if os.path.exists(SplitTrainDir) == False:
        os.makedirs(SplitTrainDir)
    if os.path.exists(SplitTestDir) == False:
        os.makedirs(SplitTestDir)

    LabelList = os.listdir(SrcDir)
    for Label in LabelList:
        PerLabelSetDir = os.path.join(SrcDir, Label)
        ImgList = os.listdir(PerLabelSetDir)
        TrainImgList, TestImgList = train_test_split(ImgList, test_size=0.2, random_state=0)
        SplitPerLabelTrainSetDir = os.path.join(SplitTrainDir, Label)
        SplitPerLabelTestSetDir = os.path.join(SplitTestDir, Label)
        if os.path.exists(SplitPerLabelTrainSetDir) == False:
            os.makedirs(SplitPerLabelTrainSetDir)
        if os.path.exists(SplitPerLabelTestSetDir) == False:
            os.makedirs(SplitPerLabelTestSetDir)

        for TrainImg in TrainImgList:
            SrcFile = os.path.join(PerLabelSetDir, TrainImg)
            DestFile = os.path.join(SplitPerLabelTrainSetDir, TrainImg)
            mymovefile(SrcFile, DestFile, "copy")

        for TestImg in TestImgList:
            SrcFile = os.path.join(PerLabelSetDir, TestImg)
            DestFile = os.path.join(SplitPerLabelTestSetDir, TestImg)
            mymovefile(SrcFile, DestFile, "copy")


def train_model(device, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        starttime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(' ')
        endtime = time.time()
        print((endtime-starttime) / 60, "min")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def TrainProcess(dataloaders):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 本次比赛主要采取ResNet101模型， 如果计算条件允许，可以采用ResNet152模型进行尝试
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 8)
    model_ft.fc.weight.data.normal_(0, 0.01)
    model_ft.fc.bias.data.zero_()
    model_ft = model_ft.to(device)

    # 如果需要在已经训练好的模型继续进行训练，则使用下面注释的语句进行训练
    # model_ft = torch.load('model_accuracy.pkl')
    # model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train and evaluate
    model_ft = train_model(device, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=40)

    torch.save(model_ft, '../model/model.pkl')

def PredictProcess():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## 利用之前的训练模型进行预测
    model_ft = torch.load('../model/model.pkl')
    model_ft = model_ft.to(device)

    root = '../data/image-/test/test/'
    answer = []
    for file in os.listdir(root):
        img = Image.open(root + file)
        img = img.convert('RGB')
        img = data_transforms['val'](img)
        img.unsqueeze_(0)

        inputs = img.to(device)
        outputs = model_ft(inputs)
        # sm_outputs = F.Softmax(outputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.data.cpu().numpy()

        answer.append([file, preds[0]])
    # print(len(answer))

    header = [['Image', 'Cloth_label']]
    np.savetxt('submission.csv', header, delimiter=',', fmt='%s')
    with open('submission.csv', 'ab') as f:
        np.savetxt(f, answer, delimiter=',', fmt='%s,%s')


if __name__== "__main__":
    ClassifyImage()
    SplitTrainAndTest()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 训练集和测试集的图片处理方式需要不一样，提高泛化能力
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    }
    # 训练集使用之前切割出来的训练集和测试集进行训练
    trainset = ImageFolder("../data/image-/train1", transform=data_transforms['train'])
    valset = ImageFolder("../data/image-/test1", transform=data_transforms['val'])

    dataset_sizes = {'train': len(trainset), 'val': len(valset)}
    dataloaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=8,
                                                        shuffle=True, num_workers=1),
                   'val': torch.utils.data.DataLoader(valset, batch_size=8,
                                                      shuffle=True, num_workers=1)}

    TrainProcess(dataloaders)
    PredictProcess()
