import sys
sys.path.append('..')
from src.data.sketchy_dataset import GAN_DataGeneratorPaired_for_sketchy
from src.options import Options
from src.model import Generator
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from src.unet_model import SRNet_BG
import torch.nn.functional as F



def main():
    ## testing IMAGE


    ## Prepare data
    args = Options().parse()
    train_dataset = GAN_DataGeneratorPaired_for_sketchy(args.photo_dir, args.sketch_dir, 'train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataset = GAN_DataGeneratorPaired_for_sketchy(args.photo_dir, args.sketch_dir, 'test')
    test_trained, _ = random_split(train_dataset, [8, len(train_dataset)-8])
    test_untrained, _ = random_split(test_dataset, [8, len(test_dataset)-8])
    test_total = ConcatDataset([test_trained, test_untrained])
    imglist = []
    sketchlist = []
    for i in range(len(test_total)):
        image, sketch = test_total[i]
        imglist.append(image)
        sketchlist.append(sketch)
    save_image(imglist, '../figure/' + 'image' + '.jpg', padding=5)
    save_image(sketchlist, '../figure/' + 'sketch' + '.png', padding=5)


    ## Init model
    model = SRNet_BG(4, 64)
    #model = Generator()
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda()


    model.train()
    for i in range(args.epoch):
        epoch_index = i + 1
        total_loss = 0.0
        for images, sketches in tqdm(train_dataloader, desc='Train epoch: {}'.format(epoch_index)):
            images, sketches = images.cuda(), sketches.cuda()
            images, sketches = Variable(images), Variable(sketches)
            optimizer.zero_grad()
            outputs = model(images)
            batch_loss = loss(outputs, sketches)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss


        info = 'Epoch: {}, train loss: {:.3f}'.format(epoch_index, total_loss/len(train_dataloader))
        output_test_list = []
        testloader = DataLoader(dataset=test_total, batch_size=len(test_total), shuffle=False)
        for _, datas in enumerate(testloader):
            tesimage, xx = datas
            tesimage = Variable(tesimage.cuda())
            output_test = model(tesimage)
        save_image(output_test,
                   '../figure/' + 'sketch_epoch_' + str(
                       epoch_index) + '.png', padding=5)
        # for indexoftes in range(len(test_total)):
        #     tesimage, _ = test_total[indexoftes]
        #     tesimage.unsqueeze_(0)
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     tesimage = tesimage.to(device)
        #     output_test = model(tesimage)
        #     output_test_list.append(output_test)









if __name__ == '__main__':
    main()


