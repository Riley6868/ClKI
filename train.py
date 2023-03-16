import os
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd

from my_dataset import MyDataSet
from model import convnext_small as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

def load_train_data():
    train_base_path = ''
    train_fila_path = []
    train_file_label = []
    path = ''
    for line in open(path):  # 逐行打开文档.
        line = line.strip()  # 去除这一行的头和尾部空格.
        data = line.split(' ', 1)  # 切片的运算,以逗号为分隔,隔成两个
        data[0] = train_base_path + data[0]
        train_fila_path.append(data[0])
        train_file_label.append(int(data[1]))
    return train_fila_path,train_file_label
def load_val_data():
    val_base_path =''
    val_fila_path = []
    val_file_label = []
    path = ''
    for line in open(path):  # 逐行打开文档.
        line = line.strip()  # 去除这一行的头和尾部空格.
        data = line.split(' ', 1)  # 切片的运算,以逗号为分隔,隔成两个
        data[0] = val_base_path + data[0]
        val_fila_path.append(data[0])
        val_file_label.append(int(data[1]))
    return val_fila_path, val_file_label

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    train_images_path, train_images_label = load_train_data()
    val_images_path, val_images_label = load_val_data()

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   # int(img_size * 1.143)
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn
                                              )

    model = create_model(args,num_classes=args.num_classes).to(device)
    total=sum([param.nelement() for param in model.parameters()])
    print("num:%.2fm" % (total/1e6))




    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    Epoch = []
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []
    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(args=args,
                                                model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler,
                                                writer=tb_writer)

        # validate
        val_loss, val_acc = evaluate(args=args,
                                     model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     writer=tb_writer)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    

        Epoch.append(epoch)
        Train_loss.append(train_loss)
        Train_acc.append(train_acc)
        Val_loss.append(val_loss)
        Val_acc.append(val_acc)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model1.pth")
            best_acc = val_acc

    sa=pd.DataFrame({'epoch':Epoch,'train_loss':Train_loss,'train_acc':Train_acc,'val_loss':Val_loss,'val_acc':Val_acc})
    sa.to_csv('',index=None,encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default='')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')


    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    parser.add_argument('--alpha', default=0.2, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')

    opt = parser.parse_args()

    main(opt)
