import argparse
import os
import logging
import mindspore
import time
from mindspore.dataset import vision, transforms
from mindspore import nn
from LeNet5 import LeNet5, train, test


def main():
    opt_parser = argparse.ArgumentParser(prog='LeNet Training')
    opt_parser.add_argument('--train_dataset', type=str, default='', help='path for training dataset')
    opt_parser.add_argument('--test_dataset', type=str, default='', help='path for test dataset')
    opt_parser.add_argument('--epoch', type=int, default=5, help='number of epoch')
    opt_parser.add_argument('--lrate', type=float, default=1e-2, help='learning rate')
    opt_parser.add_argument('--batchsize', type=int, default=64, help='batch size')
    opt_parser.add_argument('--useMnist', action='store_true', help='use MnistDataset or not, default is False')
    opt_parser.add_argument('--useGPU', action='store_true', help='use GPU or not, default is False')
    opt_parser.add_argument('--num_class', type=int, default=10, help='number of class')
    opt_parser.add_argument('--num_channel', type=int, default=1, help='number of channel')
    opt_parser.add_argument('--name', type=str, default='LetNet', help='name of this training process')

    # opt = opt_parser.parse_args()
    opt = opt_parser.parse_args('--useMnist --epoch 20 --useGPU --lrate 0.01 --name LeNet5onMnist_03'.split())

    # create directory for this trainning process
    if not os.path.exists(f"checkpoints/{opt.name}/"):
        os.makedirs(f"checkpoints/{opt.name}/")
    
    log_file_path = f"checkpoints/{opt.name}/{opt.name}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.log"
    logging.basicConfig(filename=log_file_path, encoding='utf8', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    logging.info(f"learning rate: {opt.lrate}")

    if not opt.useMnist:
        if not os.path.exists(opt.train_dataset):
            print(f"training set [{opt.train_dataset}] doesn't exist!")
            exit(-1)
        if not os.path.exists(opt.test_dataset):
            print(f"test set [{opt.test_dataset}] doesn't exist!")
            exit(-2)
        # train_dataset = 
        # test_dataset = 
    else:
        from mindspore.dataset import MnistDataset
        if not os.path.exists('./MNIST_Data/'):
            from download import download
            url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
            path = download(url, "./", kind="zip", replace=True)
        train_dataset = MnistDataset('./MNIST_Data/train', shuffle=True)
        test_dataset = MnistDataset('./MNIST_Data/test', shuffle=False)

    train_dataset = datapipe(train_dataset, opt.batchsize)
    test_dataset = datapipe(test_dataset, opt.batchsize)

    if opt.useGPU:
        mindspore.set_context(device_target='GPU')
    mindspore.set_seed(sum((ord(i) for i in 'huawei_nb')))

    model = LeNet5(opt.num_class, opt.num_channel)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.trainable_params(), learning_rate=opt.lrate)

    # if there are some pretrained weights, load them
    current_max_epoch = load_params(f"./checkpoints/{opt.name}", model, 'ckpt', logging)

    for t in range(current_max_epoch, opt.epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train(model, train_dataset, loss_fn, optimizer, logging)
        test(model, test_dataset, loss_fn, logging)
        save_model(model, f"./checkpoints/{opt.name}/{t+1}.ckpt")
    print("Done!")

def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0/255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)
    dataset = dataset.map(image_transforms, dataset.get_col_names()[0])
    dataset = dataset.map(label_transform, dataset.get_col_names()[1])
    dataset = dataset.batch(batch_size)

    return dataset

def save_model(model, store_path, logging=None):
    store_dir = os.path.dirname(store_path)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    mindspore.save_checkpoint(model, store_path)
    log_string = f"Saved model to {store_path}"
    print(log_string)
    if logging is not None:
        logging.info(log_string)

def load_params(folder, model, extension_name, logging=None):
    from pathlib import Path
    param_file_list = [path.name for path in Path(folder).glob(f'*.{extension_name}')]
    if len(param_file_list) != 0:
        import re
        pattern = re.compile(r'(\d+).' + extension_name)
        current_max_epoch = max(int(re.match(pattern, pth_name).group(1)) for pth_name in param_file_list)
        params_path = os.path.join(folder, f"{current_max_epoch}.{extension_name}")
        param_dict = mindspore.load_checkpoint(params_path)
        param_not_load = mindspore.load_param_into_net(model, param_dict)
        if not param_not_load:
            log_string = f"loading [{params_path}] to model..."
            print(log_string)
            if logging is not None:
                logging.info(log_string)
        else:
            log_string = 'some parameters were failed to load...'
            print(log_string)
            if logging is not None:
                logging.error(log_string)
                exit(1)
    else:
        current_max_epoch = 0
    
    return current_max_epoch

if __name__ == '__main__':
    main()
    