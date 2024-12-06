import time
import argparse
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
# import spikingjelly as s
from spikingjelly.activation_based import neuron, encoding, surrogate, layer, functional
import seaborn as sns
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from snn import SNN
import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
from input_rank import rank_input_fun


parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cuda:7', help='device')
parser.add_argument('-b', default=64, type=int, help='batch size')
parser.add_argument('-epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-data-dir', type=str, default='./data', help='root dir of MNIST dataset')
parser.add_argument('-out-dir', type=str, default='/home/gaoheng/CODE_ALL/logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-amp', default=True, help='automatic mixed precision training')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

args = parser.parse_args()  
# 初始化数据加载器
train_dataset = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
train_data_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.b,
    shuffle=True,
    drop_last=True,
    num_workers=args.j,
    pin_memory=True  # True
)
test_data_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.b,
    shuffle=False,
    drop_last=False,
    num_workers=args.j,
    pin_memory=True  # True
)


def main():
    '''
        主函数, 用于训练SNN
    '''
    net = SNN(tau=args.tau)
    print(net)
    net.to(args.device)
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in tqdm(train_data_loader):
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max_15epoch_control.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest_15epoch_control.pth'))

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    net.eval()
    # regiter hooks
    output_layer = net.layer3[-1] # output layer
    output_layer.v_seq = []
    output_layer.s_seq = []
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)


    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy",v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy",s_t_array)



def weight_pruning(model, thr, del_type="weak"):
    '''
        基于权重幅值的剪枝，不带重新训练
    '''
    zero_vec = torch.zeros([1200])
    zero_rows = 0
    # for (name, module) in model.named_modules():
    #     print(name)
    
    # for (name, param) in model.named_parameters():
    #     print(name)
        
    for (name, module) in model.named_modules():
        if name == "layer1.1":
            weight_size = module.weight.data.shape
            weights = module.weight.data
            weight_sums = torch.abs(torch.sum(weights, axis=0))
            # plt.hist(weight_sums.cpu().numpy(), density=True, edgecolor='k')
            # plt.savefig('./weight_hist_plot.png', dpi=700, bbox_inches='tight')
            if del_type == "weak":
                for i in range(weight_size[1]):
                    if weight_sums[i] < 20:
                        module.weight.data[:, i] = zero_vec  # set zero
            elif del_type == "mid":
                for i in range(weight_size[1]):
                    if weight_sums[i] >= 20 and weight_sums[i] < 50:
                        module.weight.data[:, i] = zero_vec  # set zero
            else:  
                # delete the hub neurons 
                for i in range(weight_size[1]):
                    if weight_sums[i] >= 50:
                        module.weight.data[:, i] = zero_vec  # set zero 
            
            # print(module)    
            w_new =  module.weight.data
            w_sum2 = torch.abs(torch.sum(w_new, axis=0)) 
            for ii in range(784):
                if w_sum2[ii] == 0:
                    zero_rows += 1    # 代表着被敲除的神经元
            ratio = zero_rows / 784     # 代表被敲除神经元占首层神经元的百分比      
        else:
            pass
    
    # num_params = sum([param.nelement() for param in model.parameters()])  
    # print("The number of parameters after prunning is : {}".format(num_params))
    
    return zero_rows, ratio  # zero_rows is the number of prunned edge


def eval(model):
    '''
        模型精度评估函数
    '''
    encoder = encoding.PoissonEncoder()
    
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_data_loader):         
            out_fr = 0.
            image = image.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()
            for t in range(args.T):
                encoded_img = encoder(image)
                out_fr += model(encoded_img)
            out_fr = out_fr / args.T  
            
            test_samples += label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net=model)                   
    
    test_acc /= test_samples  
    
    return test_acc



def weight_pruning_ratio(model, weight_thres, del_type="weak"):
    zero_vec = torch.zeros([1200])
    zero_rows = 0
    prun_acc_list = []
    model.eval()    
    for (name, module) in model.named_modules():
        if name == "layer1.1":
            weight_size = module.weight.data.shape
            weights = module.weight.data
            weight_sums = torch.abs(torch.sum(weights, axis=0))
            sort_sum, indices = torch.sort(weight_sums, dim=0, decending=True)
            max_weight = torch.max(weight_sums)
            if del_type == "weak":
                for r in range(11):
                    for i in range(weight_size[1]):
                        # remove weak neuron gradually
                        if weight_sums[i] < weight_thres * r / 10:
                            module.weight.data[:, i] = zero_vec  # set zero
                    test_acc = eval(model=model)
                    prun_acc_list.append(test_acc)                            
            elif del_type == "mid":
                for r in range(11):
                    for i in range(weight_size[1]):
                        if weight_sums[i] >= 20 and weight_sums[i] < (50 - 30 * (10 - r) / 10):
                            module.weight.data[:, i] = zero_vec  # set zero
                    test_acc = eval(model=model)
                    prun_acc_list.append(test_acc)  
            else:  
                # delete the hub neurons 
                for r in range(11):
                    for i in range(weight_size[1]):
                        if weight_sums[i] >= 50 and weight_sums[i] < (max_weight - (max_weight - 50) * (10 - r) / 10):
                            module.weight.data[:, i] = zero_vec  # set zero
                    test_acc = eval(model=model)
                    prun_acc_list.append(test_acc)   
            
            # print(module)    
            # w_new =  module.weight.data
            # w_sum2 = torch.abs(torch.sum(w_new, axis=0)) 
            # for ii in range(784):
            #     if w_sum2[ii] == 0:
            #         zero_rows += 1
            # ratio = zero_rows / 784           
        else:
            pass
    
    prun_acc_list = np.array(prun_acc_list)
    print(prun_acc_list)
    np.save('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/plot_prun_acc_list_'+del_type+'.npy', prun_acc_list)  
      
    # TODO: Drop parameters
    
    
    # num_params = sum([param.nelement() for param in model.parameters()])  
    # print("The number of parameters after prunning is : {}".format(num_params))
    
    return zero_rows # zero_rows is the number of prunned edge    


def delete_and_retrain():
    '''
        给定阈值, 删除低于权重阈值的神经元并重新训练(微调)
    '''
    checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max_15epoch_control.pth'
    # load snn
    model = SNN(tau=args.tau).to(args.device)
    ckp = torch.load(checkpoint_path)
    model.load_state_dict(ckp["net"])
    
    # print(model)
    # print(model.named_modules())
    # torchsummary.summary(model, (28, 28))
    # model.eval()
    # for (name, module) in model.named_modules():
    #     print("The curent layer is:{}".format(name))
    #     # if name.find("fc") != -1:
    #         # print("hello")
    #     if name == "layers1.2.surrogate_function":
    #         # print(module.weight.data.shape)
    #         module.register_forward_hook(hook=hook)
    # Pruning 
    # del_type="weak", "mid", "hub"
    _, _ = weight_pruning(model=model, weight_thres=20, del_type="hub")
    # retain
    finetune(model=model)


def delete_and_eval(neuron_type):
    '''
        敲除神经元, 并对敲除低于给定阈值后的神经元的模型进行性能的评估
    '''
    prun_acc_list = []
    checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max_15epoch_control.pth'
    # load snn
    model = SNN(tau=args.tau).to(args.device)
    ckp = torch.load(checkpoint_path)
    model.load_state_dict(ckp["net"])
    encoder = encoding.PoissonEncoder()
    _, _ = weight_pruning(model=model, weight_thres=20, del_type=neuron_type)

    # prun_ratio_list.append(ratio)

def tsne():
    '''
        该函数用于对训练好的模型进行中间层特征分离度的可视化
    '''
    checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max.pth'  # '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max_15epoch_control.pth' 
    # load snn
    model = SNN(tau=args.tau).to(args.device)
    ckp = torch.load(checkpoint_path)
    model.load_state_dict(ckp["net"])
    zero_rows, prun_ratio = weight_pruning(model=model, thr=20, del_type="hub")
    print("{} neurons are deleted.".format(zero_rows))
    print("{} % neurons in the first layer are deleted.".format(prun_ratio * 100))
    encoder = encoding.PoissonEncoder()
    X = []
    # X2 = []
    # X3 = []
    # X4 = []
    Y = []
    # v_sum1 = torch.zeros([1, 1200])
    # use only one sample
    # print(len(test_data_loader))   # train_loader
    model.eval()
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_data_loader):
            # image = image.reshape(image.shape[0], image.shape[1]*image.shape[2]*image.shape[3])       
            if i != 156:    
                out_fr = 0.
                image = image.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                for t in range(args.T):
                    encoded_img = encoder(image)
                    out_fr += model(encoded_img)
                out_fr = out_fr / args.T
                
                test_samples += label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net=model)
                l1_out = model.x1
                # l2_out = model.x2
                # l3_out = model.x3
                # l4_out = model.out
            else:
                break     
    
            # outputs, _ = model(image)
            # for i in range(len(features_out_hook)):
            #     features_out_hook[i] = features_out_hook[i].cpu()
            # for j in range(features_out_hook[0].shape[0]):
            #     v_sum1 += features_out_hook[0][j, :].cpu()
            # v_sum1 = v_sum1 / 20
            # print(features_out_hook[0].shape)
            # X.append(features_out_hook[0].cpu())
            X.append(l1_out.cpu())  # 将第一层抽取出的特征append到列表里
            # X.append(l2_out.cpu())
            # X.append(l3_out.cpu())
            # X.append(l4_out.cpu())
            # X.append(out_fr.cpu())
            # X.append(outputs.detach().cpu())
            # X.append(image.cpu())
            # X.append(features_out_hook)
            Y.append(label.cpu())
            
            
    test_acc /= test_samples    
    print("The accuracy of (modified) network on test set is:{}".format(test_acc))    
    # np.save('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/test-acc-control.npy', test_acc)
    # X = np.stack(X, axis=0)
    X = torch.stack(X, dim=0).squeeze(1).numpy()
    # X = torch.stack(X, dim=0).numpy()
    # X = X.squeeze(2)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    # X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]*X.shape[3])
    # X = X[:10000, :]
    Y = torch.stack(Y, dim=0).numpy()
    # Y = np.stack(Y, axis=0)
    Y = Y.reshape(Y.shape[0]*Y.shape[1])
    # Y = Y[:10000]                
    input_layer_digits = TSNE(perplexity=30).fit_transform(X)
    plot(input_layer_digits, Y, layer_idx=1)  # layer_idx = 1, 2, 3, 4


def plot(x, colors, layer_idx):
    # Choosing color palette
    palette = np.array(sns.color_palette("pastel", 10))
    # pastel, husl, and so on
    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)

    plt.axis('off')
    plt.savefig('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/digits-tsne-del-hub-layer{}.png'.format(layer_idx), dpi=700, bbox_inches="tight")
    return f, ax, txts


def prun_ratio_tunning(if_finetune=False):
    '''
        逐渐调大阈值, 对模型剪枝(one-shot pruning)
    '''
    prun_acc_list = []
    prun_ratio_list = []
    checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max_15epoch_control.pth'
    # load snn
    model = SNN(tau=args.tau).to(args.device)
    ckp = torch.load(checkpoint_path)
    model.load_state_dict(ckp["net"])
    encoder = encoding.PoissonEncoder()
    for thr in range(0, 120, 2):
        # prune the network
        n_zero_rows, ratio = weight_pruning(model, thr)  # prunning based on connectivity for first layer
        print("Ratio = {:.3f}".format(ratio))
        # pruner = L2FilterPruner(model)
        # finetune the network
        if if_finetune:
            finetune(thr, model)    
        model.eval()
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for i, (image, label) in enumerate(test_data_loader):         
                out_fr = 0.
                image = image.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                for t in range(args.T):
                    encoded_img = encoder(image)
                    out_fr += model(encoded_img)
                out_fr = out_fr / args.T  
                
                test_samples += label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net=model)
        
        test_acc /= test_samples
        prun_acc_list.append(test_acc)
        prun_ratio_list.append(ratio)
    
    prun_acc_list = np.array(prun_acc_list)
    prun_ratio_list = np.array(prun_ratio_list)
    plt.plot(prun_ratio_list, prun_acc_list)
    plt.savefig('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/prunning_acc_curve_wo_finetune.png', dpi=700)
    np.save('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/prun_acc_list_wo_finetune.npy', prun_acc_list)
    np.save('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/prun_ratio_list_wo_finetune.npy', prun_acc_list)

def weight_distribution(model):
    for (name, module) in model.named_modules():
        if name == "layer1.1":
            weight_size = module.weight.data.shape
            weights = module.weight.data
            weight_sums = torch.abs(torch.sum(weights, axis=0))
           
            plt.hist(weight_sums.cpu().numpy(), bins=40, facecolor='lightgrey', edgecolor='w', density=True)
            plt.xlim((0, 120))
            plt.ylim((0.00, 0.08))
            plt.savefig('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/weight_hist_plot.png', dpi=700, bbox_inches='tight')
                # for i in range(weight_size[1]):
                #     if weight_sums[i] == 0:


def finetune(thr=20, model=None, fintune_epoch=15):
    '''
        对剪枝后的model进行微调
        thr : 剪枝的阈值
    '''
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
    start_epoch = 0
    max_test_acc = -1
    
    # freeze the weight matrix in layer 1
    # for (name, params) in model.named_parameters():
    #     if "layer1" not in name:
    #         params.required_grad = False
                           
    # parameters = [p for p in model.parameters() if p.requires_grad]
    pretrain_parameters = [p for n, p in model.named_parameters() if p.requires_grad and "layer4" not in n]
    fc_parameters = [p for n, p in model.named_parameters() if "layer4" in n]       
    if args.opt == 'sgd':
        # finetune_opt = torch.optim.SGD(parameters, lr=args.lr*0.1, momentum=args.momentum)
        finetune_opt = torch.optim.SGD(
            {"params": pretrain_parameters, "lr":1e-4},
            {"params":fc_parameters, "lr":1e-3},
            momentum=args.momentum
        )
    elif args.opt == 'adam':
        # finetune_opt = torch.optim.Adam(parameters, lr=args.lr*0.1)
        finetune_opt = torch.optim.Adam(
            [{'params': pretrain_parameters, 'lr':1e-4},
            {'params': fc_parameters, 'lr':args.lr}]    # args.lr
        )
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['net'])
        finetune_opt.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr*0.1}')

    if args.amp:
        out_dir += '_hub_amp_retrain'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, fintune_epoch):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in tqdm(train_data_loader):
            finetune_opt.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += model(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(finetune_opt)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += model(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                finetune_opt.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(model)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += model(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(model)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': finetune_opt.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'finetune_del_hub_max_thr_' + str(thr) + '.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'finetune_del_hub_latest_thr_' + str(thr) + '.pth'))

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


def ranked_input_test(if_rank=True):
    # if_rank 参数用于控制函数是否对输入的图像序列进行排序，若为True则排序，若为False则不排序
    # 对输入进行降序排序，输入到对应权重大小的神经元中
    checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max_15epoch_control.pth'
    # load snn
    model = SNN(tau=args.tau).to(args.device)
    ckp = torch.load(checkpoint_path)
    model.load_state_dict(ckp["net"])
    encoder = encoding.PoissonEncoder()
    model.eval()
    test_acc = 0.
    test_samples = 0.
    with torch.no_grad():
        for i, (image, label) in enumerate(test_data_loader):         
            out_fr = 0.
            image = image.to(args.device)
            if if_rank:
                b, c, h, w = image.shape
                image = image.reshape(b, c, h*w)
                sorted_img = rank_input_fun(image, model, False)
                sorted_img = sorted_img.reshape(b, c, h, w)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()
            for t in range(args.T):
                if if_rank:
                    encoded_img = encoder(sorted_img)
                encoded_img = encoder(image)
                # b, c, h, w = encoded_img.shape
                # # 对图像进行向量化
                # encoded_img = encoded_img.reshape(b, c, h*w)
                # # 对每个batch中的图像脉冲序列按频率由高到低进行排序
                # sorted_encoded_img = rank_input_fun(encoded_img, model)
                # # 再将图像重新reshpae回原来的形状输入
                # sorted_encoded_img = sorted_encoded_img.reshape(b, c, h, w)
                out_fr += model(encoded_img)   # org:encoded_img
            out_fr = out_fr / args.T  
            
            test_samples += label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net=model)
    
    test_acc /= test_samples

    print(format(test_acc, '.5f'))
    

if __name__ == '__main__':
    # main()
    tsne()
    # weight_distribution(model=model)
    # delete_and_retrain()
    # prun_ratio_tunning(if_finetune=False)
    # prun_acc_array = np.load('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/prun_acc_list2_wo_finetune.npy')
    # print(prun_acc_array)
    
    # ranked_input_test(if_rank=True)
    
    # del_type_list =["weak", "mid", "hub"]
    
    # checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max.pth'
    # model = SNN(tau=args.tau).cuda()
    # ckp = torch.load(checkpoint_path)
    # model.load_state_dict(ckp["net"])
    # weight_distribution(model=model) 
    # # finetune(model)   
    # # control_acc = eval(model=model)
    # # np.save('/home/gaoheng/CODE_ALL/SNN/SNN-SBP/SBP_SpikingJelly/results/plot_control_acc.npy', control_acc)
    
    # for dt in del_type_list:   
    #     checkpoint_path = '/home/gaoheng/CODE_ALL/logs/T100_b64_adam_lr0.001_amp/checkpoint_max_15epoch_control.pth'
    #     # load snn
    #     model = SNN(tau=args.tau).to(args.device)
    #     ckp = torch.load(checkpoint_path)
    #     model.load_state_dict(ckp["net"])    
    #     # finetune(model=model)
    #     # test()
    #     print("Deleting "+dt+" neurons")
    #     _ = weight_pruning_ratio(model=model, weight_thres=20, del_type=dt)