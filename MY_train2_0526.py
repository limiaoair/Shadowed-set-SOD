import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.lib2.model import SPNet
from Code.lib2.my2_model import sod0502_dul_res_threeloss, sod0814_end_light, sod0814_end_light_ablation_same_backbone_nopre, sod0814_end_light_ablation_no_twoCCM
from Code.utils2.my2_data import get_loader, test_dataset
from Code.utils2.my2_lib import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',               type=int,      default=31,                                      help='epoch number')
parser.add_argument('--lr',                  type=float,    default=1e-4,                                     help='learning rate')
parser.add_argument('--batchsize',           type=int,      default=42,                                        help='training batch size')
parser.add_argument('--trainsize',           type=int,      default=352,                                      help='training dataset size')
parser.add_argument('--clip',                type=float,    default=0.5,                                      help='gradient clipping margin')
parser.add_argument('--lw',                  type=float,    default=0.001,                                    help='weight')
parser.add_argument('--decay_rate',          type=float,    default=0.1,                                      help='decay rate of learning rate')
parser.add_argument('--decay_epoch',         type=int,      default=10,                                       help='every n epochs decay learning rate')
parser.add_argument('--gpu_id',              type=str,      default='0',                                      help='train use gpu')
# parser.add_argument('--load',                type=str,      default=None,                                     help='train from checkpoints')
parser.add_argument('--load',                type=str,      default='./Checkpoint/MY/Ablation/SS_ex-light_norm_CCM_two/SPNet_epoch_best.pth',                                     help='train from checkpoints')
# parser.add_argument('--load',                type=str,      default='./Checkpoint/MY/0625/SS_light_norm0820/SPNet_0276_best_ex_light_norm.pth',                                     help='train from checkpoints')
parser.add_argument('--rgb_label_root',      type=str,      default='./data/DUTS/DUTS-TR/DUTS-TR-Image/',     help='the training rgb images root')
# parser.add_argument('--depth_label_root',    type=str, default='/home/lm/data_for_lm/train/DUTS-TR/EDGE/',             help='the training edge images root')

parser.add_argument('--depth_label_root',    type=str,      default='./data/DUTS/DUTS-TR/DUTS-TR-I/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str,      default='./data/DUTS/DUTS-TR/DUTS-TR-Mask/',      help='the training gt images root')
parser.add_argument('--val_rgb_root',        type=str,      default='/home/lm/data_for_lm/TEST/ECSSD/RGB/',   help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str,      default='/home/lm/data_for_lm/TEST/ECSSD/I/',     help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str,      default='/home/lm/data_for_lm/TEST/ECSSD/GT/',    help='the test gt images root')
parser.add_argument('--save_path1',          type=str,      default='./Checkpoint/MY/Ablation/SS_ex-light_norm_CCM_two/',                  help='the path to save models and logs')
parser.add_argument('--save_path_sal',       type=str,      default='./test_maps/MY/Ablation/SS_ex-light_norm_ccm_two/',                   help='the path to save models and logs')

opt = parser.parse_args()


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def confident_loss(pred, gt, beta=2):
    y = torch.sigmoid(pred)
    weight = beta * y * (1 - y)
    weight = weight.detach()
    loss = (F.binary_cross_entropy(pred, gt, reduction='none') * weight).mean()
    loss2 = 0.5 * beta * (y * (1 - y)).mean()
    return loss + loss2

def MY_loss(pre1, pre2, pre3, mask):
    pool  = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    # pre2  = pool(pre)
    # pre4  = pool(pre2)
    gt2   = pool(mask)
    gt4   = pool(gt2)
    out1  = structure_loss(pre1, mask)
    out2  = structure_loss(pre2, gt2)
    out4  = structure_loss(pre3, gt4)
    out   = out1*0.7 + out2*0.15 + out4*0.15
    # out   = out1*0.7 + out2*0.15 + out4*0.15
    return out

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images   = images.cuda()
            gts      = gts.cuda()
            depths   = depths.cuda()
            pre_res0, pre_res1, pre_res2  = model(images,depths)
            # gt1      = pool_2(gts)
            loss1    = structure_loss(pre_res0, gts)
            loss2    = structure_loss(pre_res1, gts)
            loss3    = structure_loss(pre_res2, gts) 
            # loss1    = MY_loss(pre_res0, pre_res1, pre_res2, gts)
            
            loss_seg = loss1  + loss2 + loss3
            loss = loss_seg 
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 100 == 0 or i == total_step or i==1:
                # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}'.
                #     format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data))
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss1.data, loss2.data, loss3.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss1.data))
                
        loss_all/=epoch_step

        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        writer.add_images("gt1", gts, 1, dataformats='NCHW')
        writer.add_images("pre1", pre_res0, 1, dataformats='NCHW')

        if (epoch) % 40 == 0:
            torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise

def val(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt,depth, name,img_for_post = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.cuda()
            pre_res0, pre_res1, pre_res2 = model(image,depth)
            res     = pre_res0
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            
        mae = mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'SPNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
                
                for i in range(test_loader.size):
                    image, gt,depth, name,img_for_post = test_loader.load_data()
                    gt      = np.asarray(gt, np.float32)
                    gt     /= (gt.max() + 1e-8)
                    image   = image.cuda()
                    depth   = depth.cuda()
                    pre_res0, pre_res1, pre_res2 = model(image,depth)
                    res     = pre_res0
                    res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                    res     = res.sigmoid().data.cpu().numpy().squeeze()
                    res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    
                    save_map_path = save_path+ '/' + str(epoch) + '/'
                    if not os.path.exists(save_map_path):
                        os.makedirs(save_map_path)
                    cv2.imwrite(save_map_path + name,res*255)

                print("save sal maps using best pth:{} successfully...".format(epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))


if __name__ == '__main__':
    """
    这个代码是不解冻训练使用的，最终版  0510  
    """
    print("Start train...")
    if opt.gpu_id=='0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    cudnn.benchmark = True

    pretrained      = True

    Init_Epoch          = 0
    Freeze_Epoch        = 201
    Freeze_batch_size   = opt.batchsize
    Freeze_Train        = True
    UnFreeze_Epoch      = opt.epoch
    Unfreeze_batch_size = opt.batchsize

    train_image_root = opt.rgb_label_root
    train_gt_root    = opt.gt_label_root
    train_depth_root = opt.depth_label_root
    val_image_root   = opt.val_rgb_root
    val_gt_root      = opt.val_gt_root
    val_depth_root   = opt.val_depth_root
    save_path        = opt.save_path1

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # model = sod0502_dul_ablation_C_no_pool()
    # model = sod0814_end_light_ablation_same_backbone_nopre()
    model = sod0814_end_light()
    # model = sod0814_end_light_ablation_no_twoCCM()
    # model = sod0502_dul_res_threeloss()
    if(opt.load is not None):
        model.load_state_dict(torch.load(opt.load))
        print('load model from ',opt.load)
    
    model.cuda()
    params    = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    if Freeze_Train:
        for param1 in model.layer_rgb.parameters():
            param1.requires_grad = False
            # print(param1.requires_grad)
    # if Freeze_Train:
    #     for param2 in model.layer_i.parameters():
    #         param2.requires_grad = False

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))

    print('load data...')
    train_loader = get_loader(train_image_root, train_gt_root,train_depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    test_loader  = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)
    total_step   = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("BBSNet_unif-Train")
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))
    CE   = torch.nn.BCEWithLogitsLoss()

    step = 0
    writer     = SummaryWriter(save_path+'summary') 
    best_mae   = 1
    best_epoch = 0

    print(len(train_loader))

    for epoch in range(Init_Epoch, UnFreeze_Epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        
        train(train_loader, model, optimizer, epoch, save_path)

        val(test_loader,model,epoch,save_path)

