'''
@Project ：kaggle
@File    ：RetinexNet_model.py
@IDE     ：PyCharm 
@Author  ：QiuWeiXin
@Date    ：2024/4/12 10:52 
'''
import os
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,padding=1, padding_mode='replicate'),#复制填充
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0] # 计算输入图像的通道维度的最大值
        input_img = torch.cat((input_max, input_im), dim=1) # 将最大值通道拼接到输入图像中，扩展通道维度

        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        # 获取反射分量R和光照分量L，通过 sigmoid 函数确保输出在 (0, 1) 范围内
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,padding=1, padding_mode='replicate')

        self.net2_deconv1_1 = nn.Conv2d(channel * 2, channel, kernel_size,padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel * 2, channel, kernel_size,padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel * 2, channel, kernel_size,padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel * 3, channel, kernel_size=1,padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1) # 将输入的低光照图像和反射分量图像拼接在一起

        out0 = self.net2_conv0_1(input_img) # 通过网络进行前向传播，得到一系列特征图
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))
        # 上采样特征图以匹配下一层的尺寸
        out3_up = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))
        # 上采样特征图以匹配输入图像的尺寸
        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        # 将所有特征图连接在一起
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        # 通过融合层获得融合特征
        feats_fus = self.net2_fusion(feats_all)
        # 通过输出层获得最终的结果
        output = self.net2_output(feats_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self, device, size):
        super(RetinexNet, self).__init__()
        self.device = device
        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        self.size = size
    def forward(self, input_low, input_high):
        # 将输入数据转换为 PyTorch 变量，并移到 GPU 上进行计算
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).to(self.device)
        input_high = Variable(torch.FloatTensor(torch.from_numpy(input_high))).to(self.device)
        # 使用 DecomNet 进行图像分解，得到反射 R 和亮度 L
        R_low, I_low = self.DecomNet(input_low)  # 低光照条件下的反射 R 和亮度 L
        R_high, I_high = self.DecomNet(input_high)  # 高光照条件下的反射 R 和亮度 L

        # 使用 RelightNet 进行图像重照
        I_delta = self.RelightNet(I_low, R_low)  # 获取重照图像 I_delta

        # 其他变量
        # 将亮度图像 I 转换为三通道，以便与反射 R 相乘
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        # 计算损失函数
        self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low)           # 低光照条件下的重建损失
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)       # 高光照条件下的重建损失
        self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low)    # 亮度互补损失（低光照）
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)  # 亮度互补损失（高光照）
        self.equal_R_loss = F.l1_loss(R_low, R_high.detach())                 # 反射一致性损失
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)          # 重照损失

        self.Ismooth_loss_low = self.smooth(I_low, R_low)  # 低光照条件下的光滑损失
        self.Ismooth_loss_high = self.smooth(I_high, R_high)  # 高光照条件下的光滑损失
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)  # 重照图像的光滑损失

        # 计算总损失函数
        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
        self.loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta

        # 将网络输出的结果保存为 CPU 上的张量，以便后续处理和可视化
        self.output_R_low = R_low.detach().cpu()  # 保存低光照条件下的反射 R
        self.output_I_low = I_low_3.detach().cpu()  # 保存低光照条件下的亮度 L
        self.output_I_delta = I_delta_3.detach().cpu()  # 保存重照图像 I_delta
        self.output_S = R_low.detach().cpu() * I_delta_3.detach().cpu()  # 保存反射与重照图像的乘积 S


    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(self.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/' # 构建加载模型的目录
        if os.path.exists(load_dir): # 判断目录是否存在
            load_ckpts = os.listdir(load_dir)# 列出目录下的所有文件
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)# 按文件名长度排序
            if len(load_ckpts) > 0:# 如果存在检查点文件
                load_ckpt = load_ckpts[-1] # 选择最新的检查点文件
                global_step = int(load_ckpt[:-4])# 获取全局步数
                ckpt_dict = torch.load(load_dir + load_ckpt)# 加载模型参数
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0

    def predict(self, test_low_data_names, ckpt_dir):
        self.train_phase = 'Decom'# 加载预训练模型
        load_model_status, _ = self.load(ckpt_dir)
        self.train_phase = 'Relight'
        load_model_status, _ = self.load(ckpt_dir)# 加载预训练模型
        # save_R_L = False # 是否保存反射和阴影图像
        image = []
        # 对测试图像进行预测
        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]# 获取测试图像的路径
            test_low_img = self.resize_image(test_img_path, self.size)#缩放图像
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))# 转置图像维度(3, 680, 720)
            input_low_test = np.expand_dims(test_low_img, axis=0) # 扩展维度(1, 3, 680, 720)以符合模型输入要求

            if self.is_dark(input_low_test):
                test_img_name = test_img_path.split('/')[-1]  # 获取测试图像的名称
                print('Processing ', test_img_name)
                self.forward(input_low_test, input_low_test)  # 使用模型进行前向传播
                # result_1 = self.output_R_low  # 获取输出的反射图像
                # result_2 = self.output_I_low  # 获取输出的亮度图像
                # result_3 = self.output_I_delta  # 获取输出的亮度变化图像
                result_4 = self.output_S  # 获取输出的光照图像
                # input = np.squeeze(input_low_test)  # 去除输入图像的单维度
                # result_1 = np.squeeze(result_1)  # 去除反射图像的单维度
                # result_2 = np.squeeze(result_2)  # 去除亮度图像的单维度
                # result_3 = np.squeeze(result_3)  # 去除亮度变化图像的单维度
                result_4 = np.squeeze(result_4)  # 去除光照图像的单维度

                image.append(result_4)
            else:
                input_low_test = np.squeeze(input_low_test)
                image.append(input_low_test)

        return image

    def resize_image(self, input_image_path, size):
        original_image = Image.open(input_image_path)
        width, height = original_image.size
        aspect_ratio = width / height
        new_width = size
        new_height = int(size / aspect_ratio)
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image

    def is_dark(self, img, threshold=10):
        img = torch.tensor(img)
        img = torch.squeeze(img, dim=0).permute(1, 2, 0)
        gray = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2GRAY)#转灰度图 输入应该是 (通道数, 高度, 宽度)
        averge = cv2.mean(gray)[0]
        return averge < threshold #1是低照度，0不是