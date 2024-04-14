'''
@Project ：kaggle
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：QiuWeiXin
@Date    ：2024/4/12 10:35 
'''
import torch
from glob import glob
from RetinexNet_model import RetinexNet

class RetinexModelEvaluator:
    def __init__(self, data_dir='./data/test/low/', device=torch.device('cuda'), size=512,ckpt_dir='./ckpts/'):
        """
        初始化 RetinexModelEvaluator 类。
        参数：
        - data_dir：包含测试数据的目录。默认为 './data/test/low/'。
        - ckpt_dir：检查点目录，用于存储模型检查点。默认为 './ckpts/'。
        """
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.size = size
    def evaluate_model(self):
        """
        使用 Retinex 模型评估图像。
        """
        # 获取测试数据文件名列表
        test_low_data_names = glob(self.data_dir + '/' + '*.*')
        test_low_data_names.sort()
        print('Number of evaluation images: %d' % len(test_low_data_names))

        # 创建模型并将其移动到 GPU 上
        model = RetinexNet(self.device, self.size).to(self.device)
        model.eval()
        image = model.predict(test_low_data_names, ckpt_dir=self.ckpt_dir)
        return image

if __name__ == '__main__':
    '''
    判断目录下的图片是否为低光照的图片，
    低光照的图像增强，
    返回<增强后的图像>和<不需要增强的图像>
    返回格式 list(通道数，高度, 宽度) 如{Tensor: (3, 682, 512)}    
    
    参数：
    - path：包含测试数据的目录。默认为 './data/test/low/'。
    - size：为图像高度缩放后的像素大小， 宽会根据等比例变换'。
    '''

    path = 'F:\qwx\学习计算机视觉\kaggle\sci低光照增强\sci\data'
    evaluator = RetinexModelEvaluator(path, device=torch.device('cuda'), size=512)
    image = evaluator.evaluate_model()

    print(image)



