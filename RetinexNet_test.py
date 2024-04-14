'''
@Project ：kaggle
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：QiuWeiXin
@Date    ：2024/4/12 10:35 
'''
import os
from glob import glob
from RetinexNet_model import RetinexNet

class RetinexModelEvaluator:
    def __init__(self, data_dir='./data/test/low/', ckpt_dir='./ckpts/', res_dir='./results/test/low/'):
        """
        初始化 RetinexModelEvaluator 类。

        参数：
        - data_dir：包含测试数据的目录。默认为 './data/test/low/'。
        - ckpt_dir：检查点目录，用于存储模型检查点。默认为 './ckpts/'。
        - res_dir：结果保存目录，用于存储评估结果。默认为 './results/test/low/'。
        """

        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.res_dir = res_dir

    def evaluate_model(self):
        """
        使用 Retinex 模型评估图像。
        """
        # 获取测试数据文件名列表
        test_low_data_names = glob(self.data_dir + '/' + '*.*')
        test_low_data_names.sort()
        print('Number of evaluation images: %d' % len(test_low_data_names))

        # 创建保存结果的目录
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        # 创建模型并将其移动到 GPU 上
        model = RetinexNet().cuda()
        model.eval()
        # 执行预测
        image = model.predict(test_low_data_names,
                      res_dir=self.res_dir,
                      ckpt_dir=self.ckpt_dir)
        return image

if __name__ == '__main__':
    # 使用不同的参数值创建 RetinexModelEvaluator 的实例
    evaluator = RetinexModelEvaluator(data_dir='F:\qwx\学习计算机视觉\kaggle\sci低光照增强\sci\data')

    # 调用 evaluate_model 方法执行评估
    image = evaluator.evaluate_model()

    print(image)



