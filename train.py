import argparse
import math
import os

import torch
import wandb
from accelerate import Accelerator
from pylabel.importer import ImportVOC
from torch import optim
from torchsummary import summary
from tqdm import tqdm

from dataset import DACDataset
from loss import DetectorLoss
from model import Detector
from utils.datasets import *
from utils.evaluation_coco import CocoDetectionEvaluator
from utils.tool import *

# 指定后端设备CUDA&CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastestDet:
    def __init__(self):
        # 指定训练配置文件
        parser = argparse.ArgumentParser()
        parser.add_argument("--yaml", type=str, default="", help=".yaml config")
        parser.add_argument("--weight", type=str, default=None, help=".weight config")

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"

        # 解析yaml配置文件
        self.cfg = LoadYaml(opt.yaml)
        print(self.cfg.category_num)
        # exit()
        # 初始化模型结构
        if opt.weight is not None:
            print("load weight from:%s" % opt.weight)
            self.model = Detector(self.cfg.category_num, True).to(device)
            self.model.load_state_dict(torch.load(opt.weight))
        else:
            self.model = Detector(self.cfg.category_num, False).to(device)

        # # 打印网络各层的张量维度
        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))
        self.accelerator = Accelerator()
        # 构建优化器
        print("use SGD optimizer")
        self.optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=self.cfg.learn_rate,
            momentum=0.949,
            weight_decay=0.0005,
        )
        # 学习率衰减策略
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.cfg.milestones, gamma=0.1
        )

        # 定义损失函数
        self.loss_function = DetectorLoss(device)

        # 数据集加载
        # val_dataset = TensorDataset(
        #     self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, False
        # )
        # train_dataset = TensorDataset(
        #     self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, True
        # )
        pylabel_ds = ImportVOC(
            "/home/cc/Dev/IdeaProjects/UConn/Ding/dac2023-gpu/data/train/Annotations",
            "/home/cc/Dev/IdeaProjects/UConn/Ding/dac2023-gpu/data/train/JPEGImages",
        )
        pylabel_ds.splitter.GroupShuffleSplit(
            train_pct=0.7, val_pct=0.29, test_pct=0.01, random_state=42
        )
        val_dataset = DACDataset(pylabel_ds, split="val")
        train_dataset = DACDataset(pylabel_ds, split="train")

        # 定义验证函数
        self.evaluation = CocoDetectionEvaluator(
            train_dataset.category_to_int.keys(), device
        )

        # 验证集
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=False,
            persistent_workers=True,
        )
        # 训练集
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
        )
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )
        self.cfg["names"] = train_dataset.category_to_int.keys()

    def train(self):
        # 迭代训练
        batch_num = 0
        print("Starting training for %g epochs..." % self.cfg.end_epoch)
        run = wandb.init(
            project="fastestdet", config=self.cfg.to_dict(), save_code=True
        )
        wandb.watch(self.model, log_freq=100)
        table = wandb.Table(columns=["ID", "Image"])

        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for imgs, targets in pbar:
                # 数据预处理
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
                # 模型推理
                preds = self.model(imgs)

                # loss计算
                with self.accelerator.autocast():
                    iou, obj, cls, total = self.loss_function(preds, targets)
                run.log(
                    {
                        "train/epoch": epoch,
                        "train/train_loss": total,
                        "train/iou": iou,
                        "train/obj": cls,
                    }
                )
                # 反向传播求解梯度
                self.accelerator.backward(total)
                # 更新模型参数
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 学习率预热
                for g in self.optimizer.param_groups:
                    warmup_num = 5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = math.pow(batch_num / warmup_num, 4)
                        g["lr"] = self.cfg.learn_rate * scale
                    lr = g["lr"]
                # run.log({"train/lr": lr})
                # 打印相关训练信息
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                    epoch,
                    lr,
                    iou,
                    obj,
                    cls,
                    total,
                )
                pbar.set_description(info)
                batch_num += 1

            # 模型验证及保存
            if epoch % 10 == 0 and epoch > 0:
                # 模型评估
                self.model.eval()
                print("computer mAP...")
                mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)
                # run.log({"test/mAP05": mAP05})
                torch.save(
                    self.model.state_dict(),
                    "checkpoint/weight_AP05:%f_%d-epoch.pth" % (mAP05, epoch),
                )

            # 学习率调整
            self.scheduler.step()

        # table = wandb.Table(columns=['id', 'Image'])
        for imgs, targets in self.val_dataloader:
            with torch.no_grad():
                preds = self.model(img)
                output = handle_preds(preds, device, 0.0001)
            print(output)
            # wandb.Image(img, boxes={"prediction": {
            #     "box_data": [{
            #         "position": {
            #             "minX": box["minX"],
            #             "minY": box["minY"],
            #             "maxX": box["maxX"],
            #             "maxY": box["maxY"]
            #         },
            #         "class_id": box["class_id"],
            #         "box_caption": box["caption"],
            #         "domain": "pixel"
            #     }
            #         for box in boxes
            #     ],
            #     "class_labels": class_labels
            # }})
        run.finish()


if __name__ == "__main__":
    model = FastestDet()
    model.train()
