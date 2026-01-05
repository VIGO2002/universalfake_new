import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from models import get_model
from transformers import get_cosine_schedule_with_warmup

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt.arch, opt)
        self.lr = opt.lr
        
        # 初始化分类头参数
        if hasattr(self.model, 'fc'):
            for m in self.model.fc.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0.0, opt.init_gain)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                elif isinstance(m, torch.nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

        # 参数冻结策略
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if 'fc.' in name: 
                    params.append(p)
                    p.requires_grad = True
                elif any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                    params.append(p)
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            print(f">>> Backbone fixed. Training {len(params)} tensors (Head + SVD Residuals).")
        else:
            print("Your backbone is not fixed. Training all parameters.")
            params = self.model.parameters()

        # 优化器
        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        # Loss 函数配置
        self.loss_fn = nn.CrossEntropyLoss()
        self.margin = 5.0        
        self.lambda_ebm = 0.5    
        self.lambda_smooth = 0.1 

        self.model.to(opt.gpu_ids[0])
        
        self.scheduler = None
        if hasattr(opt, 'warmup_steps') and opt.warmup_steps > 0:
            print(f">>> Using Cosine Scheduler with {opt.warmup_steps} warmup steps.")
            try:
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=opt.warmup_steps, num_training_steps=opt.niter * 1000 
                )
            except:
                pass 

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).long()

    def forward(self):
        # 训练过程直接调用，模型内部根据 self.training 返回 5 元组
        self.output = self.model(self.input)
        
        # 解析返回值
        if isinstance(self.output, tuple) and len(self.output) == 5:
            self.logits = self.output[0]
            self.e_real = self.output[1]
            self.e_fake = self.output[2]
            self.e_real_noisy = self.output[3]
            self.e_fake_noisy = self.output[4]
            self.output = self.logits
        elif isinstance(self.output, tuple) and len(self.output) == 3:
            self.logits = self.output[0]
            self.e_real = self.output[1]
            self.e_fake = self.output[2]
            self.output = self.logits
        else:
            self.logits = self.output
            self.e_real = None

    def test(self):
        # 专门用于 test_diffusion.py 的测试方法
        # 显式要求返回能量
        with torch.no_grad():
            self.output = self.model(self.input, return_energy=True)
            
            if isinstance(self.output, tuple) and len(self.output) >= 3:
                self.logits = self.output[0]
                self.e_real = self.output[1]
                self.e_fake = self.output[2]
                self.output = self.logits
            else:
                self.logits = self.output
    
def get_loss(self):
        loss_cls = self.loss_fn(self.logits, self.label)
        
        # 1. 基础分类损失
        total_loss = loss_cls

        # 2. 能量损失 (如果有)
        if self.e_real is not None:
            fake_mask = (self.label == 1)
            real_mask = (self.label == 0)
            
            loss_energy = 0.0
            # Symmetric Margin Loss
            if real_mask.sum() > 0:
                loss_energy += F.relu(self.e_real[real_mask] - self.e_fake[real_mask] + self.margin).mean()
            if fake_mask.sum() > 0:
                loss_energy += F.relu(self.e_fake[fake_mask] - self.e_real[fake_mask] + self.margin).mean()

            # Smoothness Loss
            loss_smooth = 0.0
            if hasattr(self, 'e_real_noisy') and self.e_real_noisy is not None:
                loss_smooth = F.mse_loss(self.e_real, self.e_real_noisy) + \
                              F.mse_loss(self.e_fake, self.e_fake_noisy)
            
            # 累加能量损失
            total_loss += self.lambda_ebm * loss_energy + self.lambda_smooth * loss_smooth

        # 3. 【新增】SVD 正则化损失 (Crucial for SVD Fine-tuning)
        # 只在训练时计算，且只针对包含 SVD 参数的模型
        if self.model.training:
            loss_svd = 0.0
            num_reg = 0
            # 遍历所有子模块寻找 SVD 参数
            for module in self.model.modules():
                if hasattr(module, 'S_residual') and module.S_residual is not None:
                    U = module.U_residual
                    V = module.V_residual
                    S = module.S_residual
                    
                    # (A) 正交性约束: U^T U ≈ I
                    # 我们希望 U 和 V 保持正交，这样 S 才能代表奇异值
                    if U is not None and V is not None:
                        # 简单的正交惩罚
                        u_orth = torch.norm(U.T @ U - torch.eye(U.shape[1], device=U.device), p='fro')
                        v_orth = torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device), p='fro')
                        loss_svd += (u_orth + v_orth)
                        num_reg += 1
            
            # 如果找到了 SVD 层，加上这个损失
            if num_reg > 0:
                # 权重通常设为 0.1 或 0.5，这里给一个合理的默认值
                # 这有助于防止 SVD 参数漂移过远
                total_loss += 0.1 * (loss_svd / num_reg)

        return total_loss

    def optimize_parameters(self):
        self.model.train() 
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
