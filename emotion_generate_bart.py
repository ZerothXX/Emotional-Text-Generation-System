"""
情感文本生成 - 主训练脚本 (BART)
调用 emotion_data_process.py 和 emotion_model.py
"""
import os
import random
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from emotion_data_process import DataProcessor
from emotion_model import (
    EMOTIONS,
    load_tokenizer,
    load_model,
    EmotionBartDataset
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def data_generator(model):
    processor = DataProcessor(
        corpus_path='data/corpus.txt',
        emotion_lexicon_dir='data/Emotional_Lexicon'
    )

    # 检查是否已有处理好的数据
    processed_data_path = 'saved/processed_data.json'
    if os.path.exists(processed_data_path):
        print(f"发现已处理的数据: {processed_data_path}")
        choice = input("是否重新处理数据？(y/n): ").strip().lower()
        if choice == 'y':
            data = processor.load_and_process_corpus(max_sentences=5000000)
            processor.save_processed_data(data, processed_data_path)
        else:
            data = processor.load_processed_data(processed_data_path)
    else:
        data = processor.load_and_process_corpus(max_sentences=5000000)
        processor.save_processed_data(data, processed_data_path)

    # 采样数据
    if len(data) > model['max_samples']:
        data = random.sample(data, model['max_samples'])
    print(f"使用 {len(data)} 条数据进行训练")

    return data


class Trainer:
    """训练器"""

    def __init__(self, model, tokenizer, train_loader, val_loader, device, save_dir='molde'):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # 优化器
        self.optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        # 训练初期直接用大学习率会导致梯度不稳定，破坏预训练权重。
        # 学习率调度器
        total_steps = len(train_loader) * 5
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        # 记录
        self.train_losses = []
        self.val_losses = []
        self.step_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, accum_steps=4):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc="训练")

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            actual_loss = loss.item() * accum_steps
            total_loss += actual_loss
            self.step_losses.append(actual_loss)
            pbar.set_postfix({'loss': f'{actual_loss:.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                total_loss += outputs.loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs):
        """训练模型"""
        print(f"\n开始训练，共 {num_epochs} 个 epoch")
        print(f"训练集: {len(self.train_loader.dataset)}, 验证集: {len(self.val_loader.dataset)}")
        print(f"设备: {self.device}\n")

        patience = 0

        for epoch in range(num_epochs):
            print(f"{'=' * 20} Epoch {epoch + 1}/{num_epochs} {'=' * 20}")

            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience = 0
                self.save_model('emotion_bart_best.pt')
                print("✓ 保存最佳模型")
            else:
                patience += 1
                print(f"验证损失未改善 ({patience}/2)")
                if patience >= 2:
                    print("\nEarly Stopping")
                    break

            print()

        # 保存最终模型和损失曲线
        self.save_model('emotion_bart_final.pt')
        self.plot_losses()

        print("训练完成！")

    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)

    def plot_losses(self):
        """绘制损失曲线"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Epoch 损失
        axes[0].plot(self.train_losses, 'b-o', label='训练')
        axes[0].plot(self.val_losses, 'r-s', label='验证')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Epoch 损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Step 损失（平滑）
        if len(self.step_losses) > 30:
            w = 20
            smoothed = [sum(self.step_losses[max(0, i - w):i + 1]) / (min(i + 1, w))
                        for i in range(len(self.step_losses))]
            axes[1].plot(smoothed, 'g-', alpha=0.8)
        else:
            axes[1].plot(self.step_losses, 'g-')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Step 损失曲线')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'loss_bart.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"损失曲线已保存: {save_path}")


def main():
    CONFIG = {
        'seed': 42,  # 随机种子
        'max_samples': 300000,  # 随机采样对话文本数
        'batch_size': 64,
        'max_input_len': 64,  # 最大输入
        'max_output_len': 64,  # 最大输出
        'num_epochs': 5,  # 训练轮次
        'save_dir': 'molde'  # 模型保存文件夹名
    }

    torch.manual_seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # ==================== 步骤1: 数据处理 ====================
    print("\n" + "=" * 60)
    print("步骤 1: 数据处理")

    data = data_generator(CONFIG)
    # ==================== 步骤2: 加载模型 ====================
    print("\n" + "=" * 60)
    print("步骤 2: 加载模型")

    tokenizer = load_tokenizer()
    model = load_model(tokenizer)
    model = model.to(device)
    # ==================== 步骤3: 创建数据集 ====================
    print("\n" + "=" * 60)
    print("步骤 3: 创建数据集")

    dataset = EmotionBartDataset(
        data, tokenizer,
        max_input_len=CONFIG['max_input_len'],
        max_output_len=CONFIG['max_output_len']
    )

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    # ==================== 步骤4: 训练 ====================
    print("\n" + "=" * 60)
    print("步骤 4: 训练模型")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=CONFIG['save_dir']
    )

    trainer.train(num_epochs=CONFIG['num_epochs'])
    print(f"训练完成！模型已保存到: {CONFIG['save_dir']}/")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
