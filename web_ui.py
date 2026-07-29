"""
情感文本生成系统 - Web UI
基于 Flask 的本地网页界面，包含训练和测试两个模块
"""
import os
import sys
import json
import random
import threading
import torch
from flask import Flask, render_template, request, jsonify
from torch.utils.data import DataLoader, random_split
import tkinter as tk
from tkinter import filedialog

# 导入自定义模块
from emotion_data_process import DataProcessor
from emotion_model import (
    EMOTIONS,
    load_tokenizer,
    load_model,
    EmotionBartDataset,
    generate_text
)

app = Flask(__name__)

# 文件选择相关
selected_paths = {
    'corpus': '',
    'lexicon': '',
    'output': '',
    'model': ''
}

# 全局状态
training_status = {
    'is_training': False,
    'current_step': '',
    'progress': 0,
    'total': 100,
    'epoch': 0,
    'total_epochs': 0,
    'train_loss': 0,
    'val_loss': 0,
    'early_stopped': False,
    'completed': False,
    'error': None
}

test_status = {
    'model_loaded': False,
    'loading': False,
    'progress': 0,
    'error': None
}

# 全局模型（测试用）
generator_model = None
generator_tokenizer = None
generator_device = None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html', emotions=EMOTIONS)


def open_file_dialog(file_type):
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if file_type == 'corpus':
        path = filedialog.askopenfilename(
            title='选择语料文件',
            filetypes=[('文本文件', '*.txt'), ('所有文件', '*.*')]
        )
    elif file_type == 'model':
        path = filedialog.askopenfilename(
            title='选择模型文件',
            filetypes=[('PyTorch模型', '*.pt'), ('所有文件', '*.*')]
        )
    elif file_type == 'lexicon':
        path = filedialog.askdirectory(title='选择情感词典文件夹')
    elif file_type == 'output':
        path = filedialog.askdirectory(title='选择输出文件夹')
    else:
        path = ''
    
    root.destroy()
    return path if path else ''


@app.route('/api/select_file', methods=['POST'])
def select_file():
    """选择文件或文件夹"""
    global selected_paths
    data = request.json
    file_type = data.get('type', '')
    
    path = open_file_dialog(file_type)
    
    if not path:
        return jsonify({'success': False, 'path': '', 'message': '未选择'})
    
    selected_paths[file_type] = path
    
    # 如果是情感词典文件夹，检查是否有txt文件
    if file_type == 'lexicon':
        txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
        if not txt_files:
            return jsonify({
                'success': False, 
                'path': path, 
                'message': '该文件夹内没有txt文件，请重新选择'
            })
        return jsonify({
            'success': True, 
            'path': path, 
            'message': f'找到 {len(txt_files)} 个txt文件'
        })
    
    return jsonify({'success': True, 'path': path, 'message': '已选择'})


@app.route('/api/training_status')
def get_training_status():
    """获取训练状态"""
    return jsonify(training_status)


@app.route('/api/start_training', methods=['POST'])
def start_training():
    """开始训练"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'message': '训练正在进行中'})
    
    data = request.json
    corpus_path = data.get('corpus_path', '')
    lexicon_dir = data.get('lexicon_dir', '')
    output_dir = data.get('output_dir', '')
    
    # 验证路径
    if not os.path.exists(corpus_path):
        return jsonify({'success': False, 'message': '语料文件不存在'})
    if not os.path.isdir(lexicon_dir):
        return jsonify({'success': False, 'message': '情感词典文件夹不存在'})
    
    # 检查情感词典文件夹
    txt_files = [f for f in os.listdir(lexicon_dir) if f.endswith('.txt')]
    if not txt_files:
        return jsonify({'success': False, 'message': '情感词典文件夹内没有txt文件'})
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 重置状态
    training_status = {
        'is_training': True,
        'current_step': '初始化...',
        'progress': 0,
        'total': 100,
        'epoch': 0,
        'total_epochs': 5,
        'train_loss': 0,
        'val_loss': 0,
        'early_stopped': False,
        'completed': False,
        'error': None
    }
    
    # 在后台线程中运行训练
    thread = threading.Thread(
        target=run_training,
        args=(corpus_path, lexicon_dir, output_dir)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '训练已开始'})


def run_training(corpus_path, lexicon_dir, output_dir):
    """后台训练函数"""
    global training_status
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ========== 步骤1: 数据处理 ==========
        training_status['current_step'] = '数据处理中...'
        training_status['progress'] = 0
        
        processor = DataProcessor(corpus_path, lexicon_dir)
        
        # 处理数据
        processed_data_path = os.path.join(output_dir, 'processed_data.json')
        data = processor.load_and_process_corpus(max_sentences=500000)
        processor.save_processed_data(data, processed_data_path)
        
        training_status['progress'] = 20
        
        # 采样数据
        max_samples = 100000
        if len(data) > max_samples:
            data = random.sample(data, max_samples)
        
        # ========== 步骤2: 加载预训练模型 ==========
        training_status['current_step'] = '加载预训练模型...'
        training_status['progress'] = 30
        
        tokenizer = load_tokenizer()
        training_status['progress'] = 40
        
        model = load_model(tokenizer)
        model = model.to(device)
        training_status['progress'] = 50
        
        # ========== 步骤3: 创建数据集 ==========
        training_status['current_step'] = '创建数据集...'
        
        dataset = EmotionBartDataset(data, tokenizer, max_input_len=64, max_output_len=64)
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        training_status['progress'] = 60
        
        # ========== 步骤4: 训练 ==========
        training_status['current_step'] = '训练中...'
        
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        num_epochs = 5
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        training_status['total_epochs'] = num_epochs
        
        train_losses = []
        val_losses = []
        step_losses = []
        best_val_loss = float('inf')
        patience = 0
        accum_steps = 4
        
        for epoch in range(num_epochs):
            training_status['epoch'] = epoch + 1
            training_status['current_step'] = f'训练 Epoch {epoch + 1}/{num_epochs}'
            
            # 训练
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / accum_steps
                loss.backward()
                
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                actual_loss = loss.item() * accum_steps
                total_loss += actual_loss
                step_losses.append(actual_loss)
                
                # 更新进度
                epoch_progress = (step + 1) / len(train_loader)
                overall_progress = 60 + (epoch * 8) + int(epoch_progress * 6)
                training_status['progress'] = min(overall_progress, 95)
            
            train_loss = total_loss / len(train_loader)
            train_losses.append(train_loss)
            training_status['train_loss'] = round(train_loss, 4)
            
            # 验证
            training_status['current_step'] = f'验证 Epoch {epoch + 1}/{num_epochs}'
            model.eval()
            val_total_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                    val_total_loss += outputs.loss.item()
            
            val_loss = val_total_loss / len(val_loader)
            val_losses.append(val_loss)
            training_status['val_loss'] = round(val_loss, 4)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }, os.path.join(output_dir, 'emotion_bart_best.pt'))
            else:
                patience += 1
                if patience >= 2:
                    training_status['early_stopped'] = True
                    break
        
        # 保存最终模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }, os.path.join(output_dir, 'emotion_bart_final.pt'))
        
        # 绘制损失曲线
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(train_losses, 'b-o', label='训练')
        axes[0].plot(val_losses, 'r-s', label='验证')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Epoch 损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if len(step_losses) > 30:
            w = 20
            smoothed = [sum(step_losses[max(0, i - w):i + 1]) / min(i + 1, w)
                        for i in range(len(step_losses))]
            axes[1].plot(smoothed, 'g-', alpha=0.8)
        else:
            axes[1].plot(step_losses, 'g-')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Step 损失曲线')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        
        training_status['progress'] = 100
        training_status['current_step'] = '训练完成！'
        training_status['completed'] = True
        training_status['is_training'] = False
        
    except Exception as e:
        training_status['error'] = str(e)
        training_status['is_training'] = False
        training_status['current_step'] = f'错误: {str(e)}'


# ==================== 测试相关 API ====================

@app.route('/api/test_status')
def get_test_status():
    """获取测试状态"""
    return jsonify(test_status)


@app.route('/api/load_model', methods=['POST'])
def load_test_model():
    """加载测试模型"""
    global generator_model, generator_tokenizer, generator_device, test_status
    
    if test_status['loading']:
        return jsonify({'success': False, 'message': '模型正在加载中'})
    
    data = request.json
    model_path = data.get('model_path', '')
    
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'message': '模型文件不存在'})
    
    test_status['loading'] = True
    test_status['progress'] = 0
    test_status['error'] = None
    
    # 在后台线程中加载模型
    thread = threading.Thread(target=load_model_thread, args=(model_path,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '开始加载模型'})


def load_model_thread(model_path):
    """后台加载模型"""
    global generator_model, generator_tokenizer, generator_device, test_status
    
    try:
        test_status['progress'] = 10
        generator_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        test_status['progress'] = 30
        generator_tokenizer = load_tokenizer()
        
        test_status['progress'] = 50
        generator_model = load_model(generator_tokenizer)
        
        test_status['progress'] = 70
        ckpt = torch.load(model_path, map_location=generator_device)
        generator_model.load_state_dict(ckpt['model_state_dict'])
        
        test_status['progress'] = 90
        generator_model = generator_model.to(generator_device)
        generator_model.eval()
        
        test_status['progress'] = 100
        test_status['model_loaded'] = True
        test_status['loading'] = False
        
    except Exception as e:
        test_status['error'] = str(e)
        test_status['loading'] = False
        test_status['model_loaded'] = False


@app.route('/api/generate', methods=['POST'])
def generate():
    """生成文本"""
    global generator_model, generator_tokenizer, generator_device
    
    if not test_status['model_loaded']:
        return jsonify({'success': False, 'message': '请先加载模型'})
    
    data = request.json
    keyword = data.get('keyword', '').strip()
    emotion = data.get('emotion', '平静')
    
    if not keyword:
        return jsonify({'success': False, 'message': '请输入关键词'})
    
    if emotion not in EMOTIONS:
        emotion = '平静'
    
    try:
        results = generate_text(
            generator_model, generator_tokenizer, keyword, emotion,
            generator_device, num_return=3
        )
        return jsonify({'success': True, 'results': results, 'emotion': emotion, 'keyword': keyword})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/unload_model', methods=['POST'])
def unload_model():
    """卸载模型"""
    global generator_model, generator_tokenizer, test_status
    
    generator_model = None
    generator_tokenizer = None
    test_status['model_loaded'] = False
    test_status['progress'] = 0
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return jsonify({'success': True, 'message': '模型已卸载'})


if __name__ == '__main__':
    # 创建模板目录
    os.makedirs('templates', exist_ok=True)
    print("=" * 50)
    print("情感文本生成系统 - Web UI")
    print("=" * 50)
    print("请在浏览器中打开: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
