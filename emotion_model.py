"""
情感文本生成 - 模型模块
包含 BART 模型、数据集、Tokenizer 加载等
"""
import random

import jieba
import torch
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BertTokenizer
from tqdm import tqdm
import jieba.analyse
import jieba.posseg as pseg

# 情感类别
EMOTIONS = ['开心', '伤心', '生气', '平静', '期待', '失望']


def extract_keywords(text, top_k=3):
    # 从一句中文中提取关键词（基于 TF-IDF）
    keywords = jieba.analyse.extract_tags(
        text,
        topK=top_k,
        withWeight=False,
        allowPOS=('n', 'ns', 'nt', 'v', 'vn')  # 普通名词、地名、机构团体名、普通动词、名动词
    )
    return keywords


def load_tokenizer():
    """加载 Tokenizer 并添加情感标记"""
    print("加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
    
    # 添加情感特殊标记
    special_tokens = [f'[{e}]' for e in EMOTIONS]
    tokenizer.add_tokens(special_tokens)
    
    print(f"Tokenizer 词表大小: {len(tokenizer)}")
    return tokenizer


def load_model(tokenizer, pretrained=True):
    """加载 BART 模型"""
    print("加载 BART 模型...")
    
    if pretrained:
        try:
            model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
            print("✓ 预训练模型加载成功")
        except Exception as e:
            print(f"预训练模型加载失败: {e}")
            raise e
    else:
        from transformers import BartConfig
        config = BartConfig(vocab_size=len(tokenizer))
        model = BartForConditionalGeneration(config)
        print("使用随机初始化模型")
    
    # 调整词表大小
    model.resize_token_embeddings(len(tokenizer))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    return model


class EmotionBartDataset(Dataset):
    """
    BART 情感生成数据集
    
    输入格式: "关键词 [情感]"
    输出格式: "包含关键词的完整句子"
    """
    
    def __init__(self, data, tokenizer, max_input_len=32, max_output_len=64):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.samples = []
        
        print("构建数据集...")
        
        for item in tqdm(data, desc="处理数据"):
            text = item['text'].strip()
            emotion = item['emotion']
            
            # 过滤
            if len(text) < 5 or len(text) > 40:
                continue
            if emotion not in EMOTIONS:
                continue
            
            # 提取关键词
            keywords = extract_keywords(text)
            if not keywords:
                continue
            # 随机数
            num = random.randint(1, len(keywords))-1
            keyword = keywords[num]
            
            # 确保关键词在句子中
            if keyword not in text:
                continue
            
            # 输入: "关键词 [情感]"
            input_text = f"{keyword} [{emotion}]"
            # 输出: 完整句子
            output_text = text
            
            # Tokenize 输入
            input_enc = tokenizer(
                input_text,
                max_length=max_input_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize 输出
            output_enc = tokenizer(
                output_text,
                max_length=max_output_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Labels: padding 位置设为 -100
            labels = output_enc['input_ids'].squeeze(0).clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            self.samples.append({
                'input_ids': input_enc['input_ids'].squeeze(0),
                'attention_mask': input_enc['attention_mask'].squeeze(0),
                'labels': labels
            })
        
        print(f"数据集大小: {len(self.samples)}")
        
        # 统计情感分布
        self._print_stats(data)
    
    def _print_stats(self, data):
        """打印数据统计"""
        emotion_counts = {}
        for item in data:
            e = item['emotion']
            if e in EMOTIONS:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        print("情感分布:")
        for e, c in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            print(f"  {e}: {c}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def generate_text(model, tokenizer, keyword, emotion, device, num_return=3, max_length=50):
    """
    生成情感文本
    
    Args:
        model: BART 模型
        tokenizer: Tokenizer
        keyword: 关键词
        emotion: 情感
        device: 设备
        num_return: 返回数量
        max_length: 最大长度
    
    Returns:
        生成的句子列表
    """
    model.eval()
    
    # 输入格式: "关键词 [情感]"
    input_text = f"{keyword} [{emotion}]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return,
            do_sample=True,
            temperature=0.9,  # 控制随机性
            top_k=50,  # 只从概率最高的50个词中选择
            top_p=0.92,  # 核采样，累积概率达到92%时截断
            repetition_penalty=1.2,  # 惩罚重复词
            pad_token_id=tokenizer.pad_token_id
        )
    
    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()
        if text and text not in results:
            results.append(text)
    
    return results


if __name__ == '__main__':
    # 测试模块
    tokenizer = load_tokenizer()
    model = load_model(tokenizer)
    
    print("\n测试生成:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    results = generate_text(model, tokenizer, "下雨", "开心", device)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
