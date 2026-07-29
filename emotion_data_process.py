"""
数据预处理模块
"""
import os
import json
import jieba
from collections import Counter
import random


class EmotionMapper:
    """情感映射器：将情感词典映射到简单情感"""
    def __init__(self):
        # 定义简单情感类别
        self.simple_emotions = {
            '开心': ['正面情感词语', '正面评价词语'],
            '伤心': ['负面情感词语'],
            '生气': ['负面情感词语', '负面评价词语'],
            '平静': ['程度级别词语'],
            '期待': ['正面情感词语', '主张词语'],
            '失望': ['负面情感词语', '负面评价词语']
        }
        
        self.emotion_words = {}  # 存储每种简单情感对应的词汇
        self.lexicon_to_simple = {}  # 词典类别到简单情感的映射
    
    def load_emotion_lexicon(self, emotion_lexicon_dir):
        """加载情感词典并映射到简单情感"""
        print("加载情感词典...")
        
        # 先加载原始词典
        raw_lexicon = {}
        for filename in os.listdir(emotion_lexicon_dir):
            if filename.endswith('.txt'):
                lexicon_name = filename.replace('.txt', '')
                filepath = os.path.join(emotion_lexicon_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
                    raw_lexicon[lexicon_name] = set(words)
        
        # 映射到简单情感
        for simple_emotion, lexicon_list in self.simple_emotions.items():
            combined_words = set()
            for lexicon_name in lexicon_list:
                if lexicon_name in raw_lexicon:
                    combined_words.update(raw_lexicon[lexicon_name])
                    self.lexicon_to_simple[lexicon_name] = simple_emotion
            
            self.emotion_words[simple_emotion] = combined_words
        
        print(f"映射到 {len(self.emotion_words)} 种简单情感:")
        for emotion, words in self.emotion_words.items():
            print(f"  {emotion}: {len(words)} 个词")
        
        return self.emotion_words
    
    def detect_emotion(self, sentence):
        """检测句子的情感，返回情感和强度"""
        words = jieba.lcut(sentence)
        emotion_scores = {emotion: 0 for emotion in self.emotion_words.keys()}
        
        for word in words:
            for emotion, emotion_word_set in self.emotion_words.items():
                if word in emotion_word_set:
                    emotion_scores[emotion] += 1
        
        # 返回得分最高的情感
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]
        
        if max_score > 0:
            return max_emotion, max_score
        else:
            return '平静', 0


class DataProcessor:
    def __init__(self, corpus_path, emotion_lexicon_dir):
        self.corpus_path = corpus_path
        self.emotion_mapper = EmotionMapper()
        self.emotion_mapper.load_emotion_lexicon(emotion_lexicon_dir)
        
    def load_and_process_corpus(self, max_sentences=None, min_length=5, max_length=100):
        """加载并处理语料库"""
        print("加载对话语料...")
        processed_data = []
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_sentences and idx >= max_sentences:
                    break
                
                line = line.strip()
                # 过滤太短或太长的句子
                if min_length <= len(line) <= max_length:
                    # 检测情感
                    emotion, score = self.emotion_mapper.detect_emotion(line)
                    # 对「低置信度情感」样本丢弃
                    if score < 2:
                        continue

                    processed_data.append({
                        'text': line,
                        'emotion': emotion,
                        'emotion_score': score
                    })
        
        print(f"加载了 {len(processed_data)} 条数据")
        
        # 统计情感分布
        emotion_dist = {}
        for item in processed_data:
            emotion = item['emotion']
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        print("\n情感分布:")
        for emotion, count in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count} ({count/len(processed_data)*100:.1f}%)")
        
        return processed_data
    
    def save_processed_data(self, data, save_path='saved/processed_data.json'):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据已保存到 {save_path}")
    
    def load_processed_data(self, save_path='saved/processed_data.json'):
        """加载处理后的数据"""
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"加载了 {len(data)} 条数据")
        return data
    
    def create_emotion_prompts(self, keyword, emotion):
        """创建情感提示文本"""
        # 不同情感的提示模板
        templates = {
            '开心': [
                f"[开心]{keyword}",
                f"[高兴]{keyword}",
                f"[愉快]{keyword}"
            ],
            '伤心': [
                f"[伤心]{keyword}",
                f"[难过]{keyword}",
                f"[悲伤]{keyword}"
            ],
            '生气': [
                f"[生气]{keyword}",
                f"[愤怒]{keyword}",
                f"[恼火]{keyword}"
            ],
            '平静': [
                f"[平静]{keyword}",
                f"{keyword}"
            ],
            '期待': [
                f"[期待]{keyword}",
                f"[希望]{keyword}"
            ],
            '失望': [
                f"[失望]{keyword}",
                f"[沮丧]{keyword}"
            ]
        }
        
        return random.choice(templates.get(emotion, [f"{keyword}"]))


if __name__ == '__main__':
    # 测试数据处理
    processor = DataProcessor(
        corpus_path='data/corpus.txt',
        emotion_lexicon_dir='data/Emotional_Lexicon'
    )
    
    # 处理数据
    data = processor.load_and_process_corpus(max_sentences=100000)
    
    # 保存数据
    processor.save_processed_data(data)
    
    # 显示示例
    print("\n数据示例:")
    for i in range(min(10, len(data))):
        print(f"文本: {data[i]['text']}")
        print(f"情感: {data[i]['emotion']} (强度: {data[i]['emotion_score']})")
        print()
