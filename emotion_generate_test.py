"""
情感文本生成 - 测试脚本 (BART)
调用 emotion_model.py 中的函数
"""
import os
import torch

# 导入自定义模块
from emotion_model import EMOTIONS, load_tokenizer, load_model, generate_text

# 情感别名
EMOTION_ALIAS = {
    '高兴': '开心', '快乐': '开心', '愉快': '开心',
    '难过': '伤心', '悲伤': '伤心',
    '愤怒': '生气', '恼火': '生气',
    '冷静': '平静',
    '希望': '期待',
    '沮丧': '失望'
}


class EmotionGenerator:
    """情感文本生成器"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # 加载 Tokenizer 和模型
        self.tokenizer = load_tokenizer()
        self.model = load_model(self.tokenizer)
        
        # 加载训练好的权重
        print(f"加载模型权重: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', None))
        if val_loss is not None:
            print(f"✓ 模型加载完成 (val_loss: {val_loss:.4f})")
        else:
            print("✓ 模型加载完成")
    
    def _normalize_emotion(self, emotion):
        """标准化情感名称"""
        if emotion in EMOTIONS:
            return emotion
        return EMOTION_ALIAS.get(emotion, '平静')
    
    def generate(self, keyword, emotion, num=3):
        """生成情感文本"""
        emotion = self._normalize_emotion(emotion)
        return generate_text(
            self.model, self.tokenizer, keyword, emotion, 
            self.device, num_return=num
        )
    
    def batch_test(self):
        """批量测试"""
        cases = [
            ("下雨", "开心"),
            ("下雨", "伤心"),
            ("考试", "期待"),
            ("考试", "失望"),
            ("工作", "生气"),
            ("天气", "平静"),
        ]
        
        print("\n" + "=" * 50)
        print("批量测试")
        
        for kw, emo in cases:
            print(f"\n关键词: {kw}, 情感: {emo}")
            print("-" * 40)
            
            results = self.generate(kw, emo, num=2)
            for i, r in enumerate(results, 1):
                r = r.replace(" ", "").replace(" ", "")
                mark = "✓" if kw in r else "✗"
                print(f"  {i}. {mark} {r}")
    
    def chat(self):
        """交互式生成"""
        print("\n" + "=" * 50)
        print("情感文本生成系统")
        print("=" * 50)
        print(f"支持的情感: {', '.join(EMOTIONS)}")
        print("\n输入格式: 关键词,情感")
        print("例如: 下雨,开心")
        print("\n输入 quit 退出")
        print("=" * 50)
        
        while True:
            try:
                inp = input("\n> ").strip()
                
                if inp.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break
                
                if not inp:
                    continue
                
                # 解析输入
                if ',' in inp or '，' in inp:
                    parts = inp.replace('，', ',').split(',', 1)
                    keyword = parts[0].strip()
                    emotion = parts[1].strip() if len(parts) > 1 else '平静'
                else:
                    keyword = inp
                    emotion = input(f"选择情感 ({'/'.join(EMOTIONS)}): ").strip() or '平静'
                
                emotion = self._normalize_emotion(emotion)
                
                # 生成
                print(f"\n[{emotion}] 关键词: {keyword}")
                print("-" * 40)
                
                results = self.generate(keyword, emotion, num=3)
                for i, r in enumerate(results, 1):
                    r = r.replace(" ", "").replace(" ", "")
                    mark = "✓" if keyword in r else "✗"
                    print(f"  {i}. {mark} {r}")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # 查找模型文件
    model_path = 'molde/emotion_bart_best.pt'
    if not os.path.exists(model_path):
        alt_path = 'molde/emotion_bart_final.pt'
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"模型文件不存在: {model_path}")
            print("请先运行 emotion_generate_bart.py 训练模型")
            return
    
    # 创建生成器
    generator = EmotionGenerator(model_path, device)
    # 批量测试
    generator.batch_test()
    # 交互式测试
    generator.chat()


if __name__ == '__main__':
    main()
