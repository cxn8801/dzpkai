from treys import Deck, Evaluator, Card
import random
from itertools import combinations
import time
from functools import lru_cache
from multiprocessing import Pool
import joblib
from sklearn.ensemble import RandomForestClassifier

# ======================
# 核心优化模块
# ======================

class RangeManager:
    """手牌范围管理（预计算Top手牌）"""
    def __init__(self):
        self.evaluator = Evaluator()
        self.ranked_hands = None
    
    def _evaluate_hand(self, hand):
        return self.evaluator.evaluate([], list(hand))
    
    def generate_ranges(self, save_path='hand_ranks.pkl'):
        """预生成手牌强度排名"""
        deck = Deck.GetFullDeck()
        all_hands = list(combinations(deck, 2))
        
        print("Evaluating all possible hands...")
        with Pool() as p:
            scores = p.map(self._evaluate_hand, all_hands)
        
        sorted_data = sorted(zip(all_hands, scores), key=lambda x: x[1])
        self.ranked_hands = [h for h, _ in sorted_data]
        
        # 保存预计算数据
        with open(save_path, 'wb') as f:
            joblib.dump(self.ranked_hands, f)
    
    def load_ranges(self, load_path='hand_ranks.pkl'):
        """加载预计算范围"""
        with open(load_path, 'rb') as f:
            self.ranked_hands = joblib.load(f)
    
    def get_range(self, percentile):
        """获取指定百分位的手牌范围"""
        cutoff = int(len(self.ranked_hands) * percentile / 100)
        return self.ranked_hands[:cutoff]

class CachedEvaluator:
    """带缓存的牌力评估器"""
    def __init__(self):
        self.evaluator = Evaluator()
    
    @lru_cache(maxsize=200000)
    def evaluate(self, community_tuple, hand_tuple):
        """输入必须为排序后的元组"""
        return self.evaluator.evaluate(
            list(community_tuple), 
            list(hand_tuple)
        )
    
    def get_rank_class(self, score):
        return self.evaluator.get_rank_class(score)

# ======================
# 扑克模拟器（优化版）
# ======================

class EnhancedPokerSimulator:
    def __init__(self, num_players=6, use_range=True, use_cache=True):
        self.num_players = num_players
        self.range_manager = RangeManager()
        self.use_range = use_range
        
        try:
            self.range_manager.load_ranges()
            self.top_range = self.range_manager.get_range(20)
        except:
            print("未找到预计算范围，使用随机策略")
            self.use_range = False
        
        self.evaluator = CachedEvaluator() if use_cache else Evaluator()
    
    def _get_remaining_deck(self, known_cards):
        """使用集合加速剩余牌计算"""
        full_deck = set(Deck.GetFullDeck())
        return list(full_deck - set(known_cards))
    
    def _simulate_opponents(self, remaining_deck, num_opponents):
        """智能对手手牌生成"""
        if self.use_range:
            valid_hands = [
                hand for hand in self.top_range
                if set(hand).issubset(remaining_deck)
            ]
            if len(valid_hands) >= num_opponents:
                selected = random.sample(valid_hands, num_opponents)
                used = set(card for hand in selected for card in hand)
                return selected, list(set(remaining_deck) - used)
        
        # 回退到随机策略
        selected = random.sample(remaining_deck, num_opponents*2)
        return [
            selected[i*2 : (i+1)*2] 
            for i in range(num_opponents)
        ], list(set(remaining_deck) - set(selected))
    
    def _complete_community(self, current, street, available):
        """分阶段生成公共牌"""
        needed = {'preflop':5, 'flop':2, 'turn':1, 'river':0}[street]
        return current + random.sample(available, needed)
    
    def calculate_equity(self, hero_hand, community=[], iterations=1000, street='preflop'):
        wins = ties = total = 0
        
        for _ in range(iterations):
            # 准备牌堆
            known = hero_hand + community
            deck = self._get_remaining_deck(known)
            if len(deck) < (self.num_players-1)*2 + (5-len(community)):
                continue  # 跳过牌不足的情况
                
            # 生成对手牌
            opponents, remaining = self._simulate_opponents(deck, self.num_players-1)
            
            # 生成公共牌
            community_full = self._complete_community(community.copy(), street, remaining)
            if len(community_full) !=5:
                continue
                
            # 转换缓存键
            comm_key = tuple(sorted(community_full))
            hero_key = tuple(sorted(hero_hand))
            
            # 评估牌力
            try:
                hero_score = self.evaluator.evaluate(comm_key, hero_key)
                opp_scores = [
                    self.evaluator.evaluate(comm_key, tuple(sorted(h))) 
                    for h in opponents
                ]
                min_opp = min(opp_scores)
                
                if hero_score < min_opp:
                    wins +=1
                elif hero_score == min_opp:
                    ties +=1
                total +=1
            except:
                continue
        
        equity = (wins + ties/2) / total if total else 0
        return equity, (wins, ties, total)

# ======================
# 智能决策模块
# ======================

class NashStrategy:
    """纳什均衡策略库"""
    STRATEGY_TABLE = {
        # 格式: (位置, 胜率区间): {动作: 权重}
        ('BTN', (0.7, 1.0)): {'raise': 65, 'call': 25, 'fold': 10},
        ('BTN', (0.5, 0.7)): {'raise': 40, 'call': 50, 'fold': 10},
        ('UTG', (0.7, 1.0)): {'raise': 50, 'call': 30, 'fold': 20},
        # ...可扩展更多策略
    }
    
    @classmethod
    def get_strategy(cls, equity, position):
        for (pos_range, (min_eq, max_eq)), weights in cls.STRATEGY_TABLE.items():
            if position in pos_range and min_eq <= equity < max_eq:
                return weights
        return {'call': 50, 'fold': 50}  # 默认策略

# ======================
# 修复后的机器学习模块
# ======================
class MLPredictor:
    """机器学习预测模块（带应急机制）"""
    def __init__(self, model_path='poker_model.pkl'):
        self.model = None
        self.is_valid = False  # 模型是否有效
        
        try:
            self.model = joblib.load(model_path)
            self.is_valid = True
            print("成功加载预训练模型")
        except Exception as e:
            print(f"模型加载失败: {str(e)}，初始化应急模型")
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """初始化可安全使用的应急模型"""
        from sklearn.dummy import DummyClassifier
        self.model = DummyClassifier(strategy="uniform")
        # 生成虚拟数据
        import numpy as np
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        self.model.fit(X, y)
    
    def predict(self, features):
        try:
            proba = self.model.predict_proba([features])[0]
            return {
                'fold': proba[0],
                'call': proba[1],
                'raise': proba[2]
            }
        except Exception as e:
            print(f"预测失败: {str(e)}，返回均匀分布")
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}

# ======================
# 整合决策引擎
# ======================

class DecisionEngine:
    def __init__(self):
        self.simulator = EnhancedPokerSimulator()
        self.strategy = NashStrategy()
        self.ml = MLPredictor()
    
    def make_decision(self, game_state):
        # 计算胜率
        equity, _ = self.simulator.calculate_equity(
            game_state['hero_hand'],
            game_state['community'],
            iterations=500,
            street=game_state['street']
        )
        
        # 获取基础策略
        base_weights = self.strategy.get_strategy(
            equity, 
            game_state['position']
        )
        
        # 获取机器学习预测
        features = self._extract_features(game_state, equity)
        # 安全获取机器学习预测
        try:
            ml_weights = self.ml.predict(features)
            # 验证预测结果
            if not all(0 <= v <= 1 for v in ml_weights.values()):
                raise ValueError("无效的概率值")
        except Exception as e:
            print(f"机器学习预测异常: {str(e)}，使用备用策略")
            ml_weights = {'fold': 0, 'call': 1, 'raise': 0}  # 保守策略
        
        # 调整策略融合逻辑
        final_weights = {}
        total_ml = sum(ml_weights.values()) or 1
        for action in base_weights.keys():
            # 标准化机器学习权重
            ml_normalized = ml_weights.get(action, 0) / total_ml
            
            # 加权融合
            final = 0.7 * base_weights.get(action, 0) + 0.3 * ml_normalized * 100
            final_weights[action] = max(0, int(final))
        
        # 生成决策
        return self._format_decision(final_weights, game_state)
    
    def _extract_features(self, state, equity):
        """特征工程"""
        return [
            equity,
            len(state['community']),  # 当前阶段
            state['current_pot'] / 100,  # 标准化筹码量
            state['position'] in ['BTN', 'CO'],  # 位置优势
            len(state['previous_actions'])
        ]
    
    def _format_decision(self, weights, game_state):
        """生成可读决策"""
        total = sum(weights.values()) or 1
        actions = []
        for action, weight in weights.items():
            prob = int(weight / total * 100)
            
            if action in ['raise', 'bet']:
                # 智能下注量计算
                size = self._calculate_bet_size(game_state)
                actions.append(f"{prob}% {action} {size}BB")
            else:
                actions.append(f"{prob}% {action}")
        return actions
    
    def _calculate_bet_size(self, game_state):
        """基于底池的智能下注量"""
        pot_size = game_state['current_pot']
        return int(pot_size * 0.6)  # 下注60%底池

# ======================
# 示例用法
# ======================

if __name__ == "__main__":
    # 初始化决策引擎
    engine = DecisionEngine()
    
    # 构造游戏状态
    game_state = {
        'hero_hand': [Card.new('Ah'), Card.new('Kd')],
        'community': [Card.new('Qs'), Card.new('Jh'), Card.new('Tc')],
        'street': 'flop',
        'position': 'BTN',
        'current_pot': 150,
        'previous_actions': ['check', 'call'],
        'num_players': 6
    }
    
    # 获取决策
    decision = engine.make_decision(game_state)
    print("推荐决策:", decision)
    
    # 性能测试
    start = time.time()
    for _ in range(10):
        engine.make_decision(game_state)
    print(f"平均决策时间: {(time.time()-start)/10:.3f}s")