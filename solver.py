from treys import Deck, Evaluator, Card
import random
from itertools import combinations
import time
from functools import lru_cache
from multiprocessing import Pool
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier
from joblib import dump, load
from datetime import datetime
from collections import deque
import hashlib
import os
import json
from threading import Lock
import re
from collections import defaultdict, deque
# 原导入部分（添加以下内容）
from sklearn.linear_model import Ridge  # 新增导入

# ====================== 常量与工具 ======================
NUM_FEATURES = 15
STAGES = ['preflop', 'flop', 'turn', 'river']
POSITIONS = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
ACTION_SPACE = ['fold', 'call', 'raise']

BB = 1.0       # 大盲单位
SB = 0.5       # 小盲单位

def create_card(card_str):
    """字符串转treys Card对象"""
    return Card.new(card_str)

# ====================== Range与牌力模块 ======================
class RangeManager:
    """手牌范围管理（预计算Top手牌）"""
    def __init__(self):
        self.evaluator = Evaluator()
        self.ranked_hands = None

    def _evaluate_hand(self, hand):
        return self.evaluator.evaluate([], list(hand))

    def generate_ranges(self, save_path='hand_ranks.pkl'):
        deck = Deck.GetFullDeck()
        all_hands = list(combinations(deck, 2))
        print("Evaluating all possible hands...")
        with Pool() as p:
            scores = p.map(self._evaluate_hand, all_hands)
        sorted_data = sorted(zip(all_hands, scores), key=lambda x: x[1])
        self.ranked_hands = [h for h, _ in sorted_data]
        with open(save_path, 'wb') as f:
            joblib.dump(self.ranked_hands, f)

    def load_ranges(self, load_path='hand_ranks.pkl'):
        with open(load_path, 'rb') as f:
            self.ranked_hands = joblib.load(f)

    def get_range(self, percentile):
        cutoff = int(len(self.ranked_hands) * percentile / 100)
        return self.ranked_hands[:cutoff]

class CachedEvaluator:
    """带缓存的牌力评估器"""
    def __init__(self):
        self.evaluator = Evaluator()

    @lru_cache(maxsize=200000)
    def evaluate(self, community_tuple, hand_tuple):
        return self.evaluator.evaluate(list(community_tuple), list(hand_tuple))

    def get_rank_class(self, score):
        return self.evaluator.get_rank_class(score)

# ====================== 扑克模拟器 ======================
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

    def _execute_action(self, game_state, action_type):
        """统一动作执行逻辑"""
        new_state = game_state.copy()
        record = {'action': action_type}

        if action_type == 'call':
            record['bet_size'] = new_state.get('to_call', 0.0)
        elif action_type == 'raise':
            # 计算实际加注量（BB单位）
            last_bet = new_state.get('last_bet', 0.0)
            min_raise = max(2 * last_bet, self.bb)
            raise_size = max(min_raise, new_state['current_pot'] * 0.5)
            raise_size = min(raise_size, new_state['stack'])
            record['bet_size'] = raise_size

        new_state['previous_actions'].append(record)
        return new_state

    def _get_remaining_deck(self, known_cards):
        full_deck = set(Deck.GetFullDeck())
        return list(full_deck - set(known_cards))

    def _simulate_opponents(self, remaining_deck, num_opponents):
        if self.use_range:
            valid_hands = [hand for hand in self.top_range if set(hand).issubset(remaining_deck)]
            if len(valid_hands) >= num_opponents:
                selected = random.sample(valid_hands, num_opponents)
                used = set(card for hand in selected for card in hand)
                return selected, list(set(remaining_deck) - used)
        selected = random.sample(remaining_deck, num_opponents*2)
        return [
            selected[i*2 : (i+1)*2]
            for i in range(num_opponents)
        ], list(set(remaining_deck) - set(selected))

    def _complete_community(self, current, street, available):
        needed = {'preflop':5, 'flop':2, 'turn':1, 'river':0}[street]
        return current + random.sample(available, needed)

    def calculate_equity(self, hero_hand, community=[], iterations=1000, street='preflop'):
        wins = ties = total = 0
        for _ in range(iterations):
            known = hero_hand + community
            deck = self._get_remaining_deck(known)
            if len(deck) < (self.num_players-1)*2 + (5-len(community)):
                continue
            opponents, remaining = self._simulate_opponents(deck, self.num_players-1)
            community_full = self._complete_community(community.copy(), street, remaining)
            if len(community_full) != 5:
                continue
            try:
                comm_key = tuple(sorted(community_full))
                hero_key = tuple(sorted(hero_hand))
                hero_score = self.evaluator.evaluate(comm_key, hero_key)
                opp_scores = [self.evaluator.evaluate(comm_key, tuple(sorted(h))) for h in opponents]
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

# ====================== 机器学习预测模块 ======================
class MLPredictor:
    """机器学习预测模块（带应急机制）"""
    def __init__(self, model_path='poker_model.pkl'):
        self.model = None
        try:
            self.model = joblib.load(model_path)
            print("成功加载预训练模型")
        except Exception as e:
            print(f"模型加载失败: {e}，初始化应急模型")
            self._init_safe_model()
    def _init_safe_model(self):
        self.model = DummyClassifier(strategy="uniform")
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        self.model.fit(X, y)

    def predict(self, features):
        try:
            proba = self.model.predict_proba([features])[0]
            return {'fold': proba[0], 'call': proba[1], 'raise': proba[2]}
        except Exception as e:
            print(f"预测异常: {e}，使用安全返回值")
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}

# ====================== 神经网络模型 ======================
class StageLSTM(nn.Module):
    """分阶段LSTM"""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.stage_emb = nn.Embedding(4, 4)
        self.linear_adapter = nn.Linear(11, 32)
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x, stage_idx, hidden=None):
        stage_emb = self.stage_emb(stage_idx)
        x = torch.cat([x, stage_emb], dim=-1)
        x = self.linear_adapter(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1]), hidden

class GlobalContextNet(nn.Module):
    """上下文网络"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    def forward(self, x):
        return self.encoder(x)

# ====================== 对手建模与经验回放 ======================
class OpponentProfiler:
    """实时对手建模系统"""
    def __init__(self):
        self.cluster_model = MiniBatchKMeans(n_clusters=5, random_state=42)
        self.action_sequences = defaultdict(lambda: deque(maxlen=100))
        self.profile_cache = {}
        self.lock = Lock()

    def update(self, game_state):
        """实时更新对手画像"""
        with self.lock:
            player_id = game_state['player_id']
            action_record = {
                'action': game_state['action'],
                'bet_size': game_state.get('bet_size', 0) / game_state['big_blind'],  # 转换为BB单位
                'timing': game_state.get('timing', 1.0)
            }
            
            # 更新动作序列
            self.action_sequences[player_id].append(action_record)
            
            # 提取特征并更新聚类
            features = self._extract_behavior_features(player_id)
            try:
                self.cluster_model.partial_fit([features])
                self.profile_cache[player_id] = self._get_cluster_profile(
                    self.cluster_model.predict([features])[0]
                )
            except ValueError:
                self.cluster_model = MiniBatchKMeans(n_clusters=5)
                self.cluster_model.partial_fit([features])

    def predict(self, game_state):
        """获取当前对手画像"""
        player_id = game_state['player_id']
        with self.lock:
            # 使用缓存提高性能
            if player_id in self.profile_cache:
                return self.profile_cache[player_id]
            
            # 新玩家默认画像
            return {
                "style": "unknown",
                "fold": 0.33,
                "call": 0.34,
                "raise": 0.33
            }
    
    def _init_clustering(self):
        return MiniBatchKMeans(n_clusters=5, random_state=42)
    
    def _extract_behavior_features(self, player_id):
        """精细化的行为特征提取"""
        seq = list(self.action_sequences[player_id])
        if len(seq) < 5:  # 最少需要5个动作
            return [0.5, 0.0, 0.0, 0.5, 0.0]
        
        # 时间窗口特征
        last_10 = seq[-10:]
        last_5 = seq[-5:]
        
        # 关键指标计算
        metrics = {
            'agg_ratio_10': sum(1 for a in last_10 if a['action'] == 'raise') / 10,
            'avg_bet_10': np.mean([a['bet_size'] for a in last_10]),
            'timing_var': np.var([a['timing'] for a in last_10]),
            'agg_ratio_5': sum(1 for a in last_5 if a['action'] == 'raise') / 5,
            'bet_change': np.mean(np.diff([a['bet_size'] for a in last_5 if a['action'] == 'raise']))
        }
        
        # 处理NaN值
        features = [
            metrics['agg_ratio_10'],
            metrics['avg_bet_10'] if not np.isnan(metrics['avg_bet_10']) else 0.0,
            metrics['timing_var'] if not np.isnan(metrics['timing_var']) else 0.1,
            metrics['agg_ratio_5'],
            metrics['bet_change'] if not np.isnan(metrics['bet_change']) else 0.0
        ]
        
        return features
    
    def _get_cluster_profile(self, cluster):
        profiles = {
            0: {"style": "aggressive", "fold": 0.2, "call": 0.3, "raise": 0.5},
            1: {"style": "passive",    "fold": 0.4, "call": 0.5, "raise": 0.1},
            2: {"style": "loose",      "fold": 0.1, "call": 0.6, "raise": 0.3},
            3: {"style": "tight",      "fold": 0.6, "call": 0.3, "raise": 0.1},
            4: {"style": "balanced",   "fold": 0.3, "call": 0.4, "raise": 0.3},
        }
        return profiles.get(cluster, {"style": "unknown", "fold": 0.33, "call": 0.34, "raise": 0.33})

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def store(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        size = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, size) if size else []

class OnlineAdaptor:
    def __init__(self):
        self.model = self._init_online_model()
    def _init_online_model(self):
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss='log_loss')
    def update(self, batch):
        if not batch: return
        X, y = zip(*batch)
        self.model.partial_fit(X, y, classes=[0,1,2])

# ====================== 混合AI主控 ======================
class HybridPokerAI:
    def __init__(self):       
        self.model_path = 'models/'
        os.makedirs(self.model_path, exist_ok=True)
        self.decision_history = []
        self.reward_normalizer = RewardNormalizer()  # 新增此行

        # 初始化 context_model
        self.context_model = GlobalContextNet()

        self.stage_models = {
            'preflop': XGBClassifier(n_estimators=100),
            'flop': GradientBoostingClassifier(),
            'turn': self._init_pytorch_model('turn'),
            'river': self._init_pytorch_model('river')
        }

        required_files = [
            f'{self.model_path}turn_model.pth',
            f'{self.model_path}river_model.pth',
            f'{self.model_path}preflop_model.joblib'
        ]
        if not all(os.path.exists(f) for f in required_files):
            print("检测到缺失模型文件，正在初始化基础模型...")
            self._create_initial_models()

        self.opponent_model = OpponentProfiler()
        self.memory = ExperienceReplayBuffer(10000)
        self.online_learner = OnlineAdaptor()
        self._load_model_weights()

        self.meta_learner = LinearRegression()
        self.weight_optimizer = None
        self.decision_history = []
        self.history_file = 'decision_history.json'  # 定义历史记录文件路径
        # self.history_file = os.path.abspath('decision_history.json')  # 使用绝对路径
        # os.makedirs(os.path.dirname(self.history_file), exist_ok=True)  # 确保目录存在
        self._train_meta_learner()

        # 新增胜率缓存系统
        self.equity_cache = {}  # 缓存字典 {hash_key: equity}
        self.cache_hits = 0     # 缓存命中统计
        self.cache_misses = 0   # 缓存未命中统计
        self.calc_times = []    # 计算耗时记录
        
        # 并发控制（如果使用多线程）
        self.cache_lock = Lock()

    # ====================== 增强版元学习器训练 ======================
    def _train_meta_learner(self):
        """带默认初始化的元学习器训练"""
        historical_data = self._load_decision_history()
        
        # 初始化默认权重
        self.meta_learner.coef_ = np.array([
            # 上下文特征权重 (7个特征)
            [0.3, 0.2, 0.1, 0.15, 0.15, 0.05, 0.05],   # fold
            [0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05],   # call
            [0.25, 0.15, 0.15, 0.2, 0.15, 0.05, 0.05]  # raise
        ])
        self.meta_learner.intercept_ = np.array([0.1, 0.1, 0.1])
        
        if not historical_data:
            print("使用默认元学习权重")
            return

        X = []
        y = []
        for d in historical_data:
            features = d.get('features', [])
            if isinstance(features, dict):
                features = features.get('context', []) + features.get('stage_features', [])
            if len(features) >= NUM_FEATURES:
                X.append(features[:NUM_FEATURES])
                weights = d.get('decision', {'fold':0.33, 'call':0.34, 'raise':0.33})
                y.append([weights['fold'], weights['call'], weights['raise']])
        
        if len(X) < 10:
            print(f"样本不足 ({len(X)}), 保持默认权重")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # 添加L2正则化防止过拟合
        self.meta_learner = Ridge(alpha=0.5)
        self.meta_learner.fit(X, y)
        print("元学习器训练完成，特征维度:", X.shape)

    def _load_decision_history(self):
        """带默认数据生成的历史记录加载"""
        default_data = []
        
        # 生成20条模拟数据（正态分布）
        np.random.seed(42)
        for _ in range(20):
            features = np.clip(np.random.normal(0.5, 0.2, 15), 0, 1).tolist()
            decision = {
                'fold': float(np.random.beta(2, 5)),
                'call': float(np.random.beta(5, 5)),
                'raise': float(np.random.beta(2, 3))
            }
            # 归一化
            total = sum(decision.values())
            decision = {k: v/total for k, v in decision.items()}
            
            default_data.append({
                'features': features[:7] + features[7:15],  # 确保15维
                'decision': decision,
                'timestamp': datetime.now().isoformat()
            })

        try:
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=2)
                return default_data
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                
            # 数据清洗
            valid_data = []
            for d in raw:
                if not isinstance(d, dict): continue
                if 'features' not in d or 'decision' not in d: continue
                if len(d['features']) != 15: continue
                valid_data.append(d)
                
            return valid_data if len(valid_data) >= 10 else default_data
            
        except Exception as e:
            print(f"历史记录加载失败: {str(e)}，使用默认数据")
            return default_data
        
    def _save_decision_history(self, new_record):
        try:
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append(new_record)
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(
                    history, 
                    f, 
                    indent=2, 
                    default=lambda o: str(o)  # 处理无法序列化的对象
                )
            print(f"成功保存记录到 {self.history_file}")  # 确认保存成功
        except Exception as e:
            print(f"保存失败！错误详情: {str(e)}", file=sys.stderr)  # 输出到标准错误
            import traceback
            traceback.print_exc()  # 打印完整堆栈跟踪

    def _adapt_weights(self, game_state):
        features = self.extract_features(game_state)
        weights = self.meta_learner.predict([features['context']])[0]
        return {action: max(0, weight) for action, weight in zip(ACTION_SPACE, weights)}
    
    def _blend_strategies(self, ml_probs, game_state, opp_profile):
        """动态权重策略融合（带探索机制）"""
        # 获取元学习器权重预测
        try:
            context_features = self.extract_features(game_state)['context']
            weights = self.meta_learner.predict([context_features])[0]
        except Exception as e:
            print(f"元学习器异常: {str(e)}")
            weights = [0.5, 0.3, 0.2]  # 默认权重
        
        # 权重安全限制
        weights = np.clip(weights, 0.1, 0.7)
        weights /= np.sum(weights)  # 归一化

        # 获取各策略基准
        gto_probs = self._get_gto_baseline(game_state)
        opp_adjust = self._calculate_opponent_adjustment(opp_profile)
        
        # 转换机器学习概率
        if isinstance(ml_probs, np.ndarray):
            ml_dict = {
                ACTION_SPACE[i]: float(ml_probs[i])
                for i in range(len(ACTION_SPACE))
            }
        else:
            ml_dict = ml_probs

        # 混合计算
        blended = defaultdict(float)
        for action in ACTION_SPACE:
            blended[action] = (
                weights[0] * ml_dict.get(action, 0.0) +
                weights[1] * gto_probs.get(action, 0.0) +
                weights[2] * opp_adjust.get(action, 0.0)
            )

        # 添加探索噪声
        exploration_rate = 0.08  # 8%探索率
        noise = np.random.dirichlet([1,1,1]) * exploration_rate
        for i, action in enumerate(ACTION_SPACE):
            blended[action] = blended[action] * (1 - exploration_rate) + noise[i]

        # 概率归一化
        total = sum(blended.values())
        return {k: v/total for k, v in blended.items()}
    
    def _get_gto_baseline(self, game_state):
        """根据 GTO 策略返回动作概率分布"""
        street = game_state.get('street', 'preflop')
        position = game_state.get('position', 'BTN')

        # GTO 概率基线
        if street == 'preflop':
            if position in ['BTN', 'CO']:
                return {'fold': 0.15, 'call': 0.35, 'raise': 0.5}
            else:
                return {'fold': 0.25, 'call': 0.5, 'raise': 0.25}
        elif street == 'flop':
            return {'fold': 0.20, 'call': 0.5, 'raise': 0.3}
        elif street == 'turn':
            return {'fold': 0.25, 'call': 0.6, 'raise': 0.15}
        elif street == 'river':
            return {'fold': 0.3, 'call': 0.6, 'raise': 0.1}
        else:
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}
        
    def _calculate_opponent_adjustment(self, opp_profile):
        """根据对手风格返回动作概率分布"""
        if not isinstance(opp_profile, dict):
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}

        return {
            'fold': float(opp_profile.get('fold', 0.33)),
            'call': float(opp_profile.get('call', 0.34)),
            'raise': float(opp_profile.get('raise', 0.33)),
        }
    
    def format_decision(self, blended_probs, game_state):
        """精确到小数点后两位的决策格式化"""
        # 输入校验
        blended_probs = defaultdict(float, blended_probs)
        total = sum(blended_probs.values())
        if total <= 0:
            blended_probs = {'fold': 0.3333, 'call': 0.3333, 'raise': 0.3334}
        else:
            blended_probs = {k: v/total for k, v in blended_probs.items()}

        # 精确计算百分比
        exact_percents = {
            'fold': blended_probs['fold'] * 100,
            'call': blended_probs['call'] * 100,
            'raise': blended_probs['raise'] * 100
        }

        # 四舍五入到两位小数
        rounded = {k: round(v, 2) for k, v in exact_percents.items()}
        total_rounded = sum(rounded.values())

        # 动态调整误差
        diff = round(100.00 - total_rounded, 2)
        if diff != 0:
            # 找到最大项调整
            max_action = max(rounded, key=lambda x: rounded[x])
            rounded[max_action] += diff
            rounded[max_action] = round(rounded[max_action], 2)

        # 格式化显示
        actions = []
        bb = game_state['big_blind']
        
        # 加注量计算（带边界检查）
        min_raise = max(2 * bb, 1.0)
        max_raise = game_state['stack']
        equity = game_state.get('calculated_equity', 0.5)
        base_raise = min(
            max(min_raise, game_state['current_pot'] * (0.4 + 0.3 * equity)),
            max_raise
        )
        raise_bb = round(base_raise, 2)

        for action in ACTION_SPACE:
            percent = rounded.get(action, 0.0)
            
            # 处理小数点显示
            if percent.is_integer():
                display_percent = f"{int(percent)}%"
            else:
                display_percent = f"{percent:.2f}%".rstrip('0').rstrip('.') + "%"

            # 特殊处理call/check
            if action == 'call':
                to_call = game_state.get('to_call', 0.0)
                if to_call == 0:
                    display_action = f"{display_percent} check"
                else:
                    call_bb = f"{to_call:.2f}BB".replace(".00BB", "BB")
                    display_action = f"{display_percent} call {call_bb}"
            
            elif action == 'raise':
                raise_bb_str = f"{raise_bb:.2f}BB".replace(".00BB", "BB")
                display_action = f"{display_percent} raise {raise_bb_str}"
            
            else:
                display_action = f"{display_percent} {action}"

            actions.append(display_action)

        # 最终校验
        final_total = sum(float(a.split('%')[0]) for a in actions)
        if abs(final_total - 100.0) > 0.01:
            return ["33.33% fold", "33.33% call", "33.34% raise"]
        
        return actions

    # ====================== 修改后的模型加载方法 ======================
    def _init_pytorch_model(self, stage):
        """完整修复版PyTorch模型初始化"""
        # 初始化模型结构（必须与训练时完全一致）
        input_size = 32 if stage == 'turn' else 48
        model = StageLSTM(
            input_size=input_size,
            hidden_size=64,
            num_classes=3
        )
        model_path = f'{self.model_path}{stage}_model.pth'

        try:
            # 加载状态字典
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 过滤不匹配的键（兼容性处理）
            model_state = model.state_dict()
            matched_state = {k: v for k, v in state_dict.items() if k in model_state}
            
            # 加载匹配参数
            model_state.update(matched_state)
            model.load_state_dict(model_state)
            
            # 验证模型完整性
            assert model.linear_adapter.in_features == input_size + 4, "输入维度不匹配"
            assert model.stage_emb.num_embeddings == 4, "阶段嵌入层异常"
            
            # 测试推理
            with torch.no_grad():
                test_input = torch.randn(1, 1, input_size)  # (batch, seq, features)
                stage_idx = torch.tensor([[STAGES.index(stage)]], dtype=torch.long)
                _ = model(test_input, stage_idx)
                
            model.eval()
            print(f"✅ {stage}模型加载成功 | 输入维度: {input_size}")
            return model
            
        except Exception as e:
            print(f"❌ {stage}模型加载失败: {str(e)}")
            # 应急初始化
            model.stage_emb = nn.Embedding(4, 4)
            model.linear_adapter = nn.Linear(input_size + 4, 32)
            print(f"⚠️ 应急{stage}模型已启用 | 输入维度: {input_size}")
            return model

    def _load_model_weights(self):
        """完整模型加载流程"""
        print("\n=== 模型加载流程开始 ===")
        
        model_files = {
            'preflop': ('preflop_model.joblib', XGBClassifier),
            'flop': ('flop_model.joblib', GradientBoostingClassifier),
            'turn': ('turn_model.pth', StageLSTM),
            'river': ('river_model.pth', StageLSTM)
        }

        for stage, (filename, model_type) in model_files.items():
            filepath = os.path.join(self.model_path, filename)
            try:
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"{filename} 不存在")

                # 加载传统模型
                if stage in ['preflop', 'flop']:
                    model = joblib.load(filepath)
                    if model.n_features_in_ != NUM_FEATURES:
                        raise ValueError(f"特征数不匹配: 模型{model.n_features_in_} vs 定义{NUM_FEATURES}")
                    self.stage_models[stage] = model
                    print(f"✅ {stage}模型加载成功 | 特征数: {model.n_features_in_}")
                    
                # 加载PyTorch模型
                else:
                    model = self._init_pytorch_model(stage)
                    
                    # 维度校验
                    input_size = 32 if stage == 'turn' else 48
                    test_input = torch.randn(1, 1, input_size)
                    stage_idx = torch.tensor([[STAGES.index(stage)]], dtype=torch.long)
                    with torch.no_grad():
                        _ = model(test_input, stage_idx)
                        
                    self.stage_models[stage] = model
                    print(f"✅ {stage}模型加载成功 | 参数数: {sum(p.numel() for p in model.parameters())}")

            except Exception as e:
                print(f"❌ {stage}模型加载失败: {str(e)}")
                self._init_fallback_model(stage)

        print("=== 模型加载完成 ===\n")

    def _init_fallback_model(self, stage):
        """带特征校验的应急模型初始化"""
        # 生成符合特征维度的测试数据
        np.random.seed(42)
        dummy_X = np.random.rand(100, NUM_FEATURES)
        dummy_y = np.random.choice([0,1,2], p=[0.3,0.4,0.3], size=100)
        
        if stage in ['preflop', 'flop']:
            # 树模型初始化
            if stage == 'preflop':
                model = XGBClassifier(n_estimators=50)
            else:
                model = GradientBoostingClassifier(n_estimators=50)
                
            model.fit(dummy_X, dummy_y)
            print(f"⚠️ 应急{stage}模型已启用 | 特征数: {model.n_features_in_}")
            
        else:
            # 神经网络模型
            model = StageLSTM(
                input_size=32 if stage == 'turn' else 48,
                hidden_size=64,
                num_classes=3
            )
            print(f"⚠️ 应急{stage}模型已启用 | 输入维度: {32 if stage == 'turn' else 48}")
            
        self.stage_models[stage] = model

    def _create_initial_models(self):
        print("正在生成初始模型文件...")
        # 修改训练样本生成逻辑
        dummy_X = np.random.rand(100, NUM_FEATURES)
        # 调整标签分布反映BB单位策略
        dummy_y = np.random.choice([0,1,2], p=[0.3,0.4,0.3], size=100)

        # 训练 preflop 模型
        self.stage_models['preflop'].fit(dummy_X, dummy_y)
        dump(self.stage_models['preflop'], f'{self.model_path}preflop_model.joblib')

        # 训练 flop 模型
        self.stage_models['flop'].fit(dummy_X, dummy_y)
        dump(self.stage_models['flop'], f'{self.model_path}flop_model.joblib')

        # 保存 turn 和 river 的 PyTorch 模型
        turn_model = StageLSTM(32, 64, 3)
        torch.save(turn_model.state_dict(), f'{self.model_path}turn_model.pth')
        river_model = StageLSTM(48, 64, 3)
        torch.save(river_model.state_dict(), f'{self.model_path}river_model.pth')
        print(f"基础模型已保存至：{os.path.abspath(self.model_path)}")

    def _update_models(self):
        # 从经验池采样
        batch = self.memory.sample(512)
        if not batch:
            return
        # 假设经验格式为 (features, label)
        X, y = zip(*batch)
        X = np.array(X)
        y = np.array(y)
        # 针对每个阶段模型做增量训练（如果支持partial_fit）
        for stage, model in self.stage_models.items():
            if hasattr(model, "partial_fit"):
                classes = np.unique(y)
                model.partial_fit(X, y, classes=classes)
            else:
                # 不支持partial_fit的模型（如XGB/GBDT），可考虑refit或跳过
                pass

    def decide_action(self, game_state):
        """完整修复版决策方法"""
        # 极端情况处理：短码全押
        if game_state.get('stack', 0) <= 2.0:  # ≤2BB时强制全押
            return ['0% fold', '0% check', '100% all-in']
        
        # 状态初始化
        try:
            bb = float(game_state.get('big_blind', BB))  # 确保转换为浮点数
            game_state.setdefault('previous_actions', [])
            game_state.setdefault('street', 'preflop')
        except (TypeError, ValueError) as e:
            print(f"状态初始化错误: {str(e)}, 使用默认值")
            bb = BB
            game_state['street'] = 'preflop'

        # 特征提取（带异常保护）
        try:
            raw_features = self.extract_features(game_state)
            context_features = raw_features['context']
            stage_features = raw_features['stage_features']
        except KeyError as e:
            print(f"特征提取关键字段缺失: {str(e)}, 使用安全特征")
            context_features = np.zeros(7)
            stage_features = np.zeros(NUM_FEATURES-7)

        # 阶段识别
        stage = game_state['street']
        if stage not in STAGES:
            stage = 'preflop'

        # 特征统一处理
        full_features = np.concatenate([context_features, stage_features])
        full_features = full_features.astype(np.float32)

        # 阶段特异性预测
        probs = None
        try:
            if stage in ['turn', 'river']:
                # PyTorch模型处理
                with torch.no_grad():
                    # 确保输入维度正确 (batch_size=1, seq_len=1, features)
                    features_tensor = torch.tensor(
                        full_features, 
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # shape: (1,1,15)
                    
                    # 生成阶段索引
                    stage_idx = torch.tensor(
                        [[STAGES.index(stage)]], 
                        dtype=torch.long
                    )
                    
                    # 模型前向传播
                    logits, _ = self.stage_models[stage](
                        features_tensor, 
                        stage_idx
                    )
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            else:
                # 传统机器学习模型处理
                features = full_features.reshape(1, -1)
                
                # 特征维度修正
                if features.shape[1] < NUM_FEATURES:
                    # 填充零值
                    features = np.pad(
                        features,
                        ((0,0), (0, NUM_FEATURES - features.shape[1])),
                        mode='constant'
                    )
                elif features.shape[1] > NUM_FEATURES:
                    # 截断多余特征
                    features = features[:, :NUM_FEATURES]
                
                # 预测概率
                if hasattr(self.stage_models[stage], 'predict_proba'):
                    probs = self.stage_models[stage].predict_proba(features)[0]
                else:
                    # 应急处理（神经网络模型意外出现在传统阶段）
                    with torch.no_grad():
                        features_tensor = torch.tensor(
                            features, 
                            dtype=torch.float32
                        ).unsqueeze(1)  # 添加序列维度
                        stage_idx = torch.tensor(
                            [[STAGES.index(stage)]], 
                            dtype=torch.long
                        )
                        logits, _ = self.stage_models[stage](
                            features_tensor, 
                            stage_idx
                        )
                        probs = torch.softmax(logits, dim=-1).numpy()[0]
        except Exception as e:
            print(f"模型预测异常: {str(e)}, 使用安全概率")
            probs = np.array([0.33, 0.34, 0.33])

        # 概率合法性校验
        probs = np.clip(probs, 0.0, 1.0)
        if np.sum(probs) == 0:
            probs = np.array([0.33, 0.34, 0.33])
        probs /= np.sum(probs)  # 确保概率归一化

        # 获取对手画像
        try:
            opp_profile = self.opponent_model.predict(game_state)
        except Exception as e:
            print(f"对手建模异常: {str(e)}, 使用默认画像")
            opp_profile = {
                "style": "unknown",
                "fold": 0.33, 
                "call": 0.34, 
                "raise": 0.33
            }

        # 策略融合
        try:
            blended_probs = self._blend_strategies(
                probs, 
                game_state, 
                opp_profile
            )
        except Exception as e:
            print(f"策略融合异常: {str(e)}, 使用原始概率")
            blended_probs = {
                'fold': probs[0],
                'call': probs[1],
                'raise': probs[2]
            }

        # 记录决策（带异常保护）
        try:
            self._record_decision(game_state, blended_probs)
        except Exception as e:
            print(f"决策记录失败: {str(e)}")

        # 格式化输出（带单位转换保护）
        try:
            formatted_decision = self.format_decision(
                blended_probs, 
                game_state
            )
        except Exception as e:
            print(f"格式化异常: {str(e)}, 使用默认格式")
            formatted_decision = [
                "33.3% fold",
                "33.3% call", 
                "33.3% raise"
            ]

        return formatted_decision

    def _preprocess_preflop(self, raw_features):
        features = np.array(raw_features['context'] + raw_features['stage_features'])
        return features

    def _record_decision(self, game_state, final_probs):
        features = self.extract_features(game_state)
        
        # 转换为可序列化类型
        record = {
            'features': features['context'].tolist() if isinstance(features['context'], np.ndarray) else features['context'],
            'stage_features': features['stage_features'].tolist() if isinstance(features['stage_features'], np.ndarray) else features['stage_features'],
            'decision': {k: float(v) for k, v in final_probs.items()},
            'timestamp': datetime.now().isoformat()
        }
        self._save_decision_history(record)

    def _extract_bet_statistics(self, previous_actions):
        """
        根据历史下注动作提取统计信息。

        参数:
        - previous_actions: 包含玩家之前动作的列表。例如：
        [{'action': 'raise', 'bet_size': 50}, {'action': 'call', 'bet_size': 20}]

        返回:
        - 一个特征列表，例如 [aggressive_ratio, avg_bet, bet_count]
        """
        if not previous_actions:
            # 如果没有历史动作，返回默认值
            return [0.0, 0.0, 0]

        # 定义激进行为和被动行为
        aggressive_actions = ['raise']
        bet_sizes = [action.get('bet_size', 0) for action in previous_actions if 'bet_size' in action]

        # 统计数据
        total_actions = len(previous_actions)
        aggressive_count = sum(1 for action in previous_actions if action['action'] in aggressive_actions)
        avg_bet = np.mean(bet_sizes) if bet_sizes else 0.0
        bet_count = len(bet_sizes)

        # 计算激进行为比例
        aggressive_ratio = aggressive_count / total_actions if total_actions > 0 else 0.0

        return [aggressive_ratio, avg_bet, bet_count]
    
    def extract_features(self, game_state):
        """动态特征提取方法（带标准化和异常处理）"""
        try:
            # 基础特征（全部转换为BB单位）
            bb = game_state['big_blind']
            stack_bb = game_state['stack']
            pot_bb = game_state['current_pot']
            to_call_bb = game_state.get('to_call', 0.0)
            equity = game_state.get('calculated_equity', 0.5)
            position = POSITIONS.index(game_state['position'])
            
            # 标准化处理
            normalized_features = [
                equity,  # 原始胜率 0-1
                pot_bb / 200,  # 假设最大底池200BB
                stack_bb / 200,  # 假设最大筹码200BB
                to_call_bb / (pot_bb + 1e-8),  # 跟注比例
                position / (len(POSITIONS) - 1)  # 位置标准化
            ]

            # 对手动态特征（最近3轮动作）
            opp_features = []
            for action in game_state['previous_actions'][-3:]:
                opp_features.extend([
                    1 if action['action'] == 'raise' else 0,  # 激进行为标志
                    action.get('bet_size', 0) / bb  # 转换为BB单位
                ])
            # 填充不足部分
            while len(opp_features) < 6:  # 3轮动作×2特征
                opp_features.append(0.0)
            
            # 时间动态特征
            time_features = [
                len(game_state['previous_actions']) / 20,  # 动作频率
                (time.time() - game_state.get('hand_start', time.time())) / 60  # 牌局时长(分钟)
            ]

            # 合并所有特征
            full_features = normalized_features + opp_features[:6] + time_features
            
            # 特征维度校验
            if len(full_features) < NUM_FEATURES:
                full_features += [0.0] * (NUM_FEATURES - len(full_features))
            elif len(full_features) > NUM_FEATURES:
                full_features = full_features[:NUM_FEATURES]

            return {
                'context': np.array(full_features[:7], dtype=np.float32),
                'stage_features': np.array(full_features[7:], dtype=np.float32)
            }

        except Exception as e:
            print(f"特征提取异常: {str(e)}")
            # 返回安全特征
            return {
                'context': np.zeros(7),
                'stage_features': np.zeros(NUM_FEATURES-7)
            }
    
    def _get_stage_specific_features(self, game_state):
        street = game_state['street']
        if street == 'preflop':
            return self._preflop_features(game_state)
        else:
            return self._postflop_features(game_state)

    def _preflop_features(self, game_state):
        hand = game_state['hero_hand']
        return [
            Card.get_rank_int(hand[0]),
            Card.get_rank_int(hand[1]),
            int(Card.get_suit_int(hand[0]) == Card.get_suit_int(hand[1])),
            game_state['position'] in ['BTN', 'CO']
        ]

    def _postflop_features(self, game_state):
        return [
            len(game_state['community']),
            self._calculate_hand_potential(game_state),
            self._calculate_aggression_factor(game_state)
        ]

    def _calculate_hand_potential(self, game_state):
        evaluator = Evaluator()
        hand = game_state['hero_hand']
        community = game_state['community']
        current_strength = evaluator.evaluate(community, hand)
        flush_draw = self._check_flush_draw(hand, community)
        straight_draw = self._check_straight_draw(hand, community)
        potential = 0.0
        if flush_draw:
            potential += 0.35
        if straight_draw:
            potential += 0.3
        if len(community) == 3:
            potential = min(potential * 1.2, 0.9)
        elif len(community) == 4:
            potential = min(potential * 0.8, 0.7)
        return potential

    def _check_flush_draw(self, hand, community):
        suits = [Card.get_suit_int(c) for c in hand + community]
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_count = max(suit_counts.values())
        return max_count >= 4

    def _check_straight_draw(self, hand, community):
        all_cards = hand + community
        ranks = sorted([Card.get_rank_int(c) for c in all_cards])
        unique_ranks = sorted(list(set(ranks)))
        if len(unique_ranks) < 4:
            return False
        gaps = 0
        for i in range(1, len(unique_ranks)):
            diff = unique_ranks[i] - unique_ranks[i-1]
            if diff > 1:
                gaps += (diff - 1)
        return gaps <= 1 and (unique_ranks[-1] - unique_ranks[0] <= 5)

    def _calculate_aggression_factor(self, game_state):
        actions = game_state.get('previous_actions', [])
        if not actions:
            return 0.5
        aggressive_actions = ['raise', 'bet', 're-raise']
        passive_actions = ['check', 'call', 'fold']
        agg_count = sum(1 for a in actions if a in aggressive_actions)
        total_valid = sum(1 for a in actions if a in aggressive_actions + passive_actions)
        if total_valid == 0:
            return 0.5
        return agg_count / total_valid

    def _preprocess_postflop(self, raw_features, context):
        context_tensor = torch.FloatTensor(raw_features['context']).unsqueeze(0)
        stage_tensor = torch.FloatTensor(raw_features['stage_features']).unsqueeze(0)
        combined = torch.cat([context_tensor, stage_tensor], dim=1)
        return combined

    def _get_gto_baseline(self, game_state):
        street = game_state.get('street', 'preflop')
        position = game_state.get('position', 'BTN')
        if street == 'preflop':
            if position in ['BTN', 'CO']:
                return {'fold': 0.15, 'call': 0.35, 'raise': 0.5}
            else:
                return {'fold': 0.25, 'call': 0.5, 'raise': 0.25}
        elif street == 'flop':
            return {'fold': 0.20, 'call': 0.5, 'raise': 0.3}
        elif street == 'turn':
            return {'fold': 0.25, 'call': 0.6, 'raise': 0.15}
        elif street == 'river':
            return {'fold': 0.3, 'call': 0.6, 'raise': 0.1}
        else:
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}

    def _calculate_opponent_adjustment(self, opp_profile):
        if not isinstance(opp_profile, dict):
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}
        return {
            'fold': float(opp_profile.get('fold', 0.33)),
            'call': float(opp_profile.get('call', 0.34)),
            'raise': float(opp_profile.get('raise', 0.33)),
        }
    
    def online_learn(self, experience_batch):
        """增强版在线学习方法"""
        # 转换经验数据
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        
        # 特征提取
        X = [self.extract_features(s)['context'] for s in states]
        y = []
        for i in range(len(experience_batch)):
            # 计算目标Q值
            current_q = self._predict_q_value(X[i])
            target = rewards[i]
            if not dones[i]:
                next_feature = self.extract_features(next_states[i])['context']
                next_q = self._predict_q_value(next_feature)
                target += 0.9 * np.max(next_q)  # 折扣因子0.9
            
            # 更新对应动作的Q值
            action_idx = ACTION_SPACE.index(actions[i])
            current_q[action_idx] = target
            y.append(current_q)
        
        # 转换Tensor
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.FloatTensor(np.array(y))
        
        # 神经网络训练
        for stage in ['turn', 'river']:
            self.stage_models[stage].train()
            optimizer = torch.optim.AdamW(self.stage_models[stage].parameters(), lr=0.001)
            loss_fn = nn.SmoothL1Loss()  # Huber损失
            
            for _ in range(3):  # 3次迭代
                preds, _ = self.stage_models[stage](X_tensor)
                loss = loss_fn(preds, y_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.stage_models[stage].parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
        
        # 树模型更新
        self._update_tree_models(X, np.argmax(y, axis=1))

    def _update_tree_models(self, X, y):
        """树模型增量学习"""
        # XGBoost
        if hasattr(self.stage_models['preflop'], 'partial_fit'):
            self.stage_models['preflop'].partial_fit(
                X, y,
                classes=[0,1,2],
                sample_weight=np.linspace(0.5, 1.0, len(X))  # 最近样本权重更高
            )
        
        # GBDT
        self.stage_models['flop'].n_estimators += 10
        self.stage_models['flop'].fit(X, y)

    def _execute_action(self, game_state, action):
        """执行动作并返回新状态"""
        new_state = game_state.copy()
        
        # 解析动作类型
        action_type, *params = action.split()
        bet_size = int(params[0]) if params else 0
        
        # 更新游戏状态
        if action_type == 'fold':
            new_state['active_players'][self.player_id] = False
        elif action_type == 'call':
            # 跟注：用BB单位
            to_call = new_state.get('to_call', 0.0)  # BB
            new_state['stack'] -= to_call
            new_state['current_pot'] += to_call
            # 更新已投入
            new_state['previous_actions'].append({'action': 'call', 'bet_size': to_call})
        elif action_type == 'raise':
            # 加注，bet_size已为BB单位
            min_raise = new_state.get('min_raise', 2 * new_state['big_blind'])  # BB
            raise_size = max(min_raise, bet_size)  # BB
            raise_size = min(raise_size, new_state['stack'])  # 不能超出all-in
            new_state['stack'] -= raise_size
            new_state['current_pot'] += raise_size
            new_state['min_raise'] = raise_size  # 记录本轮最小加注量（BB）
            new_state['previous_actions'].append({'action': 'raise', 'bet_size': raise_size})
        
        # 推进游戏阶段
        if self._should_advance_street(new_state):
            new_state['street'] = self._next_street(new_state['street'])
            new_state['min_raise'] = new_state['big_blind']  # BB
            # 不重置current_pot，pot一直累计BB
        
        # 计算新胜率
        if new_state['street'] != game_state['street']:
            self.update_equity(new_state)
        
        return new_state
    
    def _should_advance_street(self, state):
        """判断是否进入下一阶段"""
        active_players = sum(state['active_players'])
        if active_players <= 1:
            return True
        bets_equal = all(p['current_bet'] == state['current_bet'] 
                        for p in state['players'] if p['active'])
        return bets_equal
    
    def _next_street(self, current_street):
        streets = ['preflop', 'flop', 'turn', 'river']
        current_index = streets.index(current_street)
        return streets[current_index + 1] if current_index < 3 else current_street
    
    def calculate_chip_reward(self, old_stack, new_stack, pot_contribution):
        """使用BB单位计算"""
        delta = new_stack - old_stack
        risk_factor = 1 - (pot_contribution / (old_stack + 1e-8))
        return delta * risk_factor  # 直接使用BB单位 ← 修改点
    
    def calculate_equity_reward(self, initial_equity, final_equity, street):
        """根据胜率变化计算奖励"""
        equity_gain = final_equity - initial_equity
        # 阶段权重：越后期胜率变化越重要
        stage_weights = {'preflop': 0.3, 'flop': 0.5, 'turn': 0.7, 'river': 1.0}
        return equity_gain * stage_weights.get(street, 0.5)
    
    # 使用蒙特卡洛模拟估算期望价值
    def estimate_expected_value(self, game_state):
        simulator = EnhancedPokerSimulator()
        outcomes = []
        for _ in range(200):
            outcome = simulator.simulate_hand_outcome(game_state)
            outcomes.append(outcome)
        return np.mean(outcomes)
    
    def evaluate_action_quality(self, action, game_state, expected_value, actual_result):
        """评估动作的合理性（基于GTO基准）"""
        # 获取GTO推荐动作概率
        gto_probs = self._get_gto_baseline(game_state)
        
        # 计算动作偏离度
        action_idx = ACTION_SPACE.index(action)
        deviation = 1 - gto_probs[action]  # 偏离GTO建议的程度
        
        # 价值调整：实际收益与期望值的差异
        value_diff = (actual_result - expected_value) * 100
        return value_diff - deviation * 0.5
    
    def calculate_exploitation_bonus(self, action, opp_profile):
        """根据对手类型给予额外奖励"""
        style = opp_profile.get('style', 'balanced')
        # 针对不同对手风格的奖励调整
        style_bonus = {
            'aggressive': {'fold': 0, 'call': -0.2, 'raise': 0.3},
            'passive':    {'fold': 0.2, 'call': 0, 'raise': -0.1},
            'tight':      {'fold': -0.3, 'call': 0.1, 'raise': 0.2},
            'loose':      {'fold': 0.3, 'call': -0.1, 'raise': 0}
        }.get(style, {})
        return style_bonus.get(action, 0)

    def calculate_reward(self, old_state, action, new_state):
        """完整奖励计算实现"""
        # 增加期望价值计算
        expected_value = self.estimate_expected_value(old_state)
        # 计算实际筹码变化
        actual_result = new_state['stack'] - old_state['stack']
        
        # 传递实际结果到评估函数
        action_quality = self.evaluate_action_quality(
            action=action,
            game_state=old_state,
            expected_value=expected_value,
            actual_result=actual_result  # ✅ 传递参数
        )

        # 确保胜率已计算
        if 'calculated_equity' not in old_state:
            self.update_equity(old_state)
        if 'calculated_equity' not in new_state:
            self.update_equity(new_state)

        # 获取对手画像
        opp_profile = self.opponent_model.predict(old_state)

        # 计算各奖励分量
        chip_reward = self.calculate_chip_reward(
            old_stack=old_state['stack'],
            new_stack=new_state['stack'],
            pot_contribution=old_state['current_pot']
        )
        
        equity_reward = self.calculate_equity_reward(
            initial_equity=old_state['calculated_equity'],
            final_equity=new_state['calculated_equity'],
            street=old_state['street']
        )

        # 需要实现期望价值估算
        expected_value = self.estimate_expected_value(old_state)
        
        action_quality = self.evaluate_action_quality(
            action=action,
            game_state=old_state,
            expected_value=expected_value
        )

        exploit_bonus = self.calculate_exploitation_bonus(
            action=action,
            opp_profile=opp_profile
        )

        # 综合奖励
        total_reward = (
            chip_reward * 0.6 +
            equity_reward * 0.25 +
            action_quality * 0.1 +
            exploit_bonus * 0.05
        )

        # 终局奖励放大
        if not new_state['active_players'][self.player_id]:
            total_reward *= 2.5 if new_state['stack'] > old_state['stack'] else -1.5

        return total_reward
    
    def _predict_q_value(self, features):
        """预测各动作Q值"""
        stage = self.current_stage
        if stage in ['turn', 'river']:
            with torch.no_grad():
                tensor_features = torch.FloatTensor(features).unsqueeze(0)
                stage_idx = torch.LongTensor([[STAGES.index(stage)]])
                q_values, _ = self.stage_models[stage](tensor_features, stage_idx)
                return q_values.numpy()[0]
        else:
            return self.stage_models[stage].predict_proba([features])[0]
        

    def update_equity(self, game_state, force_update=False):
        """
        动态更新游戏状态的胜率（带缓存和增量更新）
        :param game_state: 当前游戏状态字典（会被修改）
        :param force_update: 是否强制重新计算（默认使用缓存）
        """
        # 生成缓存键
        cache_key = self._generate_equity_cache_key(game_state)

        # 加锁保证线程安全
        with self.cache_lock:
            if not force_update and cache_key in self.equity_cache:
                self.cache_hits += 1
                game_state['calculated_equity'] = self.equity_cache[cache_key]
                return
            
            self.cache_misses += 1
        
        # 检查缓存是否有效
        if not force_update and cache_key in self.equity_cache:
            game_state['calculated_equity'] = self.equity_cache[cache_key]
            return
        
        # 获取必要参数
        hero_hand = game_state['hero_hand']
        community = game_state['community']
        street = game_state['street']
        
        # 根据阶段调整迭代次数（性能优化）
        iteration_map = {
            'preflop': 500,   # 翻前精确度要求低
            'flop': 1000,
            'turn': 2000,
            'river': 5000     # 河牌需要高精度
        }
        iterations = iteration_map.get(street, 1000)
        
        # 调用模拟器计算胜率
        simulator = EnhancedPokerSimulator()
        equity, _ = simulator.calculate_equity(
            hero_hand=hero_hand,
            community=community,
            iterations=iterations,
            street=street
        )
        
        # 更新状态并缓存
        game_state['calculated_equity'] = equity
        self.equity_cache[cache_key] = equity
        
        # 缓存清理（LRU机制）
        if len(self.equity_cache) > 1000:
            oldest_key = next(iter(self.equity_cache))
            del self.equity_cache[oldest_key]

        # 更新缓存（再次加锁）
        with self.cache_lock:
            self.equity_cache[cache_key] = equity
            if len(self.equity_cache) > 1000:
                # LRU淘汰策略
                oldest_key = next(iter(self.equity_cache))
                del self.equity_cache[oldest_key]

    def _generate_equity_cache_key(self, game_state):
        """生成唯一的胜率缓存键"""
        # 获取牌面信息
        hero_cards = tuple(sorted(Card.int_to_str(c) for c in game_state['hero_hand']))
        community_cards = tuple(sorted(Card.int_to_str(c) for c in game_state['community']))
        
        # 生成哈希键
        hash_str = f"{hero_cards}|{community_cards}|{game_state['street']}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    @property
    def cache_status(self):
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses + 1e-8),
            'size': len(self.equity_cache)
        }


#-----------------------------------------------------------------------------------------------

# ====================== 新增奖励归一化类 ======================
class RewardNormalizer:
    def __init__(self, buffer_size=10000, epsilon=1e-8):
        """
        动态奖励归一化器
        :param buffer_size: 统计窗口大小（默认保留最近1万条奖励）
        :param epsilon: 防止除零的小量
        """
        self.buffer = deque(maxlen=buffer_size)
        self.epsilon = epsilon
        self.mean = 0.0
        self.std = 1.0
        
    def update_stats(self):
        """更新统计量（均值/标准差）"""
        if len(self.buffer) == 0:
            return
        
        # 计算指数移动平均（EMA）更关注近期数据
        alpha = 0.1  # 新旧数据权重
        new_mean = np.mean(self.buffer)
        new_std = np.std(self.buffer)
        
        self.mean = alpha * new_mean + (1 - alpha) * self.mean
        self.std = alpha * new_std + (1 - alpha) * self.std
        
    def normalize(self, reward):
        """
        归一化奖励值
        :param reward: 原始奖励值
        :return: 标准化后的奖励值（Z-Score）
        """
        self.buffer.append(reward)
        self.update_stats()
        
        # 防止初始阶段标准差为零
        safe_std = self.std if self.std > self.epsilon else 1.0
        return (reward - self.mean) / safe_std
    
    def denormalize(self, normalized_reward):
        """反归一化（用于调试）"""
        return normalized_reward * self.std + self.mean



# ====================== 演示主流程 ======================
def simulate_game_state():
    """完全使用BB单位的游戏状态"""
    return {
        'player_id': 5,
        'hero_hand': [create_card('Ah'), create_card('Kd')],
        'community': [create_card('Qs'), create_card('Jh'), create_card('Tc')],
        'street': 'flop',
        'position': 'BTN',
        'current_pot': 5.0,       # BB单位
        'previous_actions': [
            {'action': 'raise', 'bet_size': 0.83},  # BB单位
            {'action': 'call', 'bet_size': 0.33}    # BB单位
        ],
        'stack': 16.67,          # BB单位
        'to_call': 0.33,         # BB单位
        'big_blind': BB,         # 固定1BB
        'small_blind': SB        # 固定0.5BB
    }

if __name__ == "__main__":
    poker_ai = HybridPokerAI()
    game_state = simulate_game_state()
    decision = poker_ai.decide_action(game_state)
    print("当前牌局状态：")
    print(f"手牌: {[Card.int_to_str(c) for c in game_state['hero_hand']]}")
    print(f"公共牌: {[Card.int_to_str(c) for c in game_state['community']]}")
    print(f"位置: {game_state['position']}")
    print("\n推荐决策：")
    for action in decision:
        print(f"- {action}")