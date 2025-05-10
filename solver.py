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
    def __init__(self):
        self.cluster_model = self._init_clustering()
        self.action_sequences = {}
        dummy = np.zeros((5, 3))
        self.cluster_model.partial_fit(dummy)
    def update(self, game_state):
        player_id = game_state['player_id']
        if player_id not in self.action_sequences:
            self.action_sequences[player_id] = deque(maxlen=100)
        self.action_sequences[player_id].append({
            'action': game_state['action'],
            'timing': game_state['timing'],
            'bet_size': game_state['bet_size']
        })
        features = self._extract_behavior_features(player_id)
        self.cluster_model.partial_fit([features])
    def predict(self, game_state):
        features = self._extract_behavior_features(game_state['player_id'])
        cluster = self.cluster_model.predict([features])[0]
        return self._get_cluster_profile(cluster)
    
    def _init_clustering(self):
        return MiniBatchKMeans(n_clusters=5, random_state=42)
    
    def _extract_behavior_features(self, player_id):
        """
        修正后的行为特征提取方法
        现在只使用raise作为激进行为标识
        """
        seq = self.action_sequences.get(player_id, [])
        if not seq:
            return [0.5, 0.0, 0]  # 默认激进率0.5, 平均下注0BB, 下注次数0

        actions = [a['action'] for a in seq]
        # bet_sizes字段全为BB单位
        bet_sizes = [a.get('bet_size', 0) for a in seq]

        # 关键修改：只识别raise作为激进行为
        aggressive_actions = ['raise']
        aggressive_count = sum(1 for a in actions if a in aggressive_actions)
        total_count = len(actions)
        avg_bet_bb = np.mean(bet_sizes) if bet_sizes else 0.0
        aggressive_ratio = aggressive_count / total_count if total_count else 0.0

        # 返回全部为BB单位的特征
        return [aggressive_ratio, avg_bet_bb, total_count]
    
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

    def _train_meta_learner(self):
        """
        训练元学习器
        所有涉及金额的特征（如stack, pot, bet_size等）均为BB单位
        """
        historical_data = self._load_decision_history()
        if not historical_data:
            print("无有效历史记录，跳过元学习器训练")
            return

        X = []
        y = []
        for d in historical_data:
            # 兼容新版和旧版特征结构，所有金额相关字段都以BB单位存储
            features = d.get('features', [])
            if isinstance(features, dict):
                # context和stage_features都应为BB单位
                features = features.get('context', []) + features.get('stage_features', [])
            
            # 假如历史数据有chip为单位的，需要这里转换为BB单位
            # 例如：features[i] = features[i] / BB_chip_value  # 视具体特征定义而定
            # 若已统一为BB则无需处理

            if len(features) >= NUM_FEATURES:
                X.append(features[:NUM_FEATURES])  # 截断多余特征，全部为BB
                weights = d.get('weights', [0.33, 0.34, 0.33])
                y.append(weights)

        if len(X) < 10:
            print(f"有效样本不足 ({len(X)})，跳过训练")
            return

        X = np.array(X)
        y = np.array(y)

        # 明确所有X中金额特征为BB单位（如stack、pot、bet_size等）
        self.meta_learner.fit(X, y)
        print("元学习器训练完成，特征维度:", X.shape, "（金额全部为BB单位）")

    def _load_decision_history(self):
        historical_data = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    raw_data = json.load(f)
                    # 过滤无效记录
                    historical_data = [
                        d for d in raw_data 
                        if 'features' in d and 'decision' in d
                    ]
                    # 旧数据迁移（兼容旧版本）
                    for d in raw_data:
                        if 'feature' in d:  # 处理旧键名
                            d['features'] = d.pop('feature')
            except Exception as e:
                print(f"加载历史文件失败: {e}")
        return historical_data
        
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
        """融合 ML 预测、GTO 策略和对手调整策略"""
        # 获取 GTO 基线概率
        gto_probs = self._get_gto_baseline(game_state)

        # 获取对手调整概率
        opp_adjustment = self._calculate_opponent_adjustment(opp_profile)

        # 默认权重
        weights = [0.6, 0.3, 0.1]  # 机器学习、GTO、对手调整的权重

        blended = {}
        for idx, action in enumerate(ACTION_SPACE):
            blended[action] = (
                weights[0] * ml_probs[idx] +
                weights[1] * gto_probs[action] +
                weights[2] * opp_adjustment[action]
            )
        return blended
    
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
        """最终版决策格式化方法"""
        total = sum(blended_probs.values()) or 1
        actions = []
        
        # 获取加注量（BB单位）
        bb = game_state['big_blind']
        min_raise = max(2 * bb, 1.0)
        max_raise = game_state['stack']
        equity = game_state.get('calculated_equity', 0.5)
        base_raise = max(
            min_raise,
            game_state['current_pot'] * (0.4 + 0.3 * equity)
        )
        raise_bb = round(min(base_raise, max_raise), 2)

        for action in ACTION_SPACE:
            prob = blended_probs.get(action, 0.0) / total
            percent = int(prob * 100)
            
            # 特殊处理call/check
            if action == 'call':
                to_call_bb = game_state.get('to_call', 0.0)
                
                if to_call_bb == 0:  # check情况
                    display_action = 'check'
                    actions.append(f"{percent}% {display_action}")
                else:  # 实际跟注
                    # 格式化显示去零
                    call_amount = f"{to_call_bb:.2f}BB".replace(".00BB", "BB")
                    actions.append(f"{percent}% call {call_amount}")
                    
            elif action == 'raise':
                # 格式化加注量显示
                raise_str = f"{raise_bb:.2f}BB".replace(".00BB", "BB")
                actions.append(f"{percent}% {action} {raise_str}")
            else:
                actions.append(f"{percent}% {action}")

        return actions

    def _init_pytorch_model(self, stage):
        model = StageLSTM(
            input_size=32 if stage == 'turn' else 48,
            hidden_size=64,
            num_classes=3
        )
        model_path = f'{self.model_path}{stage}_model.pth'
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path))
            except Exception as e:
                print(f"加载{stage}模型失败: {e}，使用初始权重")
        else:
            print(f"未找到{stage}模型文件，使用初始权重")
        return model

    def _load_model_weights(self):
        preflop_path = f'{self.model_path}preflop_model.joblib'
        flop_path = f'{self.model_path}flop_model.joblib'
        if os.path.exists(preflop_path):
            try:
                self.stage_models['preflop'] = load(preflop_path)
            except Exception as e:
                print(f"加载preflop模型失败: {e}，使用新实例")
        if os.path.exists(flop_path):
            try:
                self.stage_models['flop'] = load(flop_path)
            except Exception as e:
                print(f"加载flop模型失败: {e}，使用新实例并fit")
                dummy_X = np.random.rand(100, 15)
                dummy_y = np.random.randint(0, 3, 100)
                self.stage_models['flop'].fit(dummy_X, dummy_y)
        else:
            dummy_X = np.random.rand(100, 15)
            dummy_y = np.random.randint(0, 3, 100)
            self.stage_models['flop'].fit(dummy_X, dummy_y)

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
        # 短码全押特殊处理（优先判断）
        if game_state['stack'] <= 2.0:  # ≤2BB时强制全押
            return ['0% fold', '0% check', '100% all-in'] 
        
        if 'big_blind' not in game_state:
            game_state['big_blind'] = BB  # 默认 1BB
        
        raw_features = self.extract_features(game_state)

        # 提取阶段特异性特征
        stage = game_state['street']
        if stage == 'preflop':
            features = self._preprocess_preflop(raw_features)
        else:
            context = self.context_model(torch.Tensor(raw_features['context']))
            features = self._preprocess_postflop(raw_features, context)

        # 确保特征维度符合模型要求
        features = np.array(list(features))
        expected_features = NUM_FEATURES  # 模型期望的特征数量
        if features.ndim > 2:
            features = features.squeeze()  # 去掉多余维度
        if features.shape[0] < expected_features:
            # 如果特征数量不足，填充 0
            features = np.pad(features, (0, expected_features - features.shape[0]), mode='constant')
        elif features.shape[0] > expected_features:
            # 如果特征数量过多，截断
            features = features[:expected_features]

        # 修改特征处理部分
        if stage in ['turn', 'river']:
            with torch.no_grad():
                logits, _ = self.stage_models[stage](
                    torch.tensor(features).unsqueeze(0), 
                    torch.tensor([[STAGES.index(stage)]])
                )
                probs = torch.softmax(logits, dim=-1).numpy()
        else:
            # 新增维度修正
            if features.ndim == 3:
                features = features.squeeze(0)  # 将三维降为二维
            elif features.ndim == 1:
                features = features.reshape(1, -1)  # 确保二维
                
            # 添加特征校验
            if features.shape[1] != NUM_FEATURES:
                features = features[:, :NUM_FEATURES]  # 截断多余特征
                
            probs = self.stage_models[stage].predict_proba(features)[0]  # 移除多余列表包装

        opp_profile = self.opponent_model.predict(game_state)
        final_probs = self._blend_strategies(probs, game_state, opp_profile)
        self._record_decision(game_state, final_probs)
        self._record_decision(game_state, final_probs)
        print("决策记录已生成")  # 调试日志
        return self.format_decision(final_probs, game_state)

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
        equity, _ = EnhancedPokerSimulator().calculate_equity(
            game_state['hero_hand'],
            game_state['community'],
            iterations=200,
            street=game_state['street']
        )

        # 提取 context 特征
        # 修改后的context特征（移除错误的比例缩放）
        context = [
            equity,
            len(game_state['community']),
            game_state['current_pot'],  # 直接使用BB单位 ← 修改点
            int(game_state['position'] in ['BTN', 'CO']),
            len(game_state.get('previous_actions', [])),
            game_state.get('to_call', 0) / (game_state['current_pot'] + 1e-8),  # 用BB单位计算比例 ← 修改点
            game_state.get('stack', 100)  # 直接使用BB单位 ← 修改点
        ]

        # 填充或裁剪 context 特征到固定长度（如 7）
        context_length = 7
        if len(context) < context_length:
            context = context + [0.0] * (context_length - len(context))  # 填充 0
        elif len(context) > context_length:
            context = context[:context_length]  # 截断

        stage_features = [
            *self._get_stage_specific_features(game_state),
        ]

        # 修改后处理逻辑
        return {
            'context': np.array(context).flatten(),  # 确保一维化
            'stage_features': np.array(stage_features).flatten()
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
    
    #在线学习部分
    def online_learn(self, experience_batch):
        """批量在线学习实现"""
        # 转换经验格式
        states, actions, rewards, next_states, dones = zip(*experience_batch)
        
        # 转换为模型输入格式
        X = []
        y = []
        for i in range(len(states)):
            # 提取特征
            state_features = self.extract_features(states[i])['context']
            next_features = self.extract_features(next_states[i])['context']
            
            # 当前Q值
            current_q = self._predict_q_value(state_features)
            
            # 目标Q值
            target = rewards[i]
            if not dones[i]:
                next_q = self._predict_q_value(next_features)
                target += self.gamma * np.max(next_q)
            
            # 更新动作对应的Q值
            action_idx = ACTION_SPACE.index(actions[i])
            current_q[action_idx] = target
            
            X.append(state_features)
            y.append(current_q)
        
        # 转换为Tensor
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.FloatTensor(np.array(y))
        
        # 训练神经网络
        for stage in ['turn', 'river']:
            self.stage_models[stage].train()
            optimizer = torch.optim.Adam(self.stage_models[stage].parameters())
            
            # 训练循环
            for _ in range(3):  # 小批量多次更新
                predictions, _ = self.stage_models[stage](X_tensor)
                loss = nn.MSELoss()(predictions, y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 更新树模型
        self._update_tree_models(X, y)

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
        
    def _update_tree_models(self, X, y):
        """更新XGBoost/GBDT模型"""
        # 转换标签
        y_labels = np.argmax(y, axis=1)
        
        # 更新preflop模型
        if hasattr(self.stage_models['preflop'], 'partial_fit'):
            self.stage_models['preflop'].partial_fit(X, y_labels, classes=[0,1,2])
        
        # 更新flop模型
        self.stage_models['flop'].fit(X, y_labels)

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