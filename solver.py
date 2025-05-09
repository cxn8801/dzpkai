from treys import Deck, Evaluator, Card
import random
from itertools import combinations
import time
from functools import lru_cache
from multiprocessing import Pool
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import MiniBatchKMeans
from xgboost import XGBClassifier
from joblib import dump, load
from datetime import datetime
from collections import deque
import hashlib
import random
from functools import lru_cache
from itertools import combinations
import multiprocessing as mp
import os
from sklearn.exceptions import NotFittedError

# ======================
# 核心优化模块
# ======================

NUM_FEATURES = 15  # 统一特征维度，模型训练和推理都用这个

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
        full_deck = set(Deck.GetFullDeck())
        return list(full_deck - set(known_cards))
    
    def _simulate_opponents(self, remaining_deck, num_opponents):
        if self.use_range:
            valid_hands = [hand for hand in self.top_range 
                          if set(hand).issubset(remaining_deck)]
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
            if len(community_full) !=5:
                continue
                
            try:
                comm_key = tuple(sorted(community_full))
                hero_key = tuple(sorted(hero_hand))
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
    STRATEGY_TABLE = {
        ('BTN', (0.7, 1.0)): {'raise': 65, 'call': 25, 'fold': 10},
        ('BTN', (0.5, 0.7)): {'raise': 40, 'call': 50, 'fold': 10},
        ('UTG', (0.7, 1.0)): {'raise': 50, 'call': 30, 'fold': 20},
    }
    
    @classmethod
    def get_strategy(cls, equity, position):
        for (pos_range, (min_eq, max_eq)), weights in cls.STRATEGY_TABLE.items():
            if position in pos_range and min_eq <= equity < max_eq:
                return weights
        return {'call': 50, 'fold': 50}

# ======================
# 修复后的机器学习模块
# ======================
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
        """创建保证可用的应急模型"""
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
    
# ======================
# 核心数据结构和常量
# ======================
STAGES = ['preflop', 'flop', 'turn', 'river']
POSITIONS = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
ACTION_SPACE = ['fold', 'call', 'raise']

# ======================
# 分层神经网络模型
# ======================
class StageLSTM(nn.Module):
    """维度修正后的分阶段LSTM"""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.stage_emb = nn.Embedding(4, 4)
        # 调整适配层输入维度为实际特征数
        self.linear_adapter = nn.Linear(11, 32)  # 11个总特征维度
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, stage_idx, hidden=None):
        stage_emb = self.stage_emb(stage_idx)
        x = torch.cat([x, stage_emb], dim=-1)
        x = self.linear_adapter(x)  # 新增适配步骤
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1]), hidden
    
class GlobalContextNet(nn.Module):
    """调整后的上下文网络"""
    def __init__(self):
        super().__init__()
        # 调整维度匹配主网络
        self.encoder = nn.Sequential(
            nn.Linear(7, 16),  # 输入层匹配实际特征维度
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
# ======================
# 混合决策系统
# ======================
class HybridPokerAI:
    def __init__(self):
        self.model_path = 'models/'
        os.makedirs(self.model_path, exist_ok=True)
        self.decision_history = []  # 别忘了初始化决策历史记录

        # 初始化各阶段模型
        self.stage_models = {
            'preflop': XGBClassifier(n_estimators=100),
            'flop': GradientBoostingClassifier(),
            'turn': self._init_pytorch_model('turn'),
            'river': self._init_pytorch_model('river')
        }
        # 检查并创建初始模型
        required_files = [
            f'{self.model_path}turn_model.pth',
            f'{self.model_path}river_model.pth',
            f'{self.model_path}preflop_model.joblib'
        ]
        if not all(os.path.exists(f) for f in required_files):
            print("检测到缺失模型文件，正在初始化基础模型...")
            self._create_initial_models()

        self.context_model = GlobalContextNet()
        self.opponent_model = OpponentProfiler()
        self.memory = ExperienceReplayBuffer(10000)
        self.online_learner = OnlineAdaptor()
        self._load_model_weights()

    def _init_pytorch_model(self, stage):
        """初始化PyTorch模型（带自动修复）"""
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
            # 如果没有flop模型文件，fit一个dummy的
            dummy_X = np.random.rand(100, 15)
            dummy_y = np.random.randint(0, 3, 100)
            self.stage_models['flop'].fit(dummy_X, dummy_y)
    
    def _create_initial_models(self):
        print("正在生成初始模型文件...")

        # 创建PyTorch模型
        turn_model = StageLSTM(32, 64, 3)
        torch.save(turn_model.state_dict(), f'{self.model_path}turn_model.pth')
        river_model = StageLSTM(48, 64, 3)
        torch.save(river_model.state_dict(), f'{self.model_path}river_model.pth')

        # 创建XGBoost模型
        dummy_X = np.random.rand(100, 15)
        dummy_y = np.random.randint(0, 3, 100)
        self.stage_models['preflop'].fit(dummy_X, dummy_y)
        dump(self.stage_models['preflop'], f'{self.model_path}preflop_model.joblib')

        # 创建GradientBoostingClassifier模型
        self.stage_models['flop'].fit(dummy_X, dummy_y)
        dump(self.stage_models['flop'], f'{self.model_path}flop_model.joblib')

        print(f"基础模型已保存至：{os.path.abspath(self.model_path)}")

    def load_models(self, ai, path='models/'):
        # 加载XGBoost模型
        self.stage_models['preflop'] = load(f'{path}preflop_model.joblib')
        
        # 加载PyTorch模型
        self.stage_models['turn'].load_state_dict(torch.load(f'{path}turn_model.pth'))
        self.stage_models['river'].load_state_dict(torch.load(f'{path}river_model.pth'))
        
        # 加载上下文模型
        self.context_model.load_state_dict(torch.load(f'{path}context_model.pth'))

    def decide_action(self, game_state):
        # 特征工程
        raw_features = self.extract_features(game_state)
        context = self.context_model(torch.Tensor(raw_features['context']))

        # 对手建模
        opp_profile = self.opponent_model.predict(game_state)
        stage = game_state['street']

        if stage == 'preflop':
            features = self._preprocess_preflop(raw_features)
            features = np.array(features)
            # 补齐特征
            if features.shape[0] < NUM_FEATURES:
                features = np.pad(features, (0, NUM_FEATURES - features.shape[0]), mode='constant')
            try:
                probs = self.stage_models[stage].predict_proba([features])[0]
                probs = np.array(probs).flatten()
                assert probs.shape[0] == 3
            except NotFittedError:
                dummy_X = np.random.rand(10, NUM_FEATURES)
                dummy_y = np.random.randint(0, 3, 10)
                self.stage_models[stage].fit(dummy_X, dummy_y)
                probs = self.stage_models[stage].predict_proba([features])[0]
                probs = np.array(probs).flatten()
                assert probs.shape[0] == 3
        else:
            features = self._preprocess_postflop(raw_features, context)
            # features 是 torch tensor [1, n]
            if features.shape[1] < NUM_FEATURES:
                pad_size = NUM_FEATURES - features.shape[1]
                features = torch.cat([features, torch.zeros((features.shape[0], pad_size))], dim=1)
            np_features = features.numpy()
            if stage in ['turn', 'river']:
                with torch.no_grad():
                    logits, _ = self.stage_models[stage](features)
                    probs = torch.softmax(logits, dim=-1).numpy()
            else:
                try:
                    probs = self.stage_models[stage].predict_proba(np_features)
                    probs = np.array(probs).flatten()
                    assert probs.shape[0] == 3
                except NotFittedError:
                    dummy_X = np.random.rand(10, NUM_FEATURES)
                    dummy_y = np.random.randint(0, 3, 10)
                    self.stage_models[stage].fit(dummy_X, dummy_y)
                    probs = self.stage_models[stage].predict_proba(np_features)
                    probs = np.array(probs).flatten()
                    assert probs.shape[0] == 3

        # 混合策略
        final_probs = self._blend_strategies(
            probs,
            game_state,
            opp_profile
        )
        # 记录决策
        self._record_decision(game_state, final_probs)
        return self.format_decision(final_probs, game_state)
    
    def format_decision(self, blended_probs, game_state):
        """
        格式化最终决策输出，确保概率归一化并输出为易读格式。
        """
        total = sum(blended_probs.values()) or 1
        actions = []
        for action in ACTION_SPACE:
            prob = blended_probs.get(action, 0.0) / total
            percent = int(prob * 100)
            if action in ['raise', 'bet']:
                size = int(game_state['current_pot'] * 0.6)
                actions.append(f"{percent}% {action} {size}BB")
            else:
                actions.append(f"{percent}% {action}")
        return actions
    
    def _preprocess_preflop(self, raw_features):
        # 假设 preflop 特征就是 context + stage_features
        features = np.array(raw_features['context'] + raw_features['stage_features'])
        return features

    def _record_decision(self, game_state, final_probs):
        """记录每一次的决策到历史列表"""
        record = {
            'game_state': game_state.copy(),  # 建议深拷贝，防止外部引用被改动
            'decision': final_probs
        }
        self.decision_history.append(record)
    
    def extract_features(self, game_state):
        """修正后的特征工程"""
        equity, _ = EnhancedPokerSimulator().calculate_equity(
            game_state['hero_hand'],
            game_state['community'],
            iterations=200,
            street=game_state['street']
        )
        
        # 确保总特征数为11
        return {
            'context': [
                equity,  # 1
                len(game_state['community']),  # 2
                game_state['current_pot'] / 100,  # 3
                int(game_state['position'] in ['BTN', 'CO']),  # 4
                len(game_state.get('previous_actions', [])),  #5
                game_state.get('to_call', 0) / game_state['current_pot'] if game_state['current_pot'] > 0 else 0,  #6
                game_state.get('stack', 1000) / 1000  #7
            ],
            'stage_features': [
                *self._get_stage_specific_features(game_state),  # +4
            ]
        }
    
    def _get_stage_specific_features(self, game_state):
        """分阶段特征"""
        street = game_state['street']
        if street == 'preflop':
            return self._preflop_features(game_state)
        else:
            return self._postflop_features(game_state)
        
    def _preflop_features(self, game_state):
        """翻牌前特征"""
        hand = game_state['hero_hand']
        return [
            Card.get_rank_int(hand[0]),
            Card.get_rank_int(hand[1]),
            int(Card.get_suit_int(hand[0]) == Card.get_suit_int(hand[1])),
            game_state['position'] in ['BTN', 'CO']
        ]
    
    def _postflop_features(self, game_state):
        """翻牌后特征"""
        return [
            len(game_state['community']),
            self._calculate_hand_potential(game_state),
            self._calculate_aggression_factor(game_state)
        ]

    def _calculate_hand_potential(self, game_state):
        """计算手牌发展潜力（听牌概率）"""
        evaluator = Evaluator()
        hand = game_state['hero_hand']
        community = game_state['community']
        
        # 基本牌力
        current_strength = evaluator.evaluate(community, hand)
        
        # 听牌检测
        flush_draw = self._check_flush_draw(hand, community)
        straight_draw = self._check_straight_draw(hand, community)
        
        # 潜力评分（0-1范围）
        potential = 0.0
        if flush_draw:
            potential += 0.35
        if straight_draw:
            potential += 0.3
        if len(community) == 3:  # 翻牌圈
            potential = min(potential * 1.2, 0.9)
        elif len(community) == 4:  # 转牌圈
            potential = min(potential * 0.8, 0.7)
            
        return potential

    def _check_flush_draw(self, hand, community):
        """检测同花听牌"""
        suits = [Card.get_suit_int(c) for c in hand + community]
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_count = max(suit_counts.values())
        return max_count >= 4  # 至少4张同花

    def _check_straight_draw(self, hand, community):
        """检测顺子听牌"""
        all_cards = hand + community
        ranks = sorted([Card.get_rank_int(c) for c in all_cards])
        
        # 去重并排序
        unique_ranks = sorted(list(set(ranks)))
        if len(unique_ranks) < 4:
            return False
            
        # 检测间隔顺子
        gaps = 0
        for i in range(1, len(unique_ranks)):
            diff = unique_ranks[i] - unique_ranks[i-1]
            if diff > 1:
                gaps += (diff - 1)
                
        return gaps <= 1 and (unique_ranks[-1] - unique_ranks[0] <= 5)

    def _calculate_aggression_factor(self, game_state):
        """计算攻击性系数（基于历史动作）"""
        actions = game_state.get('previous_actions', [])
        if not actions:
            return 0.5  # 默认中性值
            
        aggressive_actions = ['raise', 'bet', 're-raise']
        passive_actions = ['check', 'call', 'fold']
        
        agg_count = sum(1 for a in actions if a in aggressive_actions)
        total_valid = sum(1 for a in actions if a in aggressive_actions + passive_actions)
        
        if total_valid == 0:
            return 0.5
            
        return agg_count / total_valid  # 返回0-1之间的系数

    def _preprocess_postflop(self, raw_features, context):
        # 合并特征确保总维度为11
        # 先将一维变成二维，再拼接
        context_tensor = torch.FloatTensor(raw_features['context']).unsqueeze(0)
        stage_tensor = torch.FloatTensor(raw_features['stage_features']).unsqueeze(0)
        combined = torch.cat([context_tensor, stage_tensor], dim=1)
        return combined  # shape: [1, 11]
    
    def _get_additional_features(self, features):
        """补充特征维度"""
        return [
            features['context'][2],  # 底池大小
            features['context'][4],  # 历史动作数量
            features['stage_features'][-1]  # 攻击系数
        ]

    def _get_gto_baseline(self, game_state):
        """
        返回一个基础的GTO策略分布（fold/call/raise 概率），可按实际需求丰富
        """
        street = game_state.get('street', 'preflop')
        position = game_state.get('position', 'BTN')
        # 这里用简单的规则做示例，你可以按需要调整
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
            # 默认均匀策略
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}

    def _calculate_opponent_adjustment(self, opp_profile):
        """
        根据对手画像调整策略权重。
        opp_profile: 应该是一个包含 fold/call/raise 概率（或权重）的 dict
        """
        # 防御性判断
        if not isinstance(opp_profile, dict):
            return {'fold': 0.33, 'call': 0.34, 'raise': 0.33}
        # 从对手画像提取加权信息
        return {
            'fold': float(opp_profile.get('fold', 0.33)),
            'call': float(opp_profile.get('call', 0.34)),
            'raise': float(opp_profile.get('raise', 0.33)),
        }

    def _blend_strategies(self, ml_probs, game_state, opp_profile):
        # 兼容dict和list/array
        if isinstance(ml_probs, dict):
            ml_probs = [ml_probs.get(a, 0.0) for a in ACTION_SPACE]
        ml_probs = np.array(ml_probs).flatten()
        if ml_probs.shape[0] != 3:
            print(f"警告: ml_probs shape={ml_probs.shape}, 自动补齐/截断到3项")
            ml_probs = np.pad(ml_probs, (0, 3-ml_probs.shape[0]), mode='constant')[:3]

        gto_probs = self._get_gto_baseline(game_state)
        opp_adjustment = self._calculate_opponent_adjustment(opp_profile)

        blended = {}
        for idx, action in enumerate(ACTION_SPACE):
            blended[action] = (
                0.6 * ml_probs[idx] +
                0.3 * gto_probs[action] +
                0.1 * opp_adjustment[action]
            )
        return blended
    
    def online_learn(self, experience):
        """在线学习接口"""
        self.memory.store(experience)
        batch = self.memory.sample(512)
        self.online_learner.update(batch)
        self._update_models()


# ======================
# 对手建模系统
# ======================
class OpponentProfiler:
    def __init__(self):
        self.cluster_model = self._init_clustering()
        self.action_sequences = {}
        # 预先fit，防止未fit报错
        dummy = np.zeros((5, 3))  # 5个样本，每个3维
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
        简单行为特征抽取方法示例：
        返回 [激进行为比例, 平均下注额, 行为数等]
        """
        seq = self.action_sequences.get(player_id, [])
        if not seq:
            # 没有行为，用默认值
            return [0.5, 0, 0]
        actions = [a['action'] for a in seq]
        bet_sizes = [a['bet_size'] for a in seq if 'bet_size' in a]
        aggressive_count = sum(1 for a in actions if a in ['raise', 'bet'])
        total_count = len(actions)
        avg_bet = np.mean(bet_sizes) if bet_sizes else 0
        aggressive_ratio = aggressive_count / total_count if total_count else 0
        return [aggressive_ratio, avg_bet, total_count]
    
    def _get_cluster_profile(self, cluster):
        """
        根据聚类编号返回对手风格画像或加权信息。
        cluster: int, 聚类类别编号
        返回: dict, 可用于决策融合的调整参数
        """
        # 示例：根据cluster映射到不同风格
        # 你可以返回不同的加权因子、描述、或策略调整参数
        profiles = {
            0: {"style": "aggressive", "fold": 0.2, "call": 0.3, "raise": 0.5},
            1: {"style": "passive",    "fold": 0.4, "call": 0.5, "raise": 0.1},
            2: {"style": "loose",      "fold": 0.1, "call": 0.6, "raise": 0.3},
            3: {"style": "tight",      "fold": 0.6, "call": 0.3, "raise": 0.1},
            4: {"style": "balanced",   "fold": 0.3, "call": 0.4, "raise": 0.3},
        }
        # 若聚类号超出范围，返回默认
        return profiles.get(cluster, {"style": "unknown", "fold": 0.33, "call": 0.34, "raise": 0.33})
    


class ExperienceReplayBuffer:
    """经验回放缓冲池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

class OnlineAdaptor:
    """在线学习适配器"""
    def __init__(self):
        self.model = self._init_online_model()
    
    def _init_online_model(self):
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss='log_loss')
    
    def update(self, batch):
        X, y = zip(*batch)
        self.model.partial_fit(X, y, classes=[0,1,2])

# ======================
# 模型保存和加载
# ======================
def save_models(ai, path='models/'):
    # 保存XGBoost模型
    dump(ai.stage_models['preflop'], f'{path}preflop_model.joblib')
    
    # 保存PyTorch模型
    torch.save(ai.stage_models['turn'].state_dict(), f'{path}turn_model.pth')
    torch.save(ai.stage_models['river'].state_dict(), f'{path}river_model.pth')
    
    # 保存上下文模型
    torch.save(ai.context_model.state_dict(), f'{path}context_model.pth')

def quick_train():
    """快速训练模式（生成基础模型）"""
    from sklearn.datasets import make_classification
    
    # 生成示例数据
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_classes=3,
        random_state=42
    )
    
    # 训练preflop模型
    preflop_model = XGBClassifier(n_estimators=100)
    preflop_model.fit(X, y)
    dump(preflop_model, 'models/preflop_model.joblib')
    
    # 初始化神经网络
    turn_model = StageLSTM(32, 64, 3)
    torch.save(turn_model.state_dict(), 'models/turn_model.pth')
    
    river_model = StageLSTM(48, 64, 3)
    torch.save(river_model.state_dict(), 'models/river_model.pth')
    
    print("基础模型已生成")


# ======================
# 示例用法
# ======================

def create_card(card_str):
    """将字符串转换为treys Card对象"""
    return Card.new(card_str)

def simulate_game_state():
    """模拟游戏状态"""
    return {
        'hero_hand': [create_card('Ah'), create_card('Kd')],  # 手牌
        'community': [create_card('Qs'), create_card('Jh'), create_card('Tc')],  # 公共牌
        'street': 'flop',        # 当前阶段：preflop/flop/turn/river
        'position': 'BTN',       # 座位位置
        'current_pot': 150,      # 当前底池大小
        'previous_actions': ['check', 'call'],  # 本回合之前的动作
        'player_id': 1,          # 玩家ID（用于对手建模）
        'stack': 1000,           # 剩余筹码量
        'to_call': 20            # 需要跟注的金额
    }

if __name__ == "__main__":
    # 初始化AI系统（首次运行会自动创建模型）
    poker_ai = HybridPokerAI()
    
    # 生成游戏状态
    game_state = simulate_game_state()
    
    # 获取决策
    decision = poker_ai.decide_action(game_state)
    

    # 打印当前牌局状态
    print("当前牌局状态：")
    print(f"手牌: {[Card.int_to_str(c) for c in game_state['hero_hand']]}")
    print(f"公共牌: {[Card.int_to_str(c) for c in game_state['community']]}")
    print(f"位置: {game_state['position']}")
    print("\n推荐决策：")
    for action in decision:
        print(f"- {action}")