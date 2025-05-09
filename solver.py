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

# ====================== 常量与工具 ======================
NUM_FEATURES = 15
STAGES = ['preflop', 'flop', 'turn', 'river']
POSITIONS = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']
ACTION_SPACE = ['fold', 'call', 'raise']

def create_card(card_str):
    """字符串转treys Card对象"""
    return Card.new(card_str)

def simulate_game_state():
    """模拟一个典型游戏状态"""
    return {
        'hero_hand': [create_card('Ah'), create_card('Kd')],
        'community': [create_card('Qs'), create_card('Jh'), create_card('Tc')],
        'street': 'flop',
        'position': 'BTN',
        'current_pot': 150,
        'previous_actions': ['check', 'call'],
        'player_id': 1,
        'stack': 1000,
        'to_call': 20
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
        seq = self.action_sequences.get(player_id, [])
        if not seq:
            return [0.5, 0, 0]
        actions = [a['action'] for a in seq]
        bet_sizes = [a['bet_size'] for a in seq if 'bet_size' in a]
        aggressive_count = sum(1 for a in actions if a in ['raise', 'bet'])
        total_count = len(actions)
        avg_bet = np.mean(bet_sizes) if bet_sizes else 0
        aggressive_ratio = aggressive_count / total_count if total_count else 0
        return [aggressive_ratio, avg_bet, total_count]
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
        self.context_model = GlobalContextNet()
        self.opponent_model = OpponentProfiler()
        self.memory = ExperienceReplayBuffer(10000)
        self.online_learner = OnlineAdaptor()
        self._load_model_weights()

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
        turn_model = StageLSTM(32, 64, 3)
        torch.save(turn_model.state_dict(), f'{self.model_path}turn_model.pth')
        river_model = StageLSTM(48, 64, 3)
        torch.save(river_model.state_dict(), f'{self.model_path}river_model.pth')
        dummy_X = np.random.rand(100, 15)
        dummy_y = np.random.randint(0, 3, 100)
        self.stage_models['preflop'].fit(dummy_X, dummy_y)
        dump(self.stage_models['preflop'], f'{self.model_path}preflop_model.joblib')
        self.stage_models['flop'].fit(dummy_X, dummy_y)
        dump(self.stage_models['flop'], f'{self.model_path}flop_model.joblib')
        print(f"基础模型已保存至：{os.path.abspath(self.model_path)}")

    def decide_action(self, game_state):
        raw_features = self.extract_features(game_state)
        context = self.context_model(torch.Tensor(raw_features['context']))
        opp_profile = self.opponent_model.predict(game_state)
        stage = game_state['street']

        if stage == 'preflop':
            features = self._preprocess_preflop(raw_features)
            features = np.array(features)
            if features.shape[0] < NUM_FEATURES:
                features = np.pad(features, (0, NUM_FEATURES - features.shape[0]), mode='constant')
            try:
                probs = self.stage_models[stage].predict_proba([features])[0]
                probs = np.array(probs).flatten()
            except NotFittedError:
                dummy_X = np.random.rand(10, NUM_FEATURES)
                dummy_y = np.random.randint(0, 3, 10)
                self.stage_models[stage].fit(dummy_X, dummy_y)
                probs = self.stage_models[stage].predict_proba([features])[0]
                probs = np.array(probs).flatten()
        else:
            features = self._preprocess_postflop(raw_features, context)
            if features.shape[1] < NUM_FEATURES:
                pad_size = NUM_FEATURES - features.shape[1]
                features = torch.cat([features, torch.zeros((features.shape[0], pad_size))], dim=1)
            np_features = features.numpy()
            if stage in ['turn', 'river']:
                with torch.no_grad():
                    logits, _ = self.stage_models[stage](features, torch.tensor([[STAGES.index(stage)]]))
                    probs = torch.softmax(logits, dim=-1).numpy()
            else:
                try:
                    probs = self.stage_models[stage].predict_proba(np_features)
                    probs = np.array(probs).flatten()
                except NotFittedError:
                    dummy_X = np.random.rand(10, NUM_FEATURES)
                    dummy_y = np.random.randint(0, 3, 10)
                    self.stage_models[stage].fit(dummy_X, dummy_y)
                    probs = self.stage_models[stage].predict_proba(np_features)
                    probs = np.array(probs).flatten()
        final_probs = self._blend_strategies(probs, game_state, opp_profile)
        self._record_decision(game_state, final_probs)
        return self.format_decision(final_probs, game_state)

    def format_decision(self, blended_probs, game_state):
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
        features = np.array(raw_features['context'] + raw_features['stage_features'])
        return features

    def _record_decision(self, game_state, final_probs):
        record = {
            'game_state': game_state.copy(),
            'decision': final_probs
        }
        self.decision_history.append(record)

    def extract_features(self, game_state):
        equity, _ = EnhancedPokerSimulator().calculate_equity(
            game_state['hero_hand'],
            game_state['community'],
            iterations=200,
            street=game_state['street']
        )
        return {
            'context': [
                equity,
                len(game_state['community']),
                game_state['current_pot'] / 100,
                int(game_state['position'] in ['BTN', 'CO']),
                len(game_state.get('previous_actions', [])),
                game_state.get('to_call', 0) / game_state['current_pot'] if game_state['current_pot'] > 0 else 0,
                game_state.get('stack', 1000) / 1000
            ],
            'stage_features': [
                *self._get_stage_specific_features(game_state),
            ]
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

    def _blend_strategies(self, ml_probs, game_state, opp_profile):
        if isinstance(ml_probs, dict):
            ml_probs = [ml_probs.get(a, 0.0) for a in ACTION_SPACE]
        ml_probs = np.array(ml_probs).flatten()
        if ml_probs.shape[0] != 3:
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