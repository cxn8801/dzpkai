# dzpkai
提供游戏接口用于开发德州扑克AI，检验AI的实战效果


运行脚本前，要先安装下面几个包

pip install websockets

pip install loguru

pip install protobuf

pip install treys

pip install joblib

pip install scikit-learn

pip install xgboost

需要留意我的服务器IP地址，服务器到期我会找更便宜的服务器，我会及时在这里更新

'ws://81.69.249.211:19001'



安卓客户端已经上传：dzpk-release.apk




待解决问题
输出结果分析
text
未找到预计算范围，使用随机策略
模型加载失败: [Errno 2] No such file or directory: 'poker_model.pkl'，初始化应急模型
推荐决策: ['55% raise 90BB', '27% call', '17% fold']
1. 预计算范围缺失
原因：首次运行时未生成手牌范围数据

影响：对手手牌模拟从Top20%范围回退到完全随机策略

解决方案：运行预计算生成手牌强度数据

python
# 首次运行前执行
range_manager = RangeManager()
range_manager.generate_ranges()  # 约需10-15分钟
2. 机器学习模型缺失
原因：未找到预训练模型文件poker_model.pkl

影响：使用基于随机数据的应急模型，预测准确性较低

解决方案：分阶段完善模型

python
# 临时方案：使用虚拟数据快速验证
np.random.seed(42)
X = np.random.rand(1000, 5)
y = np.random.choice(['fold', 'call', 'raise'], 1000)
model = RandomForestClassifier().fit(X, y)
joblib.dump(model, 'poker_model.pkl')
3. 决策结果解读
合理性分析：

raise 55%：按钮位（BTN）+ 较高底池权益时激进策略合理

call 27%：保留后续操作空间

fold 17%：防范极端情况

可信度评估：

基础策略权重占比70%（Nash均衡）

机器学习预测权重30%（当前为随机数据）

系统完善路线图
第一阶段：基础数据生成（立即执行）
python
# 生成手牌范围数据
RangeManager().generate_ranges()

# 生成应急模型数据
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 生成10000条模拟数据
X = np.random.rand(10000, 5)
y = np.random.choice([0, 1, 2], 10000)  # 0:fold, 1:call, 2:raise
model = RandomForestClassifier(n_estimators=100).fit(X, y)
joblib.dump(model, 'poker_model.pkl')
第二阶段：数据收集系统（建议2天开发）
python
class DataCollector:
    def __init__(self):
        self.dataset = []
    
    def record_decision(self, features, action):
        """记录特征和对应动作"""
        self.dataset.append({
            'features': features,
            'action': action,
            'timestamp': time.time()
        })
    
    def save_data(self, path='poker_data.json'):
        with open(path, 'w') as f:
            json.dump(self.dataset, f)

# 在DecisionEngine中集成
class DecisionEngine:
    def __init__(self):
        self.collector = DataCollector()
    
    def make_decision(self, game_state):
        # ...原有代码...
        self.collector.record_decision(features, final_action)
第三阶段：模型训练系统（建议3天开发）
python
def train_model():
    # 加载收集的数据
    with open('poker_data.json') as f:
        data = json.load(f)
    
    X = np.array([d['features'] for d in data])
    y = np.array([d['action'] for d in data])
    
    # 使用更先进的模型
    from xgboost import XGBClassifier
    model = XGBClassifier().fit(X, y)
    
    # 模型验证
    from sklearn.metrics import classification_report
    print(classification_report(y, model.predict(X)))
    
    joblib.dump(model, 'poker_model_v2.pkl')
性能优化建议
优化方向	实施方法	预期收益
并行模拟计算	使用concurrent.futures加速蒙特卡洛模拟	提速3-5倍
缓存预热	启动时预加载常用范围数据	首决策提速60%
GPU加速	使用CUDA加速评估器	评估速度提升10倍
增量更新	实现模型热更新机制	零停机更新
典型问题排查表
现象	可能原因	解决方案
决策始终偏向fold	应急模型初始化失败	检查DummyClassifier导入
计算时间过长	未启用缓存	确认CachedEvaluator已实例化
位置策略不生效	策略表键值不匹配	检查NashStrategy.STRATEGY_TABLE格式
概率合计不等于100%	权重归一化问题	在_format_decision添加校验逻辑
通过以上改进方案，系统将从当前的基础可用状态逐步发展为专业级扑克决策系统。建议按阶段实施，并重点关注数据收集质量，这是提升模型性能的关键。