"""
FCRS多智能体系统
多个FCRS智能体协作
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from core import FCRSystem


class FCRSAgent:
    """FCRS智能体"""
    
    def __init__(self, agent_id, input_dim=10):
        self.agent_id = agent_id
        self.input_dim = input_dim
        
        # 独立FCRS系统
        self.fcrs = FCRSystem(pool_capacity=5, vector_dim=input_dim)
        self.fcrs.engine.spawn_reuse_threshold = 2
        self.fcrs.engine.min_compression_gain = 0.001
        
        # 共享表征库(类变量)
        if not hasattr(FCRSAgent, 'shared_pool'):
            FCRSAgent.shared_pool = []
    
    def step(self, x):
        """单步"""
        self.fcrs.step()
        
        # 尝试贡献到共享池
        if self.fcrs.pool.representations:
            # 贡献最优表征
            best = max(self.fcrs.pool.representations, key=lambda r: r.reuse)
            if best.reuse > 5:
                FCRSAgent.shared_pool.append({
                    'agent': self.agent_id,
                    'vector': best.vector,
                    'reuse': best.reuse
                })
        
        return self.get_info()
    
    def get_info(self):
        return {
            'id': self.agent_id,
            'dims': self.fcrs.pool.get_total_dims(),
            'new_dims': len(self.fcrs.engine.new_dim_history)
        }


class FCRSMultiAgent:
    """FCRS多智能体系统"""
    
    def __init__(self, n_agents=3, input_dim=10):
        self.n_agents = n_agents
        self.agents = []
        
        # 创建智能体
        for i in range(n_agents):
            agent = FCRSAgent(agent_id=i, input_dim=input_dim)
            self.agents.append(agent)
    
    def run(self, steps=100):
        """运行多步"""
        for step in range(steps):
            # 每个智能体处理输入
            x = np.random.randn(10)
            
            for agent in self.agents:
                agent.step(x)
            
            if (step + 1) % 50 == 0:
                print('Step ' + str(step + 1) + ':')
                for agent in self.agents:
                    info = agent.get_info()
                    print('  Agent' + str(info['id']) + ': ' + str(info['dims']) + '维, ' + str(info['new_dims']) + '新维度')
    
    def get_total_stats(self):
        """获取总体统计"""
        total_dims = 0
        total_new = 0
        
        for agent in self.agents:
            info = agent.get_info()
            total_dims += info['dims']
            total_new += info['new_dims']
        
        return {
            'n_agents': self.n_agents,
            'total_dims': total_dims,
            'total_new_dims': total_new,
            'shared_pool_size': len(FCRSAgent.shared_pool)
        }


def test_multi_agent():
    """测试多智能体"""
    print('='*60)
    print('FCRS多智能体测试')
    print('='*60)
    
    # 创建多智能体系统
    system = FCRSMultiAgent(n_agents=3)
    
    # 运行
    system.run(steps=100)
    
    # 统计
    stats = system.get_total_stats()
    
    print('\n' + '='*60)
    print('总体统计')
    print('='*60)
    print('智能体数量: ' + str(stats['n_agents']))
    print('总维度: ' + str(stats['total_dims']))
    print('总新维度: ' + str(stats['total_new_dims']))
    print('共享池大小: ' + str(stats['shared_pool_size']))


if __name__ == "__main__":
    test_multi_agent()
