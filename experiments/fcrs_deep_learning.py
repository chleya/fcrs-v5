"""
FCRS与PyTorch深度学习集成
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except:
    HAS_TORCH = False
    print("PyTorch not available, using numpy only")


# ==================== FCRS Layer for PyTorch ====================
if HAS_TORCH:
    class FCRSLayer(nn.Module):
        """FCRS层用于神经网络"""
        
        def __init__(self, input_dim, n_representations=5, lambda_val=0.5):
            super().__init__()
            
            self.input_dim = input_dim
            self.n_representations = n_representations
            self.lambda_val = lambda_val
            
            # 表征向量
            self.representations = nn.Parameter(
                torch.randn(n_representations, input_dim) * 0.1
            )
            
            # 输出层
            self.output = nn.Linear(n_representations, 10)
        
        def forward(self, x):
            # x: (batch, input_dim)
            
            # 计算与每个表征的相似度
            # cosine similarity
            x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
            r_norm = self.representations / (self.representations.norm(dim=1, keepdim=True) + 1e-8)
            
            similarity = torch.mm(x_norm, r_norm.T)  # (batch, n_reps)
            
            # 选择最强的表征
            weights = torch.softmax(similarity, dim=1)
            
            # 输出
            out = self.output(weights)
            
            return out
        
        def get_dimension(self):
            """获取当前活跃维度"""
            return self.input_dim


# ==================== Test ====================
def test_torch():
    """测试PyTorch集成"""
    print('='*60)
    print('FCRS PyTorch Integration')
    print('='*60)
    
    if not HAS_TORCH:
        print('PyTorch not available')
        return
    
    # 创建模型
    model = FCRSLayer(input_dim=64, n_representations=5, lambda_val=0.5)
    
    print(f'Model: {model}')
    
    # 测试前向
    x = torch.randn(8, 64)
    out = model(x)
    print(f'Input: {x.shape}')
    print(f'Output: {out.shape}')
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 伪造数据
    for epoch in range(10):
        x = torch.randn(32, 64)
        y = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        
        out = model(x)
        loss = criterion(out, y)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: loss={loss.item():.4f}')
    
    print('\n[OK] PyTorch integration works!')


# ==================== NumPy版本 ====================
class FCRSNumPy:
    """NumPy版本的FCRS神经网络"""
    
    def __init__(self, input_dim=64, hidden=32, output=10, n_reps=5):
        self.n_reps = n_reps
        
        # 表征
        self.representations = np.random.randn(n_reps, input_dim) * 0.1
        
        # 隐藏层
        self.W1 = np.random.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        
        # 输出层
        self.W2 = np.random.randn(hidden, output) * 0.1
        self.b2 = np.zeros(output)
    
    def forward(self, x):
        # 表征注意力
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        r_norm = self.representations / (np.linalg.norm(self.representations, axis=1, keepdims=True) + 1e-8)
        
        similarity = np.dot(x_norm, r_norm.T)
        attention = self.softmax(similarity, axis=1)
        
        # 隐藏层
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        
        # 输出层
        out = np.dot(h, self.W2) + self.b2
        
        return out
    
    def softmax(self, x, axis=1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def train(self, X, y, epochs=20):
        for epoch in range(epochs):
            for x, label in zip(X, y):
                # 前向
                out = self.forward(x.reshape(1, -1))[0]
                
                # 简化梯度
                error = out.copy()
                error[label] -= 1
                
                # 更新
                self.W2 += 0.01 * np.outer(self.W1 @ x, error)
                self.b2 += 0.01 * error
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def test_numpy():
    """测试NumPy版本"""
    print('\n' + '='*60)
    print('FCRS NumPy Version')
    print('='*60)
    
    # 使用sklearn digits
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # 训练
    model = FCRSNumPy(input_dim=64, hidden=32, output=10)
    model.train(X_train, y_train, epochs=20)
    
    # 测试
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'Accuracy: {acc:.2%}')


def main():
    if HAS_TORCH:
        test_torch()
    
    test_numpy()
    
    print('\n[OK] Deep learning integration complete!')


if __name__ == "__main__":
    main()
