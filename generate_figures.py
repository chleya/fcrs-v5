import matplotlib.pyplot as plt

# 数据
env = [1, 2, 3, 4]

# 图1: 环境复杂度 vs Loss
loss_A = [0.25, 0.45, 1.04, 0.73]
loss_B = [0.27, 0.51, 0.78, 0.62]
loss_C = [0.24, 0.41, 0.72, 0.43]

plt.figure(figsize=(10, 6))
plt.plot(env, loss_A, 'o-', label='A: Fixed', linewidth=2, markersize=8)
plt.plot(env, loss_B, 's-', label='B: Expansion', linewidth=2, markersize=8)
plt.plot(env, loss_C, '^-', label='C: ECS', linewidth=2, markersize=8)
plt.xlabel('Environment Complexity', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Figure 1: Environment Complexity vs Loss', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(env, ['E1', 'E2', 'E3', 'E4'])
plt.tight_layout()
plt.savefig('fig1_complexity_vs_loss.png', dpi=150)
plt.close()

# 图2: 环境复杂度 vs 维度
dim_A = [8, 8, 8, 8]
dim_B = [14, 24, 25, 25]
dim_C = [8, 27, 29, 28]

plt.figure(figsize=(10, 6))
plt.plot(env, dim_A, 'o-', label='A: Fixed', linewidth=2, markersize=8)
plt.plot(env, dim_B, 's-', label='B: Expansion', linewidth=2, markersize=8)
plt.plot(env, dim_C, '^-', label='C: ECS', linewidth=2, markersize=8)
plt.xlabel('Environment Complexity', fontsize=12)
plt.ylabel('Active Dimensions', fontsize=12)
plt.title('Figure 2: Environment Complexity vs Dimensions', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(env, ['E1', 'E2', 'E3', 'E4'])
plt.tight_layout()
plt.savefig('fig2_complexity_vs_dim.png', dpi=150)
plt.close()

# 图3: 维度 vs Loss
dims = [8, 8, 8, 8, 14, 24, 25, 25, 8, 27, 29, 28]
losses = [0.25, 0.45, 1.04, 0.73, 0.27, 0.51, 0.78, 0.62, 0.24, 0.41, 0.72, 0.43]
colors = ['blue']*4 + ['green']*4 + ['red']*4

plt.figure(figsize=(10, 6))
for i in range(0, 4):
    plt.scatter(dims[i], losses[i], c='blue', s=100, label='A' if i==0 else '')
for i in range(4, 8):
    plt.scatter(dims[i], losses[i], c='green', s=100, label='B' if i==4 else '')
for i in range(8, 12):
    plt.scatter(dims[i], losses[i], c='red', s=100, label='C' if i==8 else '')

plt.xlabel('Active Dimensions', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Figure 3: Dimensions vs Loss', fontsize=14)
plt.legend(['A: Fixed', 'B: Expansion', 'C: ECS'], fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_dim_vs_loss.png', dpi=150)
plt.close()

print("三张图已保存:")
print("- fig1_complexity_vs_loss.png")
print("- fig2_complexity_vs_dim.png")
print("- fig3_dim_vs_loss.png")
