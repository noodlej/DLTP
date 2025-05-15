import numpy as np
from train_stage2 import train_model

log_path = "/data/users/jnoodle/saliency/log.txt"

# 하이퍼파라미터 샘플링 함수
def sample_hyperparams():
    lr = np.random.uniform(0.000001, 0.0001)           # 1e-4 ~ 1e-2
    weight_decay = np.random.uniform(0.0001, 0.1) # 1e-6 ~ 1e-2
    eta = np.random.uniform(0.01, 0.3)           # 1e-3 ~ 1e-1
    return lr, weight_decay, eta

best_worst_val = -1.0
best_result = {}

for i in range(20):
    print(f"\n🔁 Trial {i+1}/20")
    lr, wd, eta = sample_hyperparams()
    print(f"  ▶ lr={lr:.6f}, weight_decay={wd:.6f}, eta={eta:.6f}")

    try:
        val_acc, test_acc, stopped_epoch = train_model(lr=lr, weight_decay=wd, eta=eta)
        worst_val = val_acc.min()
        print(f"  ✅ worst-group val acc: {worst_val:.4f}")

        if worst_val > best_worst_val:
            best_worst_val = worst_val
            best_result = {
                "lr": lr,
                "weight_decay": wd,
                "eta": eta,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "epoch": stopped_epoch
            }
            print("  ⭐️ New best model found!")
        else:
            print("  ❌ Not better than current best")

    except Exception as e:
        print(f"  ⚠️ Error during training: {e}")

# 결과 저장
if best_result:
    with open(log_path, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("Best Hyperparameters from 20-sweep trials:\n")
        f.write(f"lr           : {best_result['lr']:.6f}\n")
        f.write(f"weight_decay : {best_result['weight_decay']:.6f}\n")
        f.write(f"eta          : {best_result['eta']:.6f}\n\n")
        f.write(f"stopped_epoch: {best_result['epoch']}\n\n")
        f.write(f"best val group acc : {best_result['val_acc'].tolist()}\n")
        f.write(f"best test group acc: {best_result['test_acc'].tolist()}\n")
        f.write("="*80 + "\n")
        print("\n✅ Best result saved to log.txt")
else:
    print("❌ No successful trial to log.")
