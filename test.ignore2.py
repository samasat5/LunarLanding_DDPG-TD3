import pandas as pd
import matplotlib.pyplot as plt


ddpg_bias = pd.read_csv("biasesDDPG-2.csv")
td3_bias = pd.read_csv("biasesTD3-2.csv")

plt.figure(figsize=(10,6))
plt.plot(ddpg_bias["step"], ddpg_bias["smoothed"], label="DDPG (smoothed)", color="tab:blue")
plt.plot(td3_bias["step"], td3_bias["smoothed"], label="TD3 (smoothed)", color="tab:orange")


plt.plot(ddpg_bias["step"], ddpg_bias["raw"], alpha=0.2, color="tab:blue")
plt.plot(td3_bias["step"], td3_bias["raw"], alpha=0.2, color="tab:orange")

plt.title("Comparison of MC Bias (Q − Gₜ) between DDPG and TD3")
plt.xlabel("Training Steps")
plt.ylabel("MC Bias (Q − Gₜ)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

