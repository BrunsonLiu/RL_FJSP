"""Quick experiment: BC + AC fine-tuning on mk01 with low lr."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fjsp.env import FJSPDispatchEnv
from rl.agents import train_graph_actor_critic

env = FJSPDispatchEnv.from_file("data/instances/brandimarte/mk01.txt")
agent, history = train_graph_actor_critic(
    env,
    episodes=500,
    lr=3e-5,
    hidden_dim=64,
    gnn_rounds=2,
    seed=0,
    init_model="data/results/imitation_mk01_best.pt",
)
result = agent.rollout(env, greedy=True)
print(f"BC+AC greedy: {result.makespan}")
print(f"best in history: {history[-1]['best_greedy_makespan']} @ ep {int(history[-1]['best_episode'])}")
print("Trajectory (ep, sample, greedy, best):")
for h in history:
    print(f"  ep={int(h['episode']):3d} sample={h['sample_makespan']:.0f} greedy={h['greedy_makespan']:.0f} best={h['best_greedy_makespan']:.0f}")

# Save the best model
from pathlib import Path
model_path = Path("data/results/bc_ac_mk01_best.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)
agent.save(model_path)
print(f"Saved best model to {model_path}")
