"""Replicate BC + AC pipeline: BC 100 rollouts -> AC 500 episodes. Saves model + history."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r'd:\desktop2\RL_FJSP')

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents import train_graph_actor_critic, train_imitation

ROOT = Path(r'd:\desktop2\RL_FJSP')
env = FJSPDispatchEnv.from_file(str(ROOT / 'data' / 'instances' / 'brandimarte' / 'mk01.txt'))

print(f"[BC] training on mk01 with 14,6 features...")
t0 = time.perf_counter()
bc_agent, bc_history = train_imitation(
    env, rollouts=100, epochs=20, batch_size=32, lr=1e-3,
    hidden_dim=64, gnn_rounds=2, seed=0,
)
bc_path = ROOT / 'data' / 'results' / 'bc14_ac14_mk01_bc.pt'
bc_agent.save(str(bc_path))
bc_greedy = bc_agent.rollout(env, greedy=True).makespan
print(f"[BC] trained in {time.perf_counter()-t0:.1f}s, greedy={bc_greedy}")

print(f"[AC] fine-tuning 500 ep, lr=3e-5...")
t0 = time.perf_counter()
ac_agent, ac_history = train_graph_actor_critic(
    env, episodes=500, lr=3e-5, hidden_dim=64, gnn_rounds=2, seed=0,
    init_model=str(bc_path),
)
ac_path = ROOT / 'data' / 'results' / 'bc14_ac14_mk01_ac.pt'
ac_agent.save(str(ac_path))
ac_greedy = ac_agent.rollout(env, greedy=True).makespan
print(f"[AC] trained in {time.perf_counter()-t0:.1f}s, best={ac_history[-1]['best_greedy_makespan']}, final_greedy={ac_greedy}")

# Save histories + schedule
hist_path = ROOT / 'data' / 'results' / 'bc14_ac14_mk01_history.json'
hist_path.write_text(json.dumps({
    'bc_history': bc_history,
    'ac_history': ac_history,
    'bc_greedy': bc_greedy,
    'ac_greedy': ac_greedy,
    'ac_best': ac_history[-1]['best_greedy_makespan'],
}, indent=2))
sched = ac_agent.rollout(env, greedy=True)
sched_path = ROOT / 'data' / 'results' / 'bc14_ac14_mk01_schedule.json'
sched_path.write_text(json.dumps(schedule_to_dict(env.schedule), indent=2))
print(f"[DONE] history={hist_path}, schedule={sched_path}")
