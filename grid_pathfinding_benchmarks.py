# Grid pathfinding benchmarks: Dijkstra vs A* vs Q-learning (inference only)

import numpy as np
import time, math, tracemalloc
import matplotlib.pyplot as plt
from heapq import heappush, heappop

# ---------------------- Setup ----------------------

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

class GridWorld:
    def __init__(self, grid, start, goal, wall_bump_penalty=-5.0):
        self.grid = grid.astype(int)
        self.nrows, self.ncols = grid.shape
        self.start, self.goal = start, goal
        self.wall_bump_penalty = wall_bump_penalty

    def in_bounds(self, r, c): 
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def is_free(self, r, c): 
        return self.in_bounds(r, c) and self.grid[r, c] == 0

    def neighbors(self, s):
        r, c = s
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                yield (nr, nc)

    def step(self, s, a):
        r, c = s
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr, nc) or self.grid[nr, nc] == 1:
            return (r, c), self.wall_bump_penalty, False  # bump penalty, stay put
        s2 = (nr, nc)
        if s2 == self.goal:
            return s2, 100.0, True
        return s2, -1.0, False

def manhattan(a, b): 
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def reconstruct_path(prev, start, goal):
    if (prev[goal][0] == -1 and start != goal):
        return None
    if start == goal:
        return [start]
    path, cur = [], goal
    while True:
        path.append(tuple(cur))
        if tuple(cur) == start:
            break
        pr, pc = prev[cur[0], cur[1]]
        if pr == -1:
            return None
        cur = (int(pr), int(pc))
    path.reverse()
    return path

# ---------------------- Planners ----------------------

def dijkstra(env: GridWorld):
    inf = float('inf')
    dist = np.full((env.nrows, env.ncols), inf, dtype=float)
    prev = np.full((env.nrows, env.ncols, 2), -1, dtype=int)

    pq = []
    dist[env.start] = 0.0
    heappush(pq, (0.0, env.start))
    popped = 0

    while pq:
        d, u = heappop(pq); popped += 1
        if u == env.goal: 
            break
        ur, uc = u
        if d > dist[ur, uc]:
            continue
        for v in env.neighbors(u):
            vr, vc = v
            nd = d + 1.0
            if nd < dist[vr, vc]:
                dist[vr, vc] = nd
                prev[vr, vc] = [ur, uc]
                heappush(pq, (nd, v))

    path = reconstruct_path(prev, env.start, env.goal)
    cost = dist[env.goal] if path is not None else math.inf
    return path, cost, popped

def astar(env: GridWorld):
    h = lambda s: manhattan(s, env.goal)
    inf = float('inf')
    g = np.full((env.nrows, env.ncols), inf, dtype=float)
    prev = np.full((env.nrows, env.ncols, 2), -1, dtype=int)

    openpq = []
    g[env.start] = 0.0
    heappush(openpq, (h(env.start), env.start))
    popped = 0
    closed = set()

    while openpq:
        f, u = heappop(openpq); popped += 1
        if u == env.goal: 
            break
        if u in closed: 
            continue
        closed.add(u)
        ur, uc = u
        for v in env.neighbors(u):
            vr, vc = v
            tentative = g[ur, uc] + 1.0
            if tentative < g[vr, vc]:
                g[vr, vc] = tentative
                prev[vr, vc] = [ur, uc]
                heappush(openpq, (tentative + h(v), v))

    path = reconstruct_path(prev, env.start, env.goal)
    cost = g[env.goal] if path is not None else math.inf
    return path, cost, popped

# ---------------------- RL (train + inference) ----------------------

def q_learning(env: GridWorld, episodes=1500, alpha=0.5, gamma=0.99,
               eps_start=1.0, eps_end=0.05, eps_decay_episodes=800,
               max_steps=400, shaping=True):
    """Train tabular Q-learning and return Q; no plotting or printing here."""
    Q = np.zeros((env.nrows, env.ncols, len(ACTIONS)), dtype=float)
    def phi(s): return -manhattan(s, env.goal)  # potential for shaping

    eps = eps_start
    decay = (eps_start - eps_end) / max(1, eps_decay_episodes)

    for _ in range(episodes):
        s = env.start
        for _ in range(max_steps):
            r, c = s
            a = np.random.randint(len(ACTIONS)) if np.random.rand() < eps else int(np.argmax(Q[r, c, :]))
            s2, reward, done = env.step(s, a)
            if shaping:
                reward = reward + gamma * phi(s2) - phi(s)
            r2, c2 = s2
            td_target = reward + (0.0 if done else gamma * np.max(Q[r2, c2, :]))
            Q[r, c, a] += alpha * (td_target - Q[r, c, a])
            s = s2
            if done:
                break
        eps = max(eps_end, eps - decay)
    return Q

def rl_rollout(env: GridWorld, Q, max_steps=2000):
    """Greedy policy rollout: returns (steps, solved, path)."""
    s = env.start
    steps = 0
    path = [s]
    for _ in range(max_steps):
        r, c = s
        a = int(np.argmax(Q[r, c, :]))
        s2, _, done = env.step(s, a)
        steps += 1
        path.append(s2)
        if done:
            return steps, True, path
        if s2 == s:     # stuck
            return steps, False, path
        s = s2
    return steps, False, path

# ---------------------- Utilities ----------------------

def generate_problem(nrows=15, ncols=15, obstacle_prob=0.22, seed=42, tries=50):
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        grid = (rng.random((nrows, ncols)) < obstacle_prob).astype(int)
        free = list(zip(*np.where(grid == 0)))
        if len(free) < 2:
            continue
        start, goal = free[0], free[-1]
        grid[start] = 0
        grid[goal] = 0
        env = GridWorld(grid, start, goal)
        p, _, _ = astar(env)
        if p is not None:
            return env
    # fallback: empty grid
    grid = np.zeros((nrows, ncols), dtype=int)
    return GridWorld(grid, (0, 0), (nrows-1, ncols-1))

def timed_peakmem(fn, *args, **kwargs):
    """Run fn and return (result, runtime_s, peak_bytes) measured by tracemalloc."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, (t1 - t0), peak

# ---------------------- Main ----------------------

def main():
    env = generate_problem()

    # Dijkstra
    (d_path, d_cost, d_pops), d_time, d_peak = timed_peakmem(dijkstra, env)
    d_core = (env.nrows * env.ncols * 8) + (env.nrows * env.ncols * 2 * 8)  # dist (f64) + prev (i64[*,*,2])

    # A*
    (a_path, a_cost, a_pops), a_time, a_peak = timed_peakmem(astar, env)
    a_core = (env.nrows * env.ncols * 8) + (env.nrows * env.ncols * 2 * 8)  # g + prev

    # Q-learning TRAIN (reported but not used for inference comparison)
    (Q,), qtrain_time, qtrain_peak = timed_peakmem(lambda e: (q_learning(e),), env)

    # Q-learning INFERENCE ONLY
    (rl_steps, rl_solved, rl_path), qinf_time, qinf_peak = timed_peakmem(rl_rollout, env, Q, 2000)
    rl_core = Q.nbytes  # core memory = Q-table only

    print("== Benchmark (single query; RL inference only) ==")
    print(f"Dijkstra:      time={d_time*1000:.3f} ms | peak_mem={d_peak/1024:.2f} KB | core_arrays≈{d_core/1024:.2f} KB | nodes={d_pops} | path_len={(len(d_path)-1) if d_path else 'N/A'}")
    print(f"A*:            time={a_time*1000:.3f} ms | peak_mem={a_peak/1024:.2f} KB | core_arrays≈{a_core/1024:.2f} KB | nodes={a_pops} | path_len={(len(a_path)-1) if a_path else 'N/A'}")
    print(f"Q-learn TRAIN: time={qtrain_time:.3f} s | peak_mem={qtrain_peak/1024:.2f} KB")
    print(f"Q-learn INFER: time={qinf_time*1000:.3f} ms | peak_mem={qinf_peak/1024:.2f} KB | core_arrays≈{rl_core/1024:.2f} KB | path_len={rl_steps if rl_solved else 'fail'}")

    # -------- Plot overlay --------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(env.grid, origin='lower', interpolation='none')
    ax.set_xticks(np.arange(-0.5, env.ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.nrows, 1), minor=True)
    ax.grid(which='minor', linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])

    sr, sc = env.start
    gr, gc = env.goal
    ax.scatter([sc], [sr], s=80, marker='o', label='Start')
    ax.scatter([gc], [gr], s=80, marker='*', label='Goal')

    def draw(name, path):
        if not path:
            return
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, linewidth=2, label=name)

    draw("Dijkstra", d_path)
    draw("A*", a_path)
    draw("Q-learning", rl_path if rl_solved else None)

    ax.legend(loc='upper right')
    ax.set_title("Grid with Obstacles and Paths")
    plt.show()

if __name__ == "__main__":
    main()
