// Port of game/environment.py (rules + 22 state features) and the Dueling Q-network forward pass.
// Works in the browser (global SnakeCore) and in Node (module.exports) for the parity test.
(function (root) {
  'use strict';
  const W = 20, H = 20, N = W * H;
  const DIRS = [[1, 0], [0, 1], [-1, 0], [0, -1]]; // RIGHT, DOWN, LEFT, UP (same order as Python)
  const ROT = [0, 1, 3];                           // action 0 straight, 1 right, 2 left

  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Game {
    constructor(seed, starve = true) {
      this.rand = mulberry32(seed === undefined ? (Date.now() & 0xffffffff) : seed);
      this.starve = starve;
      this.reset();
    }
    reset() {
      const cx = W >> 1, cy = H >> 1;
      this.snake = [[cx, cy], [cx - 1, cy], [cx - 2, cy]]; // head first
      this.dir = 0; this.score = 0; this.hunger = 0; this.over = false; this.won = false;
      this.spawn();
    }
    spawn() {
      const occ = new Uint8Array(N);
      for (const [x, y] of this.snake) occ[y * W + x] = 1;
      const free = [];
      for (let i = 0; i < N; i++) if (!occ[i]) free.push(i);
      if (!free.length) { this.food = null; this.won = true; this.over = true; return; }
      const i = free[Math.floor(this.rand() * free.length)];
      this.food = [i % W, (i / W) | 0];
    }
    limit() { return 100 + 10 * this.snake.length; } // same starvation rule as the Python config
    step(action) {
      if (this.over) return;
      this.dir = (this.dir + ROT[action]) % 4;
      const [dx, dy] = DIRS[this.dir], [hx, hy] = this.snake[0];
      const nx = hx + dx, ny = hy + dy;
      const eat = !!this.food && nx === this.food[0] && ny === this.food[1];
      const body = eat ? this.snake : this.snake.slice(0, -1); // the tail moves away unless we eat
      if (nx < 0 || nx >= W || ny < 0 || ny >= H || body.some(([x, y]) => x === nx && y === ny)) {
        this.over = true; return;
      }
      this.snake.unshift([nx, ny]);
      if (eat) { this.score++; this.hunger = 0; this.spawn(); }
      else { this.snake.pop(); if (this.starve && ++this.hunger >= this.limit()) this.over = true; }
    }
    // Flood fill (BFS) from `start`, avoiding cells set in `mask`.
    reach(start, mask) {
      const seen = new Uint8Array(N), q = new Int16Array(N);
      const [sx, sy] = start;
      if (sx < 0 || sx >= W || sy < 0 || sy >= H || mask[sy * W + sx]) return { n: 0, seen };
      let qh = 0, qt = 0;
      q[qt++] = sy * W + sx; seen[q[0]] = 1;
      while (qh < qt) {
        const c = q[qh++], cx = c % W, cy = (c / W) | 0;
        for (const [dx, dy] of DIRS) {
          const x = cx + dx, y = cy + dy;
          if (x >= 0 && x < W && y >= 0 && y < H) {
            const k = y * W + x;
            if (!mask[k] && !seen[k]) { seen[k] = 1; q[qt++] = k; }
          }
        }
      }
      return { n: qt, seen };
    }
    // 22 features, same order as SnakeEnvironment.get_state():
    // danger(3) direction(4) food-dir(4) free-space(3) tail-reachable(3) food-reachable(3) distance length
    features() {
      const [hx, hy] = this.snake[0], [tx, ty] = this.snake[this.snake.length - 1];
      const mask = new Uint8Array(N); // body without the tail (the head is included)
      for (let i = 0; i < this.snake.length - 1; i++) { const [x, y] = this.snake[i]; mask[y * W + x] = 1; }
      const danger = [], space = [], tail = [], food = [];
      for (const rot of ROT) {
        const [dx, dy] = DIRS[(this.dir + rot) % 4], nx = hx + dx, ny = hy + dy;
        danger.push(nx < 0 || nx >= W || ny < 0 || ny >= H || mask[ny * W + nx] ? 1 : 0);
        const r = this.reach([nx, ny], mask);
        space.push(r.n / N);
        tail.push(r.seen[ty * W + tx]);
        food.push(this.food && r.seen[this.food[1] * W + this.food[0]] ? 1 : 0);
      }
      const oh = [0, 0, 0, 0]; oh[this.dir] = 1;
      const f = this.food;
      const fd = f ? [f[0] < hx, f[0] > hx, f[1] < hy, f[1] > hy].map(Number) : [0, 0, 0, 0];
      const dist = f ? (Math.abs(f[0] - hx) + Math.abs(f[1] - hy)) / (W + H) : 0;
      return Float32Array.from([...danger, ...oh, ...fd, ...space, ...tail, ...food, dist, this.snake.length / N]);
    }
  }

  const lin = (l, x) => {
    const out = new Float32Array(l.b.length);
    for (let i = 0; i < out.length; i++) {
      let s = l.b[i]; const w = l.W[i];
      for (let j = 0; j < x.length; j++) s += w[j] * x[j];
      out[i] = s;
    }
    return out;
  };
  const relu = (x) => x.map((v) => (v > 0 ? v : 0));

  // Dueling head: Q = V + A - mean(A)
  function qvalues(net, s) {
    const h = relu(lin(net.f2, relu(lin(net.f1, s))));
    const v = lin(net.v2, relu(lin(net.v1, h)))[0];
    const a = lin(net.a2, relu(lin(net.a1, h)));
    const m = (a[0] + a[1] + a[2]) / 3;
    return [v + a[0] - m, v + a[1] - m, v + a[2] - m];
  }

  const api = { Game, qvalues, W, H };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.SnakeCore = api;
})(this);