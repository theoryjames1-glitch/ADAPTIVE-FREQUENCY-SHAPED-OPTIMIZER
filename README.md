Love the ambition. Here’s a concrete design for an optimizer that **learns to converge faster as it trains**, not just by scaling steps (Adam-style) but by **adapting its own frequency response** and step size online.

# AFSO: Adaptive Frequency-Shaped Optimizer

**Goal:** minimize “time-to-target loss.”
**Mechanism:** (1) filter the gradient with a stable SOS (biquad cascade); (2) **learn** the filter’s poles/zeros + the step size via a **slow controller** that watches progress, noise, and oscillation; (3) enforce stability with safe parameterizations.

---

## What “getting better” means (a measurable signal)

We adapt hyperparameters to increase a **per-step efficiency** metric:

$$
\textstyle \mathcal{P}_t \;\triangleq\; \frac{L_{t-1}-L_t}{\lVert \Delta\theta_t\rVert + \epsilon}
$$

— “loss drop per unit step.”
We maintain an EMA, $\bar{\mathcal{P}}_t$, and we **steer** hyperparameters to raise $\bar{\mathcal{P}}_t$.

We also track:

* **Oscillation score** $O_t$: sign changes / underdamped behavior in loss or gradient (e.g., $\text{corr}(g_t,g_{t-1})<0$, or rising $\mathrm{Var}(L)$).
* **Noise ratio** $N_t$: $\mathrm{Var}(g_t)/\mathrm{Var}(\tilde g_t)$ (how much the filter is cleaning vs. passing noise).
* **Progress trend** $D_t$: EMA of raw $(L_{t-1}-L_t)$.

These three signals let us **turn the knobs** safely.

---

## Knobs we adapt (per-group / per-layer)

* **Learning rate** $\eta$
* **Biquad pole radius** $r \in (0,1)$ (smoothing strength)
* **Resonance angle** $\theta \in [0,\pi]$ (where we allow/pass vs. suppress)
* (Optional) **Notch depth** near $\pi$ (to kill high-freq mini-batch chatter)

We **parameterize for stability**:

* $r = \sigma(s)\cdot r_{\max}$ with $r_{\max}\approx 0.995$, $s\in\mathbb{R}$ (always inside unit circle)
* $\theta = \pi \cdot \sigma(u)$, $u\in\mathbb{R}$ (bounded)
* DC-gain normalized at each tweak to keep $H(1)=1$

---

## Controllers (online, lightweight)

**1) LR Controller (PID-style on progress)**

* Error: $e_t = \mathcal{P}^\star - \bar{\mathcal{P}}_t$ (target efficiency $\mathcal{P}^\star$)
* Update: $\log\eta \leftarrow \log\eta + k_p e_t + k_i \sum e + k_d(e_t-e_{t-1})$
  (log-space keeps $\eta>0$ and gives smooth multiplicative changes)

**2) Radius Controller (damp if oscillatory, relax if safe)**

* If $O_t$ rises (oscillation), **decrease $r$** (more damping).
* If $N_t$ high but $O_t$ low (no oscillation, just noise), **increase $r$** (more smoothing).
* Update: $s \leftarrow s + \alpha_r \cdot \text{clip}(w_N N_t - w_O O_t, -c, c)$.

**3) Angle / Notch Controller (match dominant oscillation)**

* Estimate dominant frequency via a tiny AR(2) fit to gradient or loss residuals (equivalent to finding a peak in the short autocorrelation); call it $\hat\theta_t$.
* Nudge $\theta \leftarrow (1-\alpha_\theta)\theta + \alpha_\theta \hat\theta_t$.
* If high-freq noise dominates, deepen a $\pi$-notch a bit; if not, relax it.

Each controller updates **slowly** (e.g., once per K steps) with small gains.

---

## PyTorch sketch (plug-and-play)

```python
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

class Biquad:
    # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    def __init__(self, device, shape):
        self.device = device
        self.x1 = torch.zeros(shape, device=device)
        self.x2 = torch.zeros(shape, device=device)
        self.y1 = torch.zeros(shape, device=device)
        self.y2 = torch.zeros(shape, device=device)
        # coefficients (buffers)
        self.b = torch.tensor([1., 0., 0.], device=device)  # will set later
        self.a = torch.tensor([1., 0., 0.], device=device)  # [1, a1, a2]

    def set_coeffs(self, b, a):
        self.b = b.clone()
        self.a = a.clone()

    def step(self, x):
        y = (self.b[0]*x + self.b[1]*self.x1 + self.b[2]*self.x2
             - self.a[1]*self.y1 - self.a[2]*self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y

def stable_biquad_from_r_theta(r, theta, dc_unity=True, device='cpu'):
    # Poles at r*e^{±jθ} → a1 = -2 r cos θ, a2 = r^2
    a1 = -2.0 * r * torch.cos(theta)
    a2 = r**2
    # Numerator single tap b0 adjusted to make H(1)=1
    b0 = 1.0 + a1 + a2 if dc_unity else 1.0
    b = torch.tensor([b0, 0.0, 0.0], device=device)
    a = torch.tensor([1.0, a1.item(), a2.item()], device=device)
    return b, a

class AFSO:
    """
    Adaptive Frequency-Shaped Optimizer:
    - Per-parameter SOS (use 1 section here; stack if desired)
    - RMSProp-like scaling
    - Meta-controllers that adapt lr, r, theta every K steps
    """
    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8,
                 r_max=0.995, device='cpu', K=20):
        self.params = list(params)
        self.beta2 = beta2
        self.eps = eps
        self.device = device
        self.K = K

        # Hyperparams in unconstrained form
        self.log_lr = torch.tensor([torch.log(torch.tensor(lr))], device=device)
        self.s_r   = torch.tensor([0.0], device=device)         # r = sigmoid(s_r)*r_max
        self.u_th  = torch.tensor([0.5], device=device)         # theta = pi*sigmoid(u_th)

        # State: per-parameter accumulators
        self.v = [torch.zeros_like(p.data, device=device) for p in self.params]
        self.filters = [Biquad(device, p.data.shape) for p in self.params]

        # Meters
        self.prev_loss = None
        self.P_bar = torch.tensor([0.0], device=device)  # progress EMA
        self.P_target = torch.tensor([1e-3], device=device)
        self.O_bar = torch.tensor([0.0], device=device)  # oscillation score EMA
        self.N_bar = torch.tensor([0.0], device=device)  # noise ratio EMA
        self.m_decay = 0.9

        # Controller gains
        self.kp = 0.2; self.ki = 0.01; self.kd = 0.1
        self.e_int = torch.tensor([0.0], device=device)
        self.e_prev = torch.tensor([0.0], device=device)
        self.alpha_r = 0.02; self.alpha_th = 0.05

        # Init filter coeffs
        self.r_max = r_max

    def _apply_filter_coeffs(self):
        r = torch.sigmoid(self.s_r) * self.r_max
        theta = torch.pi * torch.sigmoid(self.u_th)
        for f in self.filters:
            b, a = stable_biquad_from_r_theta(r, theta, dc_unity=True, device=self.device)
            f.set_coeffs(b, a)

    @torch.no_grad()
    def step(self, loss):
        # 1) Collect grads
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        self._apply_filter_coeffs()

        # 2) Filter gradients + RMSProp scale
        g_tilde = []
        for p, g, v, flt in zip(self.params, grads, self.v, self.filters):
            y = flt.step(g)
            v.mul_(self.beta2).addcmul_(y, y, value=(1 - self.beta2))
            g_tilde.append(y / (v.sqrt() + self.eps))

        # 3) Update params with current lr
        lr = self.log_lr.exp().item()
        delta_norm_sq = 0.0
        for p, gt in zip(self.params, g_tilde):
            update = -lr * gt
            p.add_(update)
            delta_norm_sq += update.pow(2).sum().item()

        # 4) Update meters
        with torch.no_grad():
            L = loss.item()
            if self.prev_loss is not None:
                dL = (self.prev_loss - L)
                step_norm = (delta_norm_sq ** 0.5) + 1e-12
                P = dL / step_norm
                self.P_bar = self.m_decay * self.P_bar + (1 - self.m_decay) * torch.tensor([P], device=self.device)

                # Simple oscillation score: sign flip & variance growth
                O = 1.0 if dL < 0 else 0.0  # penalize regressions
                self.O_bar = self.m_decay * self.O_bar + (1 - self.m_decay) * torch.tensor([O], device=self.device)

                # Rough noise ratio proxy: gradient roughness pre/post filter
                # (Use one tensor for estimate)
                if len(grads):
                    g0 = grads[0].detach().view(-1)
                    y0 = g_tilde[0].detach().view(-1)
                    var_raw = g0.var() + 1e-12
                    var_flt = y0.var() + 1e-12
                    N = (var_raw / var_flt).clamp(max=1e6).log()  # log SNR improvement
                    self.N_bar = self.m_decay * self.N_bar + (1 - self.m_decay) * N.unsqueeze(0)

            self.prev_loss = L

        # 5) Every K steps, adapt hyperparams
        if not hasattr(self, "_k"): self._k = 0
        self._k += 1
        if self._k % self.K == 0 and self.prev_loss is not None:
            # (a) LR PID on progress
            e = (self.P_target - self.P_bar)
            self.e_int += e
            e_der = (e - self.e_prev)
            self.e_prev = e
            self.log_lr += self.kp*e + self.ki*self.e_int + self.kd*e_der
            self.log_lr.clamp_(min=torch.log(torch.tensor(1e-6)), max=torch.log(torch.tensor(1.0)))

            # (b) Radius controller: up with noise cleaning, down with oscillation
            delta_s = self.alpha_r * torch.clamp( 0.5*self.N_bar - 0.8*self.O_bar, min=-0.1, max=0.1 )
            self.s_r += delta_s

            # (c) Angle controller: tiny nudge toward pi if oscillation persists
            # (proxy: if O high, push toward a higher-frequency damping)
            self.u_th += self.alpha_th * torch.clamp(self.O_bar - 0.1, min=-0.05, max=0.05)

            # Keep things sane
            self.s_r.clamp_(min=-6.0, max=6.0)
            self.u_th.clamp_(min=-4.0, max=4.0)
```

**How to use**

```python
model = ...
opt = AFSO(model.parameters(), lr=3e-4, device='cuda', K=20)

for batch in loader:
    loss = compute_loss(model, batch)
    loss.backward()
    opt.step(loss)
    model.zero_grad()
```

---

## Why this should “get better and better”

* The **LR PID** increases/decreases $\eta$ to keep “loss-drop per unit step” on target, making steps more **aggressive when safe**, conservative when not.
* The **radius controller** tunes smoothing: if you’re noisy but stable, it raises $r$ (more low-pass → cleaner descent); if you start ringing, it lowers $r$ (more damping).
* The **angle/notch controller** slowly aligns the filter’s selective suppression to the **dominant oscillation** it detects.

This is all **online**, **differentiation-free** control—cheap enough to run during training. It’s not just “adaptive learning rate”; it’s **adaptive learning rate + adaptive frequency response**.

---

## Practical safeguards

* **Step clipping** on parameter updates if you add more resonant sections.
* **Warm-start**: keep controllers frozen for the first few hundred steps.
* **Per-group adaptation**: share (r, θ) per layer to limit state size.
* **DC-unity normalization** after each tweak (already enforced above).

---

## If you want even more “learning to learn”

The above uses **signal-based controllers**. For maximum adaptation:

* Do the same, but update $(\eta, r, \theta)$ with **meta-gradients** every K steps (truncated backprop-through-time over the optimizer’s state). That’s heavier but can squeeze extra speed.

---

### TL;DR

AFSO pairs a **biquad-filtered gradient** with **online hyper-controllers** that **continuously tune** the learning rate **and** the filter’s pole/zero placement—so your optimizer literally *learns how to descend this specific problem faster* as training proceeds.

If you want, I can:

* add a **2-section (SOS) cascade** variant,
* include **per-layer statistics** and a tiny **frequency estimator** utility,
* or wrap this as a drop-in **PyTorch Optimizer** class with state dict save/load.
