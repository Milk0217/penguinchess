/// NNUE training: gradient computation + Adam optimizer + training loop.
use crate::nnue_rs::*;

// ─── Gradient Buffer ──────────────────────────────────────────

pub struct GradientBuffer {
    pub d_ft: Vec<f32>,      // (360 * FT_DIM)
    pub d_ft_b: Vec<f32>,    // FT_DIM
    pub d_fc1_t: Vec<f32>,   // (INPUT_DIM * FC1_DIM)
    pub d_fc1_b: Vec<f32>,   // FC1_DIM
    pub d_fc2_t: Vec<f32>,   // (FC1_DIM * FC2_DIM)
    pub d_fc2_b: Vec<f32>,   // FC2_DIM
    pub d_fc3_t: Vec<f32>,   // (FC2_DIM * 1)
    pub d_fc3_b: Vec<f32>,   // 1
}

impl GradientBuffer {
    pub fn new() -> Self {
        Self {
            d_ft: vec![0.0; 360 * FT_DIM], d_ft_b: vec![0.0; FT_DIM],
            d_fc1_t: vec![0.0; INPUT_DIM * FC1_DIM],
            d_fc1_b: vec![0.0; FC1_DIM],
            d_fc2_t: vec![0.0; FC1_DIM * FC2_DIM],
            d_fc2_b: vec![0.0; FC2_DIM],
            d_fc3_t: vec![0.0; FC2_DIM * 1],
            d_fc3_b: vec![0.0; 1],
        }
    }
    pub fn zero(&mut self) {
        for v in &mut self.d_ft { *v = 0.0; }
        for v in &mut self.d_ft_b { *v = 0.0; }
        for v in &mut self.d_fc1_t { *v = 0.0; }
        for v in &mut self.d_fc1_b { *v = 0.0; }
        for v in &mut self.d_fc2_t { *v = 0.0; }
        for v in &mut self.d_fc2_b { *v = 0.0; }
        for v in &mut self.d_fc3_t { *v = 0.0; }
        for v in &mut self.d_fc3_b { *v = 0.0; }
    }
    pub fn clip(&mut self, max_norm: f32) {
        let mut ss = 0.0f32;
        for v in &self.d_ft { ss += v * v; }
        for v in &self.d_ft_b { ss += v * v; }
        for v in &self.d_fc1_t { ss += v * v; }
        for v in &self.d_fc1_b { ss += v * v; }
        for v in &self.d_fc2_t { ss += v * v; }
        for v in &self.d_fc2_b { ss += v * v; }
        for v in &self.d_fc3_t { ss += v * v; }
        for v in &self.d_fc3_b { ss += v * v; }
        let n = ss.sqrt();
        if n > max_norm {
            let s = max_norm / n;
            for v in &mut self.d_ft { *v *= s; }
            for v in &mut self.d_ft_b { *v *= s; }
            for v in &mut self.d_fc1_t { *v *= s; }
            for v in &mut self.d_fc1_b { *v *= s; }
            for v in &mut self.d_fc2_t { *v *= s; }
            for v in &mut self.d_fc2_b { *v *= s; }
            for v in &mut self.d_fc3_t { *v *= s; }
            for v in &mut self.d_fc3_b { *v *= s; }
        }
    }
    pub fn scale(&mut self, f: f32) {
        for v in &mut self.d_ft { *v *= f; }
        for v in &mut self.d_ft_b { *v *= f; }
        for v in &mut self.d_fc1_t { *v *= f; }
        for v in &mut self.d_fc1_b { *v *= f; }
        for v in &mut self.d_fc2_t { *v *= f; }
        for v in &mut self.d_fc2_b { *v *= f; }
        for v in &mut self.d_fc3_t { *v *= f; }
        for v in &mut self.d_fc3_b { *v *= f; }
    }
}

// ─── Adam ─────────────────────────────────────────────────────

pub struct AdamState {
    pub m_ft: Vec<f32>, pub v_ft: Vec<f32>,
    pub m_ft_b: Vec<f32>, pub v_ft_b: Vec<f32>,
    pub m_fc1_t: Vec<f32>, pub v_fc1_t: Vec<f32>,
    pub m_fc1_b: Vec<f32>, pub v_fc1_b: Vec<f32>,
    pub m_fc2_t: Vec<f32>, pub v_fc2_t: Vec<f32>,
    pub m_fc2_b: Vec<f32>, pub v_fc2_b: Vec<f32>,
    pub m_fc3_t: Vec<f32>, pub v_fc3_t: Vec<f32>,
    pub m_fc3_b: Vec<f32>, pub v_fc3_b: Vec<f32>,
    pub t: u32,
}

impl AdamState {
    pub fn new() -> Self {
        Self {
            m_ft: vec![], v_ft: vec![], m_ft_b: vec![], v_ft_b: vec![],
            m_fc1_t: vec![], v_fc1_t: vec![], m_fc1_b: vec![], v_fc1_b: vec![],
            m_fc2_t: vec![], v_fc2_t: vec![], m_fc2_b: vec![], v_fc2_b: vec![],
            m_fc3_t: vec![], v_fc3_t: vec![], m_fc3_b: vec![], v_fc3_b: vec![],
            t: 0,
        }
    }
    pub fn init(&mut self, _w: &NNUEWeights) {
        self.m_ft = vec![0.0; 360 * FT_DIM]; self.v_ft = vec![0.0; 360 * FT_DIM];
        self.m_ft_b = vec![0.0; FT_DIM]; self.v_ft_b = vec![0.0; FT_DIM];
        self.m_fc1_t = vec![0.0; INPUT_DIM * FC1_DIM]; self.v_fc1_t = vec![0.0; INPUT_DIM * FC1_DIM];
        self.m_fc1_b = vec![0.0; FC1_DIM]; self.v_fc1_b = vec![0.0; FC1_DIM];
        self.m_fc2_t = vec![0.0; FC1_DIM * FC2_DIM]; self.v_fc2_t = vec![0.0; FC1_DIM * FC2_DIM];
        self.m_fc2_b = vec![0.0; FC2_DIM]; self.v_fc2_b = vec![0.0; FC2_DIM];
        self.m_fc3_t = vec![0.0; FC2_DIM * 1]; self.v_fc3_t = vec![0.0; FC2_DIM * 1];
        self.m_fc3_b = vec![0.0; 1]; self.v_fc3_b = vec![0.0; 1];
    }
}

fn adam_apply(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, wd: f32, b1t: f32, b2t: f32) {
    for i in 0..w.len() {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        let m_hat = m[i] * b1t;
        let v_hat = v[i] * b2t;
        w[i] -= lr * m_hat / (v_hat.sqrt() + 1e-8) + lr * wd * w[i];
    }
}

pub fn adam_step(w: &mut NNUEWeights, g: &GradientBuffer, a: &mut AdamState, lr: f32, wd: f32) {
    a.t += 1;
    let b1t = 1.0 / (1.0 - 0.9f32.powi(a.t as i32));
    let b2t = 1.0 / (1.0 - 0.999f32.powi(a.t as i32));
    // b1t, b2t = 1/(1-β^t), so m_hat = m * b1t, v_hat = v * b2t
    // Pass bias correction factors to adam_apply

    adam_apply(&mut w.ft_weight, &g.d_ft, &mut a.m_ft, &mut a.v_ft, lr, wd, b1t, b2t);
    adam_apply(&mut w.ft_bias, &g.d_ft_b, &mut a.m_ft_b, &mut a.v_ft_b, lr, wd, b1t, b2t);
    adam_apply(&mut w.fc1_weight_t, &g.d_fc1_t, &mut a.m_fc1_t, &mut a.v_fc1_t, lr, wd, b1t, b2t);
    adam_apply(&mut w.fc1_bias, &g.d_fc1_b, &mut a.m_fc1_b, &mut a.v_fc1_b, lr, wd, b1t, b2t);
    adam_apply(&mut w.fc2_weight_t, &g.d_fc2_t, &mut a.m_fc2_t, &mut a.v_fc2_t, lr, wd, b1t, b2t);
    adam_apply(&mut w.fc2_bias, &g.d_fc2_b, &mut a.m_fc2_b, &mut a.v_fc2_b, lr, wd, b1t, b2t);
    adam_apply(&mut w.fc3_weight_t, &g.d_fc3_t, &mut a.m_fc3_t, &mut a.v_fc3_t, lr, wd, b1t, b2t);
    adam_apply(&mut w.fc3_bias, &g.d_fc3_b, &mut a.m_fc3_b, &mut a.v_fc3_b, lr, wd, b1t, b2t);
}

// ─── Training ─────────────────────────────────────────────────

pub struct TrainingRecord {
    pub sparse: [i32; 6],
    pub dense: [f32; 66],
    pub label: f32,
    pub stm: i32,
}

pub fn load_records(path: &str) -> Vec<TrainingRecord> {
    use std::fs::File;
    use std::io::Read;
    let mut f = File::open(path).unwrap();
    let mut header = [0u8; 8];
    f.read_exact(&mut header).ok();
    let count = u64::from_le_bytes(header) as usize;
    let mut records = Vec::with_capacity(count.min(1_000_000));
    let mut raw = [0u8; 296];
    for _ in 0..count {
        if f.read_exact(&mut raw).is_err() { break; }
        let mut sparse = [-1i32; 6];
        for i in 0..6 {
            let val = i32::from_le_bytes([raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]]);
            if val >= 0 && val < 360 { sparse[i] = val; }
        }
        let mut dense = [0.0f32; 66];
        for i in 0..66 {
            dense[i] = f32::from_le_bytes([raw[24+i*4], raw[25+i*4], raw[26+i*4], raw[27+i*4]]);
        }
        let label = f32::from_le_bytes([raw[288], raw[289], raw[290], raw[291]]);
        let stm = i32::from_le_bytes([raw[292], raw[293], raw[294], raw[295]]);
        records.push(TrainingRecord { sparse, dense, label, stm });
    }
    records
}

/// Backward pass for one sample. Accumulates gradients into `g`.
/// Reuses forward intermediates computed here.
pub fn backward_sample(
    weights: &NNUEWeights, rec: &TrainingRecord, g: &mut GradientBuffer,
) -> f32 {
    // ── 1. Forward ────────────────────────────────────────────

    // FT accumulation
    let mut acc_stm = [0.0f32; FT_DIM];
    let mut acc_nstm = [0.0f32; FT_DIM];
    acc_stm.copy_from_slice(&weights.ft_bias);
    acc_nstm.copy_from_slice(&weights.ft_bias);

    let mut stm_features = Vec::with_capacity(3);
    let mut nstm_features = Vec::with_capacity(3);

    for &f in &rec.sparse {
        if f < 0 { continue; }
        let base = (f as usize) * FT_DIM;
        let is_p1 = f < 180;
        let is_stm = (rec.stm == 0 && is_p1) || (rec.stm == 1 && !is_p1);
        let target = if is_stm { &mut acc_stm } else { &mut acc_nstm };
        for i in 0..FT_DIM { target[i] += weights.ft_weight[base + i]; }
        if is_stm { stm_features.push(f); } else { nstm_features.push(f); }
    }

    // CReLU → x (128-dim, values in [0, 127])
    let mut x = [0.0f32; INPUT_DIM];
    for i in 0..FT_DIM {
        x[i] = acc_stm[i].clamp(0.0, 127.0);
    }
    for i in 0..FT_DIM {
        x[FT_DIM + i] = acc_nstm[i].clamp(0.0, 127.0);
    }
    x[FT_DIM*2..].copy_from_slice(&rec.dense);

    // FC1 → h1
    let mut h1 = [0.0f32; FC1_DIM];
    for j in 0..FC1_DIM { h1[j] = weights.fc1_bias[j]; }
    for k in 0..INPUT_DIM {
        let xk = x[k];
        for j in 0..FC1_DIM { h1[j] += weights.fc1_weight_t[k * FC1_DIM + j] * xk; }
    }
    let h1_pre = h1;
    for j in 0..FC1_DIM { h1[j] = if h1[j] > 0.0 { h1[j] } else { 0.0 }; }

    // FC2 → h2
    let mut h2 = [0.0f32; FC2_DIM];
    for j in 0..FC2_DIM { h2[j] = weights.fc2_bias[j]; }
    for k in 0..FC1_DIM {
        let h1k = h1[k];
        for j in 0..FC2_DIM { h2[j] += weights.fc2_weight_t[k * FC2_DIM + j] * h1k; }
    }
    let h2_pre = h2;
    for j in 0..FC2_DIM { h2[j] = if h2[j] > 0.0 { h2[j] } else { 0.0 }; }

    // FC3 → out
    let mut out = 0.0f32;
    out += weights.fc3_bias[0];
    for k in 0..FC2_DIM { out += weights.fc3_weight_t[k * 1 + 0] * h2[k]; }
    out = out.tanh();
    let pred = out;

    // ── 2. Loss ───────────────────────────────────────────────
    let d_out = pred - rec.label;  // MSE gradient
    let d_tanh = d_out * (1.0 - pred * pred);

    // ── 3. Backward FC3 ───────────────────────────────────────
    let rows3 = 1usize; let cols3 = FC2_DIM;
    for k in 0..cols3 {
        g.d_fc3_t[k * rows3 + 0] += d_tanh * h2[k];
    }
    g.d_fc3_b[0] += d_tanh;

    let mut d_h2 = [0.0f32; FC2_DIM];
    for k in 0..cols3 { d_h2[k] = weights.fc3_weight_t[k * rows3 + 0] * d_tanh; }

    // Backward ReLU2
    for j in 0..FC2_DIM {
        let dh = d_h2[j];
        if h2_pre[j] <= 0.0 { continue; }
        // ── 4. Backward FC2 ───────────────────────────────────
        g.d_fc2_b[j] += dh;
        for k in 0..FC1_DIM {
            g.d_fc2_t[k * FC2_DIM + j] += dh * h1[k];
            // d_h1 infrastructure: accumulated in next step
        }
    }

    // d_h1 from FC2 backward
    let mut d_h1 = [0.0f32; FC1_DIM];
    for j in 0..FC2_DIM {
        let dh = d_h2[j];
        if h2_pre[j] <= 0.0 { continue; }
        for k in 0..FC1_DIM {
            d_h1[k] += weights.fc2_weight_t[k * FC2_DIM + j] * dh;
        }
    }

    // Backward ReLU1
    for j in 0..FC1_DIM {
        let dh = d_h1[j];
        if h1_pre[j] <= 0.0 { continue; }
        g.d_fc1_b[j] += dh;
        for k in 0..INPUT_DIM {
            g.d_fc1_t[k * FC1_DIM + j] += dh * x[k];
        }
    }

    // d_x from FC1 backward (needed for CReLU backward)
    let mut d_x = [0.0f32; INPUT_DIM];
    for j in 0..FC1_DIM {
        let dh = d_h1[j];
        if h1_pre[j] <= 0.0 { continue; }
        for k in 0..INPUT_DIM {
            d_x[k] += weights.fc1_weight_t[k * FC1_DIM + j] * dh;
        }
    }

    // ── 5. Backward CReLU + FT ────────────────────────────────
    for i in 0..FT_DIM {
        let d_crelu = d_x[i];
        let d_acc_stm_i = if acc_stm[i] > 0.0 && acc_stm[i] < 127.0 { d_crelu } else { 0.0 };
        g.d_ft_b[i] += d_acc_stm_i;
        for &f in &stm_features {
            g.d_ft[(f as usize) * FT_DIM + i] += d_acc_stm_i;
        }

        let d_crelu_n = d_x[FT_DIM + i];
        let d_acc_nstm_i = if acc_nstm[i] > 0.0 && acc_nstm[i] < 127.0 { d_crelu_n } else { 0.0 };
        g.d_ft_b[i] += d_acc_nstm_i;
        for &f in &nstm_features {
            g.d_ft[(f as usize) * FT_DIM + i] += d_acc_nstm_i;
        }
    }

    let loss = 0.5 * (pred - rec.label) * (pred - rec.label);
    loss
}

/// Train one epoch. Returns average loss.
pub fn train_epoch(
    weights: &mut NNUEWeights, records: &[TrainingRecord],
    adam: &mut AdamState, config: &TrainingConfig,
) -> f32 {
    let n = records.len();
    let bs = config.batch_size.min(n);
    let n_batches = (n + bs - 1) / bs;
    let mut g = GradientBuffer::new();
    let mut total_loss = 0.0f32;

    for b in 0..n_batches {
        g.zero();
        let start = b * bs;
        let end = (start + bs).min(n);
        let bsize = (end - start) as f32;
        let mut batch_loss = 0.0f32;

        for i in start..end {
            let loss = backward_sample(weights, &records[i], &mut g);
            batch_loss += loss;
        }

        // Average gradients
        g.scale(1.0 / bsize);
        g.clip(config.max_norm);

        // Adam step
        adam_step(weights, &g, adam, config.learning_rate, config.weight_decay);

        total_loss += batch_loss / bsize;
    }

    total_loss / n_batches as f32
}

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub max_norm: f32,
    pub batch_size: usize,
    pub n_epochs: usize,
    pub lr_min: f32,
    pub warmup_epochs: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self { learning_rate: 3e-4, weight_decay: 1e-4, max_norm: 1.0, batch_size: 4096, n_epochs: 50,
               lr_min: 3e-5, warmup_epochs: 5 }
    }
}

pub struct TrainingResult {
    pub best_loss: f32,
    pub best_weights: NNUEWeights,
    pub epoch_losses: Vec<f32>,
}

/// Main training function.
pub fn train(weights: &mut NNUEWeights, records: &[TrainingRecord], config: &TrainingConfig) -> TrainingResult {
    let mut adam = AdamState::new();
    adam.init(weights);
    let mut best_loss = f32::MAX;
    let mut best_weights = weights.clone();
    let mut epoch_losses = Vec::with_capacity(config.n_epochs);

    for ep in 0..config.n_epochs {
        // Cosine annealing LR with linear warmup
        let lr = if ep < config.warmup_epochs {
            config.learning_rate * (ep + 1) as f32 / config.warmup_epochs as f32
        } else {
            let progress = (ep - config.warmup_epochs) as f32 / (config.n_epochs - config.warmup_epochs).max(1) as f32;
            config.lr_min + 0.5 * (config.learning_rate - config.lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
        };
        let cfg_i = TrainingConfig {
            learning_rate: lr,
            weight_decay: config.weight_decay,
            max_norm: config.max_norm,
            batch_size: config.batch_size,
            n_epochs: 1,
            lr_min: config.lr_min,
            warmup_epochs: 0,
        };
        let loss = train_epoch(weights, records, &mut adam, &cfg_i);
        epoch_losses.push(loss);
        if loss < best_loss {
            best_loss = loss;
            best_weights = weights.clone();
        }
        if ep == 0 || (ep + 1) % 10 == 0 || ep == config.n_epochs - 1 {
            println!("  ep {}/{}  lr={:.2e}  loss={:.6}", ep + 1, config.n_epochs, lr, loss);
        }
    }

    weights.clone_from(&best_weights);
    TrainingResult { best_loss, best_weights, epoch_losses }
}
