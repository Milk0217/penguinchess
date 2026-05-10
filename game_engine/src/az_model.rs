/// Rust-native inference for all AlphaZero model architectures.
/// Supports MLP, ResNetOriginal, and ResNetConfigurable with BN folding.
///
/// Weight layout (flat f32 array, BN-folded):
///   MLP:     fc1(512×206+512), fc2(256×512+256), policy(60×256+60),
///            value1(128×256+128), value2(1×128+1)
///   ResNet:  fc1(512×206+512), fc2(512×512+512), fc3(256×512+256),
///            residual fc's, policy(60×256+60), value1(128×256+128), value2(1×128+1)

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;
    #[inline(always)] pub unsafe fn fma(w: *const f32, xk: __m256, y: *mut f32, j: usize) {
        _mm256_storeu_ps(y.add(j), _mm256_fmadd_ps(_mm256_loadu_ps(w.add(j)), xk, _mm256_loadu_ps(y.add(j))));
    }
    #[inline(always)] pub unsafe fn relu(x: *mut f32, j: usize) {
        _mm256_storeu_ps(x.add(j), _mm256_max_ps(_mm256_loadu_ps(x.add(j)), _mm256_setzero_ps()));
    }
    #[inline(always)] pub unsafe fn splat(x: f32) -> __m256 { _mm256_set1_ps(x) }
    #[inline(always)] pub unsafe fn add(dst: *mut f32, src: *const f32, j: usize) {
        _mm256_storeu_ps(dst.add(j), _mm256_add_ps(_mm256_loadu_ps(dst.add(j)), _mm256_loadu_ps(src.add(j))));
    }
}

// ─── Matvec ───────────────────────────────────────────────────

#[inline(always)]
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize,
          bias: &[f32], out: &mut [f32]) {
    out.copy_from_slice(bias);
    for k in 0..cols {
        let xk = x[k];
        let col = &w[k * rows..(k + 1) * rows];
        #[cfg(target_arch = "x86_64")]
        {
            let mut j = 0usize;
            while j + 8 <= rows { unsafe { simd::fma(col.as_ptr(), simd::splat(xk), out.as_mut_ptr(), j); } j += 8; }
            for j in j..rows { out[j] += col[j] * xk; }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for j in 0..rows { out[j] += col[j] * xk; }
    }
}

fn relu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut j = 0usize;
        while j + 8 <= x.len() { unsafe { simd::relu(x.as_mut_ptr(), j); } j += 8; }
        for i in j..x.len() { if x[i] < 0.0 { x[i] = 0.0; } }
    }
    #[cfg(not(target_arch = "x86_64"))]
    for v in x.iter_mut() { if *v < 0.0 { *v = 0.0; } }
}

// ─── BN folding ───────────────────────────────────────────────

/// Fold BatchNorm into preceding Linear layer.
/// (w_folded, b_folded) where:
///   w_folded[i,j] = w[i,j] * gamma[i] / sqrt(var[i] + eps)
///   b_folded[i] = (b[i] - mean[i]) / sqrt(var[i] + eps) * gamma[i] + beta[i]
pub fn fold_bn(w: &[f32], b: &[f32],
               bn_w: &[f32], bn_b: &[f32], bn_mean: &[f32], bn_var: &[f32],
               eps: f32, rows: usize, cols: usize)
    -> (Vec<f32>, Vec<f32>)
{
    let mut wf = vec![0.0f32; rows * cols];
    let mut bf = vec![0.0f32; rows];
    for i in 0..rows {
        let s = bn_w[i] / (bn_var[i] + eps).sqrt();
        for j in 0..cols { wf[i * cols + j] = w[i * cols + j] * s; }
        bf[i] = (b[i] - bn_mean[i]) * s + bn_b[i];
    }
    (wf, bf)
}

// ─── Observation encoding ─────────────────────────────────────

pub const OBS_DIM: usize = 206;

/// Encode a game state to 206-dim observation vector.
/// Board(180): 60 hex × [q/8, r/8, points/3]
/// Pieces(24): 6 pieces × [id/10, q/8, r/8, s/8] (dead=-1,0,0,0)
/// Meta(2):    [current_player, phase]
pub fn encode_obs(board_cells: &[crate::board::HexCell],
                  pieces: &[crate::board::Piece],
                  current_player: usize, phase: u8) -> [f32; OBS_DIM]
{
    let mut obs = [0.0f32; OBS_DIM];
    for (i, cell) in board_cells.iter().enumerate() {
        if i >= 60 { break; }
        let p = cell.points as f32;
        let val = match cell.state {
            crate::board::HexState::Active => p / 3.0,
            crate::board::HexState::Occupied => p / 3.0,
            _ => 0.0,
        };
        obs[i * 3] = cell.coord.q as f32 / 8.0;
        obs[i * 3 + 1] = cell.coord.r as f32 / 8.0;
        obs[i * 3 + 2] = val;
    }
    for (i, piece) in pieces.iter().enumerate() {
        let base = 180 + i * 4;
        if piece.alive && piece.hex_idx.is_some() {
            obs[base] = piece.id as f32 / 10.0;
            if let Some(h) = piece.hex_idx {
                if h < board_cells.len() {
                    obs[base + 1] = board_cells[h].coord.q as f32 / 8.0;
                    obs[base + 2] = board_cells[h].coord.r as f32 / 8.0;
                    obs[base + 3] = board_cells[h].coord.s as f32 / 8.0;
                }
            }
        } else {
            obs[base] = -1.0;
        }
    }
    obs[204] = current_player as f32;
    obs[205] = phase as f32;
    obs
}

// ─── Architecture config ──────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AZArch {
    MLP,
    ResNet,
}

#[derive(Debug, Clone)]
pub struct AZModelWeights {
    pub arch: AZArch,
    pub layers: Vec<AZLayer>,
    // Flat weights and biases for each layer (after BN folding)
    pub weights: Vec<f32>,  // all flat weight matrices
    pub biases: Vec<f32>,   // all bias vectors
    pub layer_info: Vec<(usize, usize)>, // (rows, cols) per layer
    // Head specs
    pub policy_idx: usize,  // index in layers for policy head
    pub value1_idx: usize,  // index for value_fc1 (or None)
    pub value2_idx: usize,  // index for value_fc2 (output)
    pub value_uses_obs: bool, // if true, value head reads obs directly (PPO-style separate nets)
}

/// A single layer: weight matrix dimensions
#[derive(Debug, Clone, Copy)]
pub struct AZLayer {
    pub weight_offset: usize,  // offset into weights[]
    pub bias_offset: usize,    // offset into biases[]
    pub rows: usize,
    pub cols: usize,
    pub has_relu: bool,
    pub is_residual: bool,     // if true, input is added to output
}

impl AZModelWeights {
    /// Total flat weight count: sum of all layer weights + biases.
    pub fn total_floats(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    pub fn forward(&self, obs: &[f32; OBS_DIM]) -> (Vec<f32>, f32) {
        let mut h_prev = vec![0.0f32; 1];
        let mut x = obs.to_vec();
        let trunk_end = self.policy_idx;

        // Shared trunk (AlphaZero) or policy trunk
        for layer_idx in 0..trunk_end {
            let l = &self.layers[layer_idx];
            let w = &self.weights[l.weight_offset..l.weight_offset + l.rows * l.cols];
            let b = &self.biases[l.bias_offset..l.bias_offset + l.rows];

            if l.is_residual {
                h_prev = x.clone();
            }

            x.resize(l.cols, 0.0);
            let mut out = vec![0.0f32; l.rows];
            matvec(w, &x, l.rows, l.cols, b, &mut out);

            if l.is_residual {
                for i in 0..l.rows { out[i] += h_prev[i]; }
            }
            if l.has_relu { relu_inplace(&mut out); }

            x = out;
        }

        // Policy head
        let pl = &self.layers[self.policy_idx];
        let pw = &self.weights[pl.weight_offset..pl.weight_offset + pl.rows * pl.cols];
        let pb = &self.biases[pl.bias_offset..pl.bias_offset + pl.rows];
        let mut policy = vec![0.0f32; pl.rows];
        matvec(pw, &x, pl.rows, pl.cols, pb, &mut policy);

        // Value head (may share trunk or use separate PPO-style network)
        let value_input: Vec<f32> = if self.value_uses_obs {
            obs.to_vec() // PPO: value has its own separate network
        } else {
            x.clone() // AlphaZero: value shares trunk output
        };

        let v1 = &self.layers[self.value1_idx];
        let mut val_x = if v1.cols == OBS_DIM && self.value_uses_obs {
            value_input.clone()
        } else {
            value_input.clone()
        };
        let v1w = &self.weights[v1.weight_offset..v1.weight_offset + v1.rows * v1.cols];
        let v1b = &self.biases[v1.bias_offset..v1.bias_offset + v1.rows];
        let mut value = vec![0.0f32; v1.rows];
        matvec(v1w, &val_x, v1.rows, v1.cols, v1b, &mut value);
        relu_inplace(&mut value);

        let v2 = &self.layers[self.value2_idx];
        let v2w = &self.weights[v2.weight_offset..v2.weight_offset + v2.rows * v2.cols];
        let v2b = &self.biases[v2.bias_offset..v2.bias_offset + v2.rows];
        let mut vout = [0.0f32; 1];
        matvec(v2w, &value, 1, v2.cols, v2b, &mut vout);

        (policy, vout[0].tanh())
    }

    pub fn evaluate_batch(&self, obs_batch: &[[f32; OBS_DIM]]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n = obs_batch.len();
        let mut logits_all = Vec::with_capacity(n);
        let mut values_all = Vec::with_capacity(n);
        for obs in obs_batch {
            let (logits, val) = self.forward(obs);
            logits_all.push(logits);
            values_all.push(val);
        }
        (logits_all, values_all)
    }
}
