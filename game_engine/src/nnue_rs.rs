/// NNUE forward pass — transposed weight format for cache-friendly access.
/// Architecture:
///   360 sparse binary → FT(64) → CReLU(acc_stm || acc_nstm) → concat dense[66]
///   → FC1(256) → ReLU → FC2(128) → ReLU → FC3(1) → tanh
///
/// FC weights stored in TRANSPOSED (column-major) format:
///   weight_t[col * rows + row] = original[row][col]
/// This gives CONTIGUOUS memory access for the inner loop.

use serde::{Deserialize, Serialize};

pub const FT_DIM: usize = 64;
pub const DENSE_DIM: usize = 66;
pub const FC1_DIM: usize = 256;  // must match NNUE.py HIDDEN_DIM
pub const FC2_DIM: usize = 128;  // must match NNUE.py HIDDEN_DIM / 2
pub const INPUT_DIM: usize = FT_DIM * 2 + DENSE_DIM; // 194
pub const P1_CUTOFF: usize = 180;

// ─── Weights ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNUEWeights {
    pub ft_weight: Vec<f32>,     // (360, FT_DIM=64) row-major
    pub ft_bias: Vec<f32>,
    pub fc1_weight_t: Vec<f32>,  // (INPUT_DIM=194, FC1_DIM=256) TRANSPOSED
    pub fc1_bias: Vec<f32>,
    pub fc2_weight_t: Vec<f32>,  // (FC1_DIM=256, FC2_DIM=128) TRANSPOSED
    pub fc2_bias: Vec<f32>,
    pub fc3_weight_t: Vec<f32>,  // (FC2_DIM=128, 1) — just 128 floats
    pub fc3_bias: Vec<f32>,
}

impl NNUEWeights {
    pub fn new_empty() -> Self {
        Self {
            ft_weight: vec![0.0; 360 * FT_DIM],
            ft_bias: vec![0.0; FT_DIM],
            fc1_weight_t: vec![0.0; INPUT_DIM * FC1_DIM],
            fc1_bias: vec![0.0; FC1_DIM],
            fc2_weight_t: vec![0.0; FC1_DIM * FC2_DIM],
            fc2_bias: vec![0.0; FC2_DIM],
            fc3_weight_t: vec![0.0; FC2_DIM * 1],
            fc3_bias: vec![0.0; 1],
        }
    }

    pub const fn total_floats() -> usize {
        360 * FT_DIM + FT_DIM
            + FC1_DIM * INPUT_DIM + FC1_DIM
            + FC2_DIM * FC1_DIM + FC2_DIM
            + 1 * FC2_DIM + 1
    }

    /// Construct from flat float array. FC weights are TRANSPOSED on load.
    pub fn from_flat(data: &[f32]) -> Self {
        let mut off = 0;
        let ft_weight = data[off..off + 360 * FT_DIM].to_vec(); off += 360 * FT_DIM;
        let ft_bias = data[off..off + FT_DIM].to_vec(); off += FT_DIM;

        // fc1: original (256, 194) row-major → transpose to (194, 256)
        let fc1_rows = FC1_DIM; let fc1_cols = INPUT_DIM;
        let mut fc1_weight_t = vec![0.0f32; fc1_cols * fc1_rows];
        for row in 0..fc1_rows {
            for col in 0..fc1_cols {
                fc1_weight_t[col * fc1_rows + row] = data[off + row * fc1_cols + col];
            }
        }
        off += fc1_rows * fc1_cols;
        let fc1_bias = data[off..off + FC1_DIM].to_vec(); off += FC1_DIM;

        // fc2: original (128, 256) → transpose to (256, 128)
        let fc2_rows = FC2_DIM; let fc2_cols = FC1_DIM;
        let mut fc2_weight_t = vec![0.0f32; fc2_cols * fc2_rows];
        for row in 0..fc2_rows {
            for col in 0..fc2_cols {
                fc2_weight_t[col * fc2_rows + row] = data[off + row * fc2_cols + col];
            }
        }
        off += fc2_rows * fc2_cols;
        let fc2_bias = data[off..off + FC2_DIM].to_vec(); off += FC2_DIM;

        // fc3: original (1, 128) → transpose to (128, 1) = same layout, just reshape
        let fc3_rows = 1usize; let fc3_cols = FC2_DIM;
        let mut fc3_weight_t = vec![0.0f32; fc3_cols * fc3_rows];
        for row in 0..fc3_rows {
            for col in 0..fc3_cols {
                fc3_weight_t[col * fc3_rows + row] = data[off + row * fc3_cols + col];
            }
        }
        off += fc3_rows * fc3_cols;
        let fc3_bias = data[off..off + 1].to_vec();

        Self { ft_weight, ft_bias, fc1_weight_t, fc1_bias, fc2_weight_t, fc2_bias, fc3_weight_t, fc3_bias }
    }

    pub fn validate(&self) -> bool {
        self.ft_weight.len() == 360 * FT_DIM && self.ft_bias.len() == FT_DIM
            && self.fc1_weight_t.len() == INPUT_DIM * FC1_DIM && self.fc1_bias.len() == FC1_DIM
            && self.fc2_weight_t.len() == FC1_DIM * FC2_DIM && self.fc2_bias.len() == FC2_DIM
            && self.fc3_weight_t.len() == FC2_DIM * 1 && self.fc3_bias.len() == 1
    }
}

// ─── Accumulator ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NNUEAccumulator {
    pub acc_stm: [f32; FT_DIM],
    pub acc_nstm: [f32; FT_DIM],
}

impl NNUEAccumulator {
    pub fn new(weights: &NNUEWeights) -> Self {
        let mut acc = Self { acc_stm: [0.0; FT_DIM], acc_nstm: [0.0; FT_DIM] };
        acc.acc_stm.copy_from_slice(&weights.ft_bias);
        acc.acc_nstm.copy_from_slice(&weights.ft_bias);
        acc
    }

    pub fn reset(&mut self, weights: &NNUEWeights) {
        self.acc_stm.copy_from_slice(&weights.ft_bias);
        self.acc_nstm.copy_from_slice(&weights.ft_bias);
    }

    pub fn apply_sparse(&mut self, features: &[usize], stm_player: usize, weights: &NNUEWeights) {
        for &f in features {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let src = &weights.ft_weight[f * FT_DIM..(f + 1) * FT_DIM];
            let dst: &mut [f32; 64] = if stm { &mut self.acc_stm } else { &mut self.acc_nstm };
            for i in 0..FT_DIM { dst[i] += src[i]; }
        }
    }

    pub fn apply_diff(&mut self, removed: &[usize], added: &[usize],
                       stm_player: usize, weights: &NNUEWeights) {
        for &f in removed {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let src = &weights.ft_weight[f * FT_DIM..(f + 1) * FT_DIM];
            let dst: &mut [f32; 64] = if stm { &mut self.acc_stm } else { &mut self.acc_nstm };
            for i in 0..FT_DIM { dst[i] -= src[i]; }
        }
        for &f in added {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let src = &weights.ft_weight[f * FT_DIM..(f + 1) * FT_DIM];
            let dst: &mut [f32; 64] = if stm { &mut self.acc_stm } else { &mut self.acc_nstm };
            for i in 0..FT_DIM { dst[i] += src[i]; }
        }
    }

    pub fn get_crelu(&self) -> [f32; FT_DIM * 2] {
        let mut out = [0.0f32; FT_DIM * 2];
        for i in 0..FT_DIM {
            out[i] = self.acc_stm[i].max(0.0).min(127.0);
            out[FT_DIM + i] = self.acc_nstm[i].max(0.0).min(127.0);
        }
        out
    }
}

// ─── Matvec with transposed (column-major) weights ────────────

/// y = W * x + b, W stored column-major: w_t[col * rows + row] = original[row][col]
/// For each input element x[k], we add W[:,k] * x[k] to out, with contiguous memory access.
#[inline(always)]
fn matvec_mul_add_t(w_t: &[f32], x: &[f32], rows: usize, cols: usize,
                    bias: &[f32], out: &mut [f32]) {
    out.copy_from_slice(bias);
    for k in 0..cols {
        let xk = x[k];
        let col = &w_t[k * rows..(k + 1) * rows];
        for j in 0..rows {
            out[j] += col[j] * xk;
        }
    }
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
    }
}

// ─── Public API ───────────────────────────────────────────────

pub fn nnue_evaluate(features: &[usize], dense: &[f32],
                      stm_player: usize, weights: &NNUEWeights) -> f32 {
    let mut acc = NNUEAccumulator::new(weights);
    acc.apply_sparse(features, stm_player, weights);
    evaluate_from_acc(&acc, dense, weights)
}

pub fn evaluate_from_acc(acc: &NNUEAccumulator, dense: &[f32],
                          weights: &NNUEWeights) -> f32 {
    let crelu = acc.get_crelu();
    let mut x = [0.0f32; INPUT_DIM];
    x[..FT_DIM * 2].copy_from_slice(&crelu);
    x[FT_DIM * 2..].copy_from_slice(dense);

    // FC1: 194 → 256
    let mut h1 = [0.0f32; FC1_DIM];
    matvec_mul_add_t(&weights.fc1_weight_t, &x, FC1_DIM, INPUT_DIM, &weights.fc1_bias, &mut h1);
    relu_inplace(&mut h1);

    // FC2: 256 → 128
    let mut h2 = [0.0f32; FC2_DIM];
    matvec_mul_add_t(&weights.fc2_weight_t, &h1, FC2_DIM, FC1_DIM, &weights.fc2_bias, &mut h2);
    relu_inplace(&mut h2);

    // FC3: 128 → 1
    let mut out = [0.0f32; 1];
    matvec_mul_add_t(&weights.fc3_weight_t, &h2, 1, FC2_DIM, &weights.fc3_bias, &mut out);

    out[0].tanh()
}
