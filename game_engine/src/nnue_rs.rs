/// NNUE forward pass in pure Rust — no Python callbacks during search.
/// Architecture:
///   360 sparse binary → Feature Transformer (64) → CReLU(acc_stm || acc_nstm) → concat dense[66]
///   → FC1(256) → ReLU → FC2(128) → ReLU → FC3(1) → tanh

use serde::{Deserialize, Serialize};

pub const FT_DIM: usize = 64;
pub const DENSE_DIM: usize = 66;
pub const FC1_DIM: usize = 256;
pub const FC2_DIM: usize = 128;
pub const INPUT_DIM: usize = FT_DIM * 2 + DENSE_DIM; // 194
pub const P1_CUTOFF: usize = 180; // P1 features: 0-179, P2: 180-359

/// NNUE weights, stored flat for cache-friendly access.
/// All arrays are row-major: weight[row][col]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNUEWeights {
    pub ft_weight: Vec<f32>,  // (360, FT_DIM=64) row-major
    pub ft_bias: Vec<f32>,    // (FT_DIM=64,)
    pub fc1_weight: Vec<f32>, // (FC1_DIM=256, INPUT_DIM=194) row-major
    pub fc1_bias: Vec<f32>,   // (FC1_DIM=256,)
    pub fc2_weight: Vec<f32>, // (FC2_DIM=128, FC1_DIM=256) row-major
    pub fc2_bias: Vec<f32>,   // (FC2_DIM=128,)
    pub fc3_weight: Vec<f32>, // (1, FC2_DIM=128) row-major
    pub fc3_bias: Vec<f32>,   // (1,)
}

impl NNUEWeights {
    pub fn new_empty() -> Self {
        Self {
            ft_weight: vec![0.0; 360 * FT_DIM],
            ft_bias: vec![0.0; FT_DIM],
            fc1_weight: vec![0.0; FC1_DIM * INPUT_DIM],
            fc1_bias: vec![0.0; FC1_DIM],
            fc2_weight: vec![0.0; FC2_DIM * FC1_DIM],
            fc2_bias: vec![0.0; FC2_DIM],
            fc3_weight: vec![0.0; 1 * FC2_DIM],
            fc3_bias: vec![0.0; 1],
        }
    }

    /// Total floats in the flat weight array.
    pub const fn total_floats() -> usize {
        360 * FT_DIM + FT_DIM
            + FC1_DIM * INPUT_DIM + FC1_DIM
            + FC2_DIM * FC1_DIM + FC2_DIM
            + 1 * FC2_DIM + 1
    }

    /// Construct from flat float array layout:
    ///   ft_weight(360*64), ft_bias(64), fc1_weight(256*194), fc1_bias(256),
    ///   fc2_weight(128*256), fc2_bias(128), fc3_weight(128), fc3_bias(1)
    pub fn from_flat(data: &[f32]) -> Self {
        let mut off = 0;
        let ft_weight = data[off..off + 360 * FT_DIM].to_vec(); off += 360 * FT_DIM;
        let ft_bias = data[off..off + FT_DIM].to_vec(); off += FT_DIM;
        let fc1_weight = data[off..off + FC1_DIM * INPUT_DIM].to_vec(); off += FC1_DIM * INPUT_DIM;
        let fc1_bias = data[off..off + FC1_DIM].to_vec(); off += FC1_DIM;
        let fc2_weight = data[off..off + FC2_DIM * FC1_DIM].to_vec(); off += FC2_DIM * FC1_DIM;
        let fc2_bias = data[off..off + FC2_DIM].to_vec(); off += FC2_DIM;
        let fc3_weight = data[off..off + 1 * FC2_DIM].to_vec(); off += 1 * FC2_DIM;
        let fc3_bias = data[off..off + 1].to_vec();
        Self { ft_weight, ft_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias }
    }

    pub fn validate(&self) -> bool {
        self.ft_weight.len() == 360 * FT_DIM
            && self.ft_bias.len() == FT_DIM
            && self.fc1_weight.len() == FC1_DIM * INPUT_DIM
            && self.fc1_bias.len() == FC1_DIM
            && self.fc2_weight.len() == FC2_DIM * FC1_DIM
            && self.fc2_bias.len() == FC2_DIM
            && self.fc3_weight.len() == 1 * FC2_DIM
            && self.fc3_bias.len() == 1
    }
}

/// Incremental NNUE accumulator for O(1) position updates.
#[derive(Debug, Clone)]
pub struct NNUEAccumulator {
    pub acc_stm: [f32; FT_DIM],
    pub acc_nstm: [f32; FT_DIM],
}

impl NNUEAccumulator {
    pub fn new(weights: &NNUEWeights) -> Self {
        let mut acc = Self {
            acc_stm: [0.0; FT_DIM],
            acc_nstm: [0.0; FT_DIM],
        };
        acc.reset(weights);
        acc
    }

    pub fn reset(&mut self, weights: &NNUEWeights) {
        for i in 0..FT_DIM {
            self.acc_stm[i] = weights.ft_bias[i];
            self.acc_nstm[i] = weights.ft_bias[i];
        }
    }

    /// Apply sparse features. Call once to initialize from game state.
    pub fn apply_sparse(&mut self, features: &[usize], stm_player: usize, weights: &NNUEWeights) {
        for &f in features {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            
            let row_start = f * FT_DIM;
            let wt_row = &weights.ft_weight[row_start..row_start + FT_DIM];
            
            if stm {
                for i in 0..FT_DIM {
                    self.acc_stm[i] += wt_row[i];
                }
            } else {
                for i in 0..FT_DIM {
                    self.acc_nstm[i] += wt_row[i];
                }
            }
        }
    }

    /// Incremental update: removed and added feature indices.
    pub fn apply_diff(&mut self, removed: &[usize], added: &[usize],
                       stm_player: usize, weights: &NNUEWeights) {
        for &f in removed {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let row_start = f * FT_DIM;
            let wt_row = &weights.ft_weight[row_start..row_start + FT_DIM];
            if stm {
                for i in 0..FT_DIM { self.acc_stm[i] -= wt_row[i]; }
            } else {
                for i in 0..FT_DIM { self.acc_nstm[i] -= wt_row[i]; }
            }
        }
        for &f in added {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let row_start = f * FT_DIM;
            let wt_row = &weights.ft_weight[row_start..row_start + FT_DIM];
            if stm {
                for i in 0..FT_DIM { self.acc_stm[i] += wt_row[i]; }
            } else {
                for i in 0..FT_DIM { self.acc_nstm[i] += wt_row[i]; }
            }
        }
    }

    /// Get CReLU'd concatenation (128-dim).
    pub fn get_crelu(&self) -> [f32; FT_DIM * 2] {
        let mut out = [0.0f32; FT_DIM * 2];
        for i in 0..FT_DIM {
            out[i] = self.acc_stm[i].max(0.0).min(127.0);
            out[FT_DIM + i] = self.acc_nstm[i].max(0.0).min(127.0);
        }
        out
    }
}

// ─── Matmul helpers (SIMD-friendly, cache-optimized) ────────

#[inline(always)]
fn matvec_mul_add(w: &[f32], x: &[f32], rows: usize, cols: usize,
                  bias: &[f32], out: &mut [f32]) {
    // y_j = sum_k(w[j][k] * x[k]) + bias[j]
    // Loop over k in outer loop to reuse x[k] across all j (cache-friendly)
    for j in 0..rows {
        out[j] = bias[j];
    }
    // Well-structured for auto-vectorization by LLVM
    for k in 0..cols {
        let xk = x[k];
        for j in 0..rows {
            out[j] += w[j * cols + k] * xk;
        }
    }
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
    }
}

/// Full NNUE evaluation from scratch (sparse → value).
pub fn nnue_evaluate(features: &[usize], dense: &[f32],
                      stm_player: usize, weights: &NNUEWeights) -> f32 {
    let mut acc = NNUEAccumulator::new(weights);
    acc.apply_sparse(features, stm_player, weights);
    evaluate_from_acc(&acc, dense, weights)
}

/// Evaluate from pre-computed accumulator (FC layers only).
pub fn evaluate_from_acc(acc: &NNUEAccumulator, dense: &[f32],
                          weights: &NNUEWeights) -> f32 {
    let crelu = acc.get_crelu(); // 128-dim

    // Concatenate crelu[128] + dense[66] → 194-dim
    let mut x = [0.0f32; INPUT_DIM];
    x[..FT_DIM * 2].copy_from_slice(&crelu);
    x[FT_DIM * 2..].copy_from_slice(dense);

    // FC1: 194 → 256
    let mut h1 = [0.0f32; FC1_DIM];
    matvec_mul_add(&weights.fc1_weight, &x, FC1_DIM, INPUT_DIM, &weights.fc1_bias, &mut h1);
    relu_inplace(&mut h1);

    // FC2: 256 → 128
    let mut h2 = [0.0f32; FC2_DIM];
    matvec_mul_add(&weights.fc2_weight, &h1, FC2_DIM, FC1_DIM, &weights.fc2_bias, &mut h2);
    relu_inplace(&mut h2);

    // FC3: 128 → 1
    let mut out = [0.0f32; 1];
    matvec_mul_add(&weights.fc3_weight, &h2, 1, FC2_DIM, &weights.fc3_bias, &mut out);

    // tanh
    out[0].tanh()
}

/// Batch evaluate: (features, dense, stm_player) tuples → values.
/// Each tuple is encoded as: sparse_len, [sparse_idx; sparse_len], 66 dense, stm_player
pub fn evaluate_batch(
    batch: &[(Vec<usize>, Vec<f32>, usize)],
    weights: &NNUEWeights,
) -> Vec<f32> {
    let mut results = Vec::with_capacity(batch.len());
    for (sparse, dense, stm) in batch {
        results.push(nnue_evaluate(sparse, dense, *stm, weights));
    }
    results
}

/// Same as evaluate_batch but using pre-built accumulators (one per sample).
pub fn evaluate_batch_from_acc(
    accs: &[NNUEAccumulator],
    dense_batch: &[Vec<f32>],
    weights: &NNUEWeights,
) -> Vec<f32> {
    let mut results = Vec::with_capacity(accs.len());
    for (acc, dense) in accs.iter().zip(dense_batch.iter()) {
        results.push(evaluate_from_acc(acc, dense, weights));
    }
    results
}
