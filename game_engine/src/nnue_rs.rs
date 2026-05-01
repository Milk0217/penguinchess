/// NNUE forward pass — SIMD-accelerated with explicit AVX2 intrinsics.
/// Architecture:
///   360 sparse binary → FT(64) → CReLU(acc_stm || acc_nstm) → concat dense[66]
///   → FC1(256) → ReLU → FC2(128) → ReLU → FC3(1) → tanh
///
/// MCTS version (larger):
///   360 sparse → FT(128) → CReLU → concat dense → FC1(512) → (value:FC1→1, policy:FC1→60)
///
/// FC weights stored TRANSPOSED (column-major) for contiguous SIMD access:
///   w_t[col * rows + row] = original[row][col]

use serde::{Deserialize, Serialize};

// Value-only NNUE (gen_2, AB search)
pub const FT_DIM: usize = 64;
pub const DENSE_DIM: usize = 66;
pub const FC1_DIM: usize = 256;
pub const FC2_DIM: usize = 128;
pub const INPUT_DIM: usize = FT_DIM * 2 + DENSE_DIM; // 194
pub const P1_CUTOFF: usize = 180;

// MCTS NNUE (larger model)
pub const MCTS_FT_DIM: usize = 128;
pub const MCTS_FC1_DIM: usize = 512;
pub const MCTS_INPUT_DIM: usize = MCTS_FT_DIM * 2 + DENSE_DIM; // 322

// ─── AVX2 helpers (requires target-cpu=native) ────────────────

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;

    /// y[j..j+8] += col[j..j+8] * xk  with FMA
    #[inline(always)]
    pub unsafe fn fma_add(col: *const f32, xk: __m256, y: *mut f32, j: usize) {
        let w = _mm256_loadu_ps(col.add(j));
        let o = _mm256_loadu_ps(y.add(j));
        let r = _mm256_fmadd_ps(w, xk, o);
        _mm256_storeu_ps(y.add(j), r);
    }

    /// vec[j..j+8] += row[j..j+8]
    #[inline(always)]
    pub unsafe fn add_vec(vec: *mut f32, row: *const f32, j: usize) {
        let v = _mm256_loadu_ps(vec.add(j));
        let r = _mm256_loadu_ps(row.add(j));
        _mm256_storeu_ps(vec.add(j), _mm256_add_ps(v, r));
    }

    /// vec[j..j+8] -= row[j..j+8]
    #[inline(always)]
    pub unsafe fn sub_vec(vec: *mut f32, row: *const f32, j: usize) {
        let v = _mm256_loadu_ps(vec.add(j));
        let r = _mm256_loadu_ps(row.add(j));
        _mm256_storeu_ps(vec.add(j), _mm256_sub_ps(v, r));
    }

    /// relu: x[j..j+8] = max(x[j..j+8], 0)
    #[inline(always)]
    pub unsafe fn relu8(x: *mut f32, j: usize) {
        let v = _mm256_loadu_ps(x.add(j));
        let zero = _mm256_setzero_ps();
        _mm256_storeu_ps(x.add(j), _mm256_max_ps(v, zero));
    }

    /// clip to [0, 127]
    #[inline(always)]
    pub unsafe fn clip8(x: *const f32, j: usize) -> __m256 {
        let v = _mm256_loadu_ps(x.add(j));
        let zero = _mm256_setzero_ps();
        let maxv = _mm256_set1_ps(127.0);
        _mm256_min_ps(_mm256_max_ps(v, zero), maxv)
    }

    /// set1(xk)
    #[inline(always)]
    pub unsafe fn splat(xk: f32) -> __m256 {
        _mm256_set1_ps(xk)
    }

    /// Store to y[]
    #[inline(always)]
    pub unsafe fn store(y: *mut f32, j: usize, v: __m256) {
        _mm256_storeu_ps(y.add(j), v);
    }
}

/// Scalar fallback (always available).
#[allow(dead_code)]
fn matvec_scalar(w_t: &[f32], x: &[f32], rows: usize, cols: usize,
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

fn matvec_mul_add_t(w_t: &[f32], x: &[f32], rows: usize, cols: usize,
                     bias: &[f32], out: &mut [f32]) {
    out.copy_from_slice(bias);
    for k in 0..cols {
        let xk = x[k];
        let col = &w_t[k * rows..(k + 1) * rows];
        let mut j = 0usize;
        #[cfg(target_arch = "x86_64")]
        {
            let w_ptr = col.as_ptr();
            let o_ptr = out.as_mut_ptr();
            while j + 8 <= rows {
                unsafe { simd::fma_add(w_ptr, simd::splat(xk), o_ptr, j); }
                j += 8;
            }
        }
        for j in j..rows {
            out[j] += col[j] * xk;
        }
    }
}

fn relu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let p = x.as_mut_ptr();
        let mut j = 0usize;
        while j + 8 <= x.len() {
            simd::relu8(p, j);
            j += 8;
        }
        for i in j..x.len() {
            if x[i] < 0.0 { x[i] = 0.0; }
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    for v in x.iter_mut() { if *v < 0.0 { *v = 0.0; } }
}

// ─── Weights ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNUEWeights {
    pub ft_weight: Vec<f32>,     // (360, 64) row-major
    pub ft_bias: Vec<f32>,
    pub fc1_weight_t: Vec<f32>,  // (194, 256) TRANSPOSED
    pub fc1_bias: Vec<f32>,
    pub fc2_weight_t: Vec<f32>,  // (256, 128) TRANSPOSED
    pub fc2_bias: Vec<f32>,
    pub fc3_weight_t: Vec<f32>,  // (128, 1)
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

    pub fn from_flat(data: &[f32]) -> Self {
        let mut off = 0;
        let ft_weight = data[off..off + 360 * FT_DIM].to_vec(); off += 360 * FT_DIM;
        let ft_bias = data[off..off + FT_DIM].to_vec(); off += FT_DIM;

        let fc1_rows = FC1_DIM; let fc1_cols = INPUT_DIM;
        let mut fc1_weight_t = vec![0.0f32; fc1_cols * fc1_rows];
        for row in 0..fc1_rows { for col in 0..fc1_cols {
            fc1_weight_t[col * fc1_rows + row] = data[off + row * fc1_cols + col];
        }}
        off += fc1_rows * fc1_cols;
        let fc1_bias = data[off..off + FC1_DIM].to_vec(); off += FC1_DIM;

        let fc2_rows = FC2_DIM; let fc2_cols = FC1_DIM;
        let mut fc2_weight_t = vec![0.0f32; fc2_cols * fc2_rows];
        for row in 0..fc2_rows { for col in 0..fc2_cols {
            fc2_weight_t[col * fc2_rows + row] = data[off + row * fc2_cols + col];
        }}
        off += fc2_rows * fc2_cols;
        let fc2_bias = data[off..off + FC2_DIM].to_vec(); off += FC2_DIM;

        let fc3_rows = 1usize; let fc3_cols = FC2_DIM;
        let mut fc3_weight_t = vec![0.0f32; fc3_cols * fc3_rows];
        for row in 0..fc3_rows { for col in 0..fc3_cols {
            fc3_weight_t[col * fc3_rows + row] = data[off + row * fc3_cols + col];
        }}
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

    fn add_ft_weight(dst: &mut [f32; 64], wt_row: &[f32], sign: f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let d = dst.as_mut_ptr();
            let s = wt_row.as_ptr();
            let mut j = 0usize;
            if sign > 0.0 {
                while j + 8 <= 64 { simd::add_vec(d, s, j); j += 8; }
            } else {
                while j + 8 <= 64 { simd::sub_vec(d, s, j); j += 8; }
            }
            for i in j..64 { dst[i] += sign * wt_row[i]; }
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        for i in 0..64 { dst[i] += sign * wt_row[i]; }
    }

    pub fn apply_sparse(&mut self, features: &[usize], stm_player: usize, weights: &NNUEWeights) {
        for &f in features {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let src = &weights.ft_weight[f * FT_DIM..(f + 1) * FT_DIM];
            let dst: &mut [f32; 64] = if stm { &mut self.acc_stm } else { &mut self.acc_nstm };
            Self::add_ft_weight(dst, src, 1.0);
        }
    }

    pub fn apply_diff(&mut self, removed: &[usize], added: &[usize],
                       stm_player: usize, weights: &NNUEWeights) {
        for &f in removed {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let src = &weights.ft_weight[f * FT_DIM..(f + 1) * FT_DIM];
            let dst: &mut [f32; 64] = if stm { &mut self.acc_stm } else { &mut self.acc_nstm };
            Self::add_ft_weight(dst, src, -1.0);
        }
        for &f in added {
            let is_p1 = f < P1_CUTOFF;
            let stm = if stm_player == 0 { is_p1 } else { !is_p1 };
            let src = &weights.ft_weight[f * FT_DIM..(f + 1) * FT_DIM];
            let dst: &mut [f32; 64] = if stm { &mut self.acc_stm } else { &mut self.acc_nstm };
            Self::add_ft_weight(dst, src, 1.0);
        }
    }

    pub fn get_crelu(&self) -> [f32; FT_DIM * 2] {
        #[cfg(target_arch = "x86_64")]
        {
            let mut out = [0.0f32; FT_DIM * 2];
            unsafe {
                let mut j = 0usize;
                while j + 8 <= FT_DIM {
                    let sv = simd::clip8(self.acc_stm.as_ptr(), j);
                    let nv = simd::clip8(self.acc_nstm.as_ptr(), j);
                    simd::store(out.as_mut_ptr(), j, sv);
                    simd::store(out.as_mut_ptr(), FT_DIM + j, nv);
                    j += 8;
                }
                for i in j..FT_DIM {
                    out[i] = self.acc_stm[i].max(0.0).min(127.0);
                    out[FT_DIM + i] = self.acc_nstm[i].max(0.0).min(127.0);
                }
            }
            return out;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut out = [0.0f32; FT_DIM * 2];
            for i in 0..FT_DIM {
                out[i] = self.acc_stm[i].max(0.0).min(127.0);
                out[FT_DIM + i] = self.acc_nstm[i].max(0.0).min(127.0);
            }
            out
        }
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

// ─── NNUE MCTS Weights (value + policy heads) — larger architecture ─

pub const FC2V_DIM: usize = 1;
pub const FC2P_DIM: usize = 60;

#[derive(Debug, Clone)]
pub struct NNUEMCTSWeights {
    pub ft_weight: Vec<f32>,
    pub ft_bias: Vec<f32>,
    pub fc1_weight_t: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2v_weight: Vec<f32>,    // (MCTS_FC1_DIM, 1) row-major
    pub fc2v_bias: Vec<f32>,
    pub fc2p_weight_row: Vec<f32>, // (MCTS_FC1_DIM, 60) row-major
    pub fc2p_bias: Vec<f32>,
}

impl NNUEMCTSWeights {
    pub fn new_empty() -> Self {
        Self {
            ft_weight: vec![0.0; 360 * MCTS_FT_DIM],
            ft_bias: vec![0.0; MCTS_FT_DIM],
            fc1_weight_t: vec![0.0; MCTS_INPUT_DIM * MCTS_FC1_DIM],
            fc1_bias: vec![0.0; MCTS_FC1_DIM],
            fc2v_weight: vec![0.0; MCTS_FC1_DIM],
            fc2v_bias: vec![0.0; 1],
            fc2p_weight_row: vec![0.0; MCTS_FC1_DIM * FC2P_DIM],
            fc2p_bias: vec![0.0; FC2P_DIM],
        }
    }
}

/// Fast MCTS NNUE eval: FT → FC1 → [value(1), policy(60)].
pub fn nnue_evaluate_mcts(
    sparse: &[usize], dense: &[f32], stm: usize, w: &NNUEMCTSWeights,
) -> (Vec<f32>, f32) {
    let mut h = vec![0.0f32; MCTS_FC1_DIM];

    // FT accumulation (uses MCTS_FT_DIM)
    let mut acc_stm = w.ft_bias.clone();
    let mut acc_nstm = w.ft_bias.clone();
    for &idx in sparse {
        let base = idx * MCTS_FT_DIM;
        let is_p1 = idx < P1_CUTOFF;
        let target = if (stm == 0 && is_p1) || (stm == 1 && !is_p1) { &mut acc_stm } else { &mut acc_nstm };
        for i in 0..MCTS_FT_DIM { target[i] += w.ft_weight[base + i]; }
    }

    // CReLU + concat dense → FC1 + ReLU
    let mut x = Vec::with_capacity(MCTS_FT_DIM * 2 + DENSE_DIM);
    for i in 0..MCTS_FT_DIM { x.push((acc_stm[i].clamp(0.0, 127.0)) * 2.0 / 127.0 - 1.0); }
    for i in 0..MCTS_FT_DIM { x.push((acc_nstm[i].clamp(0.0, 127.0)) * 2.0 / 127.0 - 1.0); }
    x.extend_from_slice(dense);
    matvec_mul_add_t(&w.fc1_weight_t, &x, MCTS_FC1_DIM, MCTS_INPUT_DIM, &w.fc1_bias, &mut h);
    relu_inplace(&mut h);

    // Value head: MCTS_FC1_DIM → 1
    let mut val = w.fc2v_bias[0];
    for i in 0..MCTS_FC1_DIM { val += h[i] * w.fc2v_weight[i]; }
    let value = val.tanh();

    // Policy head: MCTS_FC1_DIM → 60 (row-major)
    let mut logits = w.fc2p_bias.clone();
    for i in 0..FC2P_DIM {
        for j in 0..MCTS_FC1_DIM {
            logits[i] += h[j] * w.fc2p_weight_row[j * FC2P_DIM + i];
        }
    }

    (logits, value)
}
