/// Full NNUE training via Candle (correct Adam, all params trained).
use candle_core::{Device, Tensor, Var};
use crate::nnue_rs::*;
use crate::nnue_train::TrainingRecord;

/// Convert sparse features + stm to one-hot vectors for stm and nstm.
/// Returns (stm_onehot, nstm_onehot) each as Vec<f32> length 360.
fn sparse_to_onehot(rec: &TrainingRecord) -> (Vec<f32>, Vec<f32>) {
    let mut stm = vec![0.0f32; 360];
    let mut nstm = vec![0.0f32; 360];
    for &f in &rec.sparse {
        if f < 0 || f >= 360 { continue; }
        let is_p1 = f < 180;
        let is_stm = (rec.stm == 0 && is_p1) || (rec.stm == 1 && !is_p1);
        if is_stm { stm[f as usize] = 1.0; }
        else { nstm[f as usize] = 1.0; }
    }
    (stm, nstm)
}

fn wt_to_std(wt: &[f32], o: usize, i: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; o * i];
    for r in 0..o { for c in 0..i { d[r * i + c] = wt[c * o + r]; } }
    d
}
fn wt_from_std(s: &[f32], o: usize, i: usize, dst: &mut [f32]) {
    for r in 0..o { for c in 0..i { dst[c * o + r] = s[r * i + c]; } }
}

/// Train ALL NNUE parameters (FT + FC) using Candle autograd + correct Adam.
pub fn train_all(
    records: &[TrainingRecord], w: &mut NNUEWeights, lr: f32, wd: f32, bs: usize, epochs: usize,
) -> f32 {
    let dev = Device::Cpu;
    let n = records.len();
    let bsz = bs.min(n);
    let ft_dim = FT_DIM;

    let mkvar = |d: Vec<f32>, s: (usize, usize)| -> Var {
        Var::from_tensor(&Tensor::from_slice(&d, (d.len(),), &dev).unwrap().reshape(s).unwrap()).unwrap()
    };
    let mkvar1 = |d: Vec<f32>, s: usize| -> Var {
        Var::from_tensor(&Tensor::from_slice(&d, (d.len(),), &dev).unwrap().reshape(s).unwrap()).unwrap()
    };

    // FT: (360, FT_DIM) weight + (FT_DIM,) bias  
    let ft_w = mkvar1(w.ft_weight.clone(), 360 * ft_dim);
    let ft_b = mkvar1(w.ft_bias.clone(), ft_dim);
    let v1w = mkvar(wt_to_std(&w.fc1_weight_t, FC1_DIM, INPUT_DIM), (FC1_DIM, INPUT_DIM));
    let v1b = mkvar1(w.fc1_bias.clone(), FC1_DIM);
    let v2w = mkvar(wt_to_std(&w.fc2_weight_t, FC2_DIM, FC1_DIM), (FC2_DIM, FC1_DIM));
    let v2b = mkvar1(w.fc2_bias.clone(), FC2_DIM);
    let v3w = mkvar(wt_to_std(&w.fc3_weight_t, 1, FC2_DIM), (1, FC2_DIM));
    let v3b = mkvar1(w.fc3_bias.clone(), 1);

    let all_vars: [&Var; 8] = [&ft_w, &ft_b, &v1w, &v1b, &v2w, &v2b, &v3w, &v3b];
    let sizes: Vec<usize> = all_vars.iter().map(|v| v.as_tensor().elem_count()).collect();
    let mut m: Vec<Vec<f32>> = sizes.iter().map(|&s| vec![0.0f32; s]).collect();
    let mut v: Vec<Vec<f32>> = sizes.iter().map(|&s| vec![0.0f32; s]).collect();

    let b1 = 0.9f32; let b2 = 0.999f32; let ep = 1e-8f32;
    let mut tstep = 0u32; let mut total = 0.0f32;

    for _ in 0..epochs {
        for b in (0..n).step_by(bsz) {
            let end = (b + bsz).min(n);
            let sz = end - b;

            // Build batch tensors
            let mut stm_oh = vec![0.0f32; sz * 360];
            let mut nstm_oh = vec![0.0f32; sz * 360];
            let mut dense = vec![0.0f32; sz * DENSE_DIM];
            let mut labels = vec![0.0f32; sz];

            for i in 0..sz {
                let r = &records[b + i];
                let (s, n) = sparse_to_onehot(r);
                for j in 0..360 { stm_oh[i * 360 + j] = s[j]; nstm_oh[i * 360 + j] = n[j]; }
                for j in 0..DENSE_DIM { dense[i * DENSE_DIM + j] = r.dense[j]; }
                labels[i] = r.label;
            }

            // FT: ft_out = onehot @ ft_weight + ft_bias  → (B, FT_DIM)
            let stm_t = Tensor::from_slice(&stm_oh, (sz * 360,), &dev).unwrap().reshape((sz, 360)).unwrap();
            let nstm_t = Tensor::from_slice(&nstm_oh, (sz * 360,), &dev).unwrap().reshape((sz, 360)).unwrap();
            let ft_w_t = ft_w.as_tensor().reshape((360, ft_dim)).unwrap();
            let stm_acc = stm_t.matmul(&ft_w_t).unwrap().broadcast_add(&ft_b.as_tensor()).unwrap();
            let nstm_acc = nstm_t.matmul(&ft_w_t).unwrap().broadcast_add(&ft_b.as_tensor()).unwrap();

            // CReLU: clamp [0, 127], scale to [-1, 1], concat with dense
            let scale = 2.0 / 127.0;
            let stm_cl = stm_acc.clamp(0.0f32, 127.0f32).unwrap();
            let nstm_cl = nstm_acc.clamp(0.0f32, 127.0f32).unwrap();
            let stm_cr = ((stm_cl * scale).unwrap() - 1.0).unwrap();
            let nstm_cr = ((nstm_cl * scale).unwrap() - 1.0).unwrap();
            let dense_t = Tensor::from_slice(&dense, (sz * DENSE_DIM,), &dev).unwrap().reshape((sz, DENSE_DIM)).unwrap();
            let xs = Tensor::cat(&[&stm_cr, &nstm_cr, &dense_t], 1).unwrap();

            // FC layers
            let ys = Tensor::from_slice(&labels, (sz,), &dev).unwrap().reshape((sz, 1)).unwrap();
            let h1 = xs.matmul(&v1w.as_tensor().t().unwrap()).unwrap().broadcast_add(&v1b.as_tensor()).unwrap().relu().unwrap();
            let h2 = h1.matmul(&v2w.as_tensor().t().unwrap()).unwrap().broadcast_add(&v2b.as_tensor()).unwrap().relu().unwrap();
            let out = h2.matmul(&v3w.as_tensor().t().unwrap()).unwrap().broadcast_add(&v3b.as_tensor()).unwrap().tanh().unwrap();
            let loss = (out - &ys).unwrap().sqr().unwrap().mean_all().unwrap();
            total += loss.to_scalar::<f32>().unwrap() * sz as f32;

            let grads = loss.backward().unwrap();
            tstep += 1;
            let b1t = 1.0 - b1.powi(tstep as i32);
            let b2t = 1.0 - b2.powi(tstep as i32);

            for i in 0..8 {
                let gt = if let Some(g) = grads.get(all_vars[i].as_tensor()) { g.clone() } else { continue; };
                let gv: Vec<f32> = gt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let cur: Vec<f32> = all_vars[i].as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let len = gv.len();
                let mut new = Vec::with_capacity(len);
                for j in 0..len {
                    m[i][j] = b1 * m[i][j] + (1.0 - b1) * gv[j];
                    v[i][j] = b2 * v[i][j] + (1.0 - b2) * gv[j] * gv[j];
                    let mh = m[i][j] / b1t; let vh = v[i][j] / b2t;
                    new.push(cur[j] - lr * mh / (vh.sqrt() + ep) + lr * wd * cur[j]);
                }
                let nt = Tensor::from_slice(&new, (new.len(),), &dev).unwrap().reshape(all_vars[i].as_tensor().shape()).unwrap();
                all_vars[i].set(&nt).unwrap();
            }
        }
    }

    // Save all weights back
    let fft: Vec<f32> = ft_w.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    w.ft_weight.copy_from_slice(&fft);
    let ffb: Vec<f32> = ft_b.as_tensor().to_vec1::<f32>().unwrap();
    w.ft_bias.copy_from_slice(&ffb);

    let f1: Vec<f32> = v1w.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    wt_from_std(&f1, FC1_DIM, INPUT_DIM, &mut w.fc1_weight_t);
    let b1v: Vec<f32> = v1b.as_tensor().to_vec1::<f32>().unwrap();
    w.fc1_bias.copy_from_slice(&b1v);
    let f2: Vec<f32> = v2w.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    wt_from_std(&f2, FC2_DIM, FC1_DIM, &mut w.fc2_weight_t);
    let b2v: Vec<f32> = v2b.as_tensor().to_vec1::<f32>().unwrap();
    w.fc2_bias.copy_from_slice(&b2v);
    let f3: Vec<f32> = v3w.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    wt_from_std(&f3, 1, FC2_DIM, &mut w.fc3_weight_t);
    let b3v: Vec<f32> = v3b.as_tensor().to_vec1::<f32>().unwrap();
    w.fc3_bias.copy_from_slice(&b3v);

    total / n as f32
}
