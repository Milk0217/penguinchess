#[cfg(feature = "ort")]
mod inner {
    use ndarray::{Array, Axis};
    use ort::{ep, inputs, session::Session, value::Value};
    use std::sync::Mutex;

    pub struct NetInfer {
        session: Mutex<Session>,
    }

    impl NetInfer {
        pub fn new(model_path: &str) -> Self {
            let session = Session::builder()
                .unwrap()
                .with_execution_providers([ep::CUDA::default().build(), ep::CPU::default().build()])
                .unwrap()
                .commit_from_file(model_path)
                .unwrap();
            NetInfer { session: Mutex::new(session) }
        }

        pub fn evaluate_batch(
            &self, obs_buf: &[f32], n: usize,
        ) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
            let obs_array = Array::from_shape_vec(ndarray::IxDyn(&[n, 206]), obs_buf.to_vec())?;
            let mut session = self.session.lock().unwrap();
            let output_values = session.run(inputs! { "obs" => Value::from_array(obs_array)? })?;
            let logits_arr = output_values[0]
                .try_extract_array::<f32>()?
                .to_owned().into_dimensionality::<ndarray::Ix2>()?;
            let val_arr = output_values[1]
                .try_extract_array::<f32>()?
                .to_owned().into_dimensionality::<ndarray::Ix1>()?;
            let logits_matrix = logits_arr.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
            let values_vec = val_arr.to_vec();
            Ok((logits_matrix, values_vec))
        }
    }
}

#[cfg(feature = "ort")]
pub use inner::NetInfer;
