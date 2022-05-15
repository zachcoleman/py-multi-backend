use numpy::*;
use pyo3::prelude::*;

/// Rust implementation of binary msk to RLE
#[pyfunction]
fn rust_mask_to_rle(_py: Python<'_>, arr: &PyArray1<bool>) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![];
    let mut count: u32 = 0;
    let mut tgt = false;

    unsafe {
        for curr in arr.as_array().iter() {
            if *curr == tgt {
                count = count + 1;
            } else {
                counts.push(count);
                tgt = *curr;
                count = 1;
            }
        }
    }
    counts.push(count);
    return counts;
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_multi_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_mask_to_rle, m)?)?;
    Ok(())
}
