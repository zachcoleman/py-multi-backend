use numpy::*;
use pyo3::prelude::*;

/// Rust implementation of binary msk to RLE
#[pyfunction]
fn rust_1D_mask_to_rle(_py: Python<'_>, arr: &PyArray1<bool>) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![];
    let mut count: u32 = 0;
    let mut tgt = false;

    for curr in arr.readonly().as_array().iter() {
        if *curr == tgt {
            count = count + 1;
        } else {
            counts.push(count);
            tgt = *curr;
            count = 1;
        }
    }

    counts.push(count);
    return counts;
}

/// Rust implementation of binary msk to RLE
#[pyfunction]
fn rust_f_order_2D_mask_to_rle(_py: Python<'_>, arr: &PyArray2<bool>) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![];
    let mut count: u32 = 0;
    let mut tgt = false;

    for col in arr.readonly().as_array().axis_iter(ndarray::Axis(1)) {
        for curr in col.iter(){
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

#[pyfunction]
fn rust_c_order_2D_mask_to_rle(_py: Python<'_>, arr: &PyArray2<bool>) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![];
    let mut count: u32 = 0;
    let mut tgt = false;

    for curr in arr.readonly().as_array().iter() {
        if *curr == tgt {
            count = count + 1;
        } else {
            counts.push(count);
            tgt = *curr;
            count = 1;
        }
    }

    counts.push(count);
    return counts;
}


/// Rust implementation of binary msk to RLE
// #[pyfunction]
// fn rust_allow_threads_mask_to_rle(py: Python<'_>, arr: &PyArray1<bool>) -> Vec<u32> {
//     py.allow_threads(move || {

//         let mut counts: Vec<u32> = vec![];
//         let mut count: u32 = 0;
//         let mut tgt = false;

//         for curr in arr.readonly().as_array().clone().iter() {
//             if *curr == tgt {
//                 count = count + 1;
//             } else {
//                 counts.push(count);
//                 tgt = *curr;
//                 count = 1;
//             }
//         }

//         counts.push(count);
//         counts
//     })
// }

/// A Python module implemented in Rust.
#[pymodule]
fn py_multi_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_1D_mask_to_rle, m)?)?;
    m.add_function(wrap_pyfunction!(rust_f_order_2D_mask_to_rle, m)?)?;
    m.add_function(wrap_pyfunction!(rust_c_order_2D_mask_to_rle, m)?)?;
    Ok(())
}
