use numpy::*;
use pyo3::prelude::*;

/// Rust implementation of binary mask to RLE
#[pyfunction]
fn arr2rle(_py: Python<'_>, arr: &PyArray1<bool>) -> Vec<u32> {
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

/// Rust implementation of binary mask to RLE
#[pyfunction]
fn f_order_mask2rle(_py: Python<'_>, arr: &PyArray2<bool>) -> Vec<u32> {
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

/// Rust implementation of binary mask to RLE
#[pyfunction]
fn c_order_mask2rle(_py: Python<'_>, arr: &PyArray2<bool>) -> Vec<u32> {
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

/// Rust implementation of binary mask to RLE
#[pyfunction]
fn thread_arr2rle(py: Python<'_>, arr: &PyArray1<bool>) -> Vec<u32> {
    let arr = arr.readonly();
    let arr = arr.as_array();

    py.allow_threads(move || {
        let mut counts: Vec<u32> = vec![];
        let mut count: u32 = 0;
        let mut tgt = false;

        for curr in arr.iter() {
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
    })
}

/// Rust implementation of binary mask to RLE
#[pyfunction]
fn thread_f_order_mask2rle(py: Python<'_>, arr: &PyArray2<bool>) -> Vec<u32> {
    let arr = arr.readonly();
    let arr = arr.as_array();

    py.allow_threads(move || {
        let mut counts: Vec<u32> = vec![];
        let mut count: u32 = 0;
        let mut tgt = false;

        for col in arr.axis_iter(ndarray::Axis(1)) {
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
    })
}

/// Rust implementation of binary mask to RLE
#[pyfunction]
fn thread_c_order_mask2rle(py: Python<'_>, arr: &PyArray2<bool>) -> Vec<u32> {
    let arr = arr.readonly();
    let arr = arr.as_array();

    py.allow_threads(move || {
        let mut counts: Vec<u32> = vec![];
        let mut count: u32 = 0;
        let mut tgt = false;

        for curr in arr.iter() {
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
    })
}



/// A Python module implemented in Rust.
#[pymodule]
fn py_multi_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(arr2rle, m)?)?;
    m.add_function(wrap_pyfunction!(f_order_mask2rle, m)?)?;
    m.add_function(wrap_pyfunction!(c_order_mask2rle, m)?)?;
    m.add_function(wrap_pyfunction!(thread_arr2rle, m)?)?;
    m.add_function(wrap_pyfunction!(thread_f_order_mask2rle, m)?)?;
    m.add_function(wrap_pyfunction!(thread_c_order_mask2rle, m)?)?;
    Ok(())
}
