pub mod engine;
pub mod server;
pub mod sharding;

use crate::engine::{GeodesicEngine, MAX_RECALL_BYTES, MAX_VALUE_BYTES};
use pyo3::prelude::*;
use std::ffi::{CStr, c_char};
use std::ptr;
use std::slice;
use std::sync::Mutex;

#[pymodule]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeodesicEngine>()?;
    Ok(())
}

#[pyclass]
struct PyGeodesicEngine {
    inner: GeodesicEngine,
}

#[pymethods]
impl PyGeodesicEngine {
    #[new]
    fn new(path: String, size_mb: u64) -> PyResult<Self> {
        let engine = GeodesicEngine::new(path, size_mb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyGeodesicEngine { inner: engine })
    }

    fn write(&mut self, token_id: String, value: Vec<u8>) -> PyResult<u64> {
        self.inner
            .write(&token_id, value)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    // New Binding for Hybrid Search
    fn write_with_vector(
        &mut self,
        token_id: String,
        value: Vec<u8>,
        vector: Vec<f32>,
    ) -> PyResult<u64> {
        self.inner
            .write_with_vector(&token_id, value, vector)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn read_latest(&self, token_id: String) -> PyResult<Option<Vec<u8>>> {
        if let Some(node) = self.inner.read_latest(&token_id) {
            Ok(Some(node.value))
        } else {
            Ok(None)
        }
    }

    fn recall(&self, token_id: String, depth: usize) -> PyResult<Vec<Vec<u8>>> {
        let nodes = self.inner.recall(&token_id, depth);
        Ok(nodes.into_iter().map(|n| n.value).collect())
    }

    // New Binding for Similarity Search
    fn search_similar(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<Vec<u8>>> {
        let nodes = self.inner.search_similar(vector, k);
        Ok(nodes.into_iter().map(|n| n.value).collect())
    }

    // Similarity search that also returns the cosine distance per hit
    // (smaller = closer) so callers can threshold on relevance.
    fn search_similar_scored(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<(f32, Vec<u8>)>> {
        Ok(self
            .inner
            .search_similar_scored(vector, k)
            .into_iter()
            .map(|(distance, node)| (distance, node.value))
            .collect())
    }

    // Defer per-write metadata fsync (huge throughput win for bulk ingestion);
    // call flush() to persist. See GeodesicEngine::set_sync_on_write.
    fn set_sync_on_write(&mut self, sync_on_write: bool) {
        self.inner.set_sync_on_write(sync_on_write);
    }

    fn flush(&mut self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }
}

pub struct OxtaMemHandle {
    inner: Mutex<GeodesicEngine>,
}

fn write_allocated_buffer(bytes: Vec<u8>, out_data: *mut *mut u8, out_len: *mut usize) -> bool {
    if out_data.is_null() || out_len.is_null() {
        return false;
    }

    let len = bytes.len();
    let mut boxed = bytes.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        *out_data = ptr;
        *out_len = len;
    }
    true
}

fn serialize_nodes(values: Vec<Vec<u8>>) -> Option<Vec<u8>> {
    let mut total = sizeof_u64();
    for value in &values {
        total = total.checked_add(sizeof_u64())?.checked_add(value.len())?;
        if total > MAX_RECALL_BYTES {
            return None;
        }
    }
    let mut encoded = Vec::with_capacity(total);
    encoded.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        encoded.extend_from_slice(&(value.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&value);
    }
    Some(encoded)
}

const fn sizeof_u64() -> usize {
    std::mem::size_of::<u64>()
}

fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_str().ok().map(|s| s.to_string())
}

/// # Safety
///
/// `path` must be a valid, NUL-terminated C string pointer for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxtamem_create(path: *const c_char, size_mb: u64) -> *mut OxtaMemHandle {
    let Some(path_string) = cstr_to_string(path) else {
        return ptr::null_mut();
    };

    match GeodesicEngine::new(path_string, size_mb) {
        Ok(engine) => Box::into_raw(Box::new(OxtaMemHandle {
            inner: Mutex::new(engine),
        })),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
///
/// `handle` must be a pointer returned by `oxtamem_create` and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxtamem_destroy(handle: *mut OxtaMemHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
///
/// `handle` must be valid. `token_id` must be a valid C string. `value` must point to
/// `value_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxtamem_write(
    handle: *mut OxtaMemHandle,
    token_id: *const c_char,
    value: *const u8,
    value_len: usize,
) -> bool {
    if handle.is_null() || value.is_null() || value_len > MAX_VALUE_BYTES {
        return false;
    }

    let Some(token) = cstr_to_string(token_id) else {
        return false;
    };
    let payload = unsafe { slice::from_raw_parts(value, value_len) }.to_vec();
    let guard = unsafe { &*handle };
    match guard.inner.lock() {
        Ok(mut engine) => engine.write(&token, payload).is_ok(),
        Err(_) => false,
    }
}

/// # Safety
///
/// `handle` must be valid. `token_id` must be a valid C string. `out_data` and `out_len`
/// must be valid writable pointers. Returned buffers must be released with
/// `oxtamem_free_buffer`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxtamem_read_latest(
    handle: *mut OxtaMemHandle,
    token_id: *const c_char,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let Some(token) = cstr_to_string(token_id) else {
        return false;
    };
    let guard = unsafe { &*handle };
    let Ok(engine) = guard.inner.lock() else {
        return false;
    };

    match engine.read_latest(&token) {
        Some(node) => write_allocated_buffer(node.value, out_data, out_len),
        None => false,
    }
}

/// # Safety
///
/// `handle` must be valid. `token_id` must be a valid C string. `out_data` and `out_len`
/// must be valid writable pointers. Returned buffers must be released with
/// `oxtamem_free_buffer`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxtamem_recall(
    handle: *mut OxtaMemHandle,
    token_id: *const c_char,
    depth: usize,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let Some(token) = cstr_to_string(token_id) else {
        return false;
    };
    let guard = unsafe { &*handle };
    let Ok(engine) = guard.inner.lock() else {
        return false;
    };

    let payloads = engine
        .recall(&token, depth)
        .into_iter()
        .map(|node| node.value)
        .collect::<Vec<_>>();
    match serialize_nodes(payloads) {
        Some(encoded) => write_allocated_buffer(encoded, out_data, out_len),
        None => false,
    }
}

/// # Safety
///
/// `data` and `len` must exactly match a buffer returned by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxtamem_free_buffer(data: *mut u8, len: usize) {
    if data.is_null() {
        return;
    }

    unsafe {
        drop(Vec::from_raw_parts(data, len, len));
    }
}
