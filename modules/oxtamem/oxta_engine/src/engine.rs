use memmap2::MmapMut;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub const DEFAULT_VECTOR_DIMENSIONS: usize = 128;
const MIN_ARENA_MB: u64 = 1;
const MAX_ARENA_MB: u64 = 32 * 1024;
pub const MAX_TOKEN_ID_BYTES: usize = 1024;
pub const MAX_VALUE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_RECALL_DEPTH: usize = 1024;
pub const MAX_RECALL_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_SEARCH_RESULTS: usize = 4096;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_HEADS: usize = 1_000_000;
const MAX_VECTOR_RECORDS: usize = 100_000;
const MAX_ARCHIVED_NODE_BYTES: u64 = MAX_VALUE_BYTES as u64 + 1024 * 1024;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
// rkyv 0.8 compatibility
pub struct Node {
    pub value: Vec<u8>,
    pub prev: Option<u64>, // Offset in the file
    pub timestamp: u64,
}

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct EngineMetadata {
    pub heads: HashMap<String, u64>,
    pub current_offset: u64,
    pub vector_records: HashMap<u64, Vec<f32>>,
}

pub struct GeodesicEngine {
    pub heads: HashMap<String, u64>, // Maps Variable ID -> Offset in File
    #[allow(dead_code)] // File needs to be kept alive for mmap
    file: File,
    mmap: MmapMut,
    current_offset: u64,
    meta_path: PathBuf,
    // Hybrid Search: In-Memory Vector Index
    vector_index: Option<Index>,
    vector_records: HashMap<u64, Vec<f32>>,
    max_size_bytes: u64,
}

impl GeodesicEngine {
    pub fn new<P: AsRef<Path>>(path: P, size_mb: u64) -> std::io::Result<Self> {
        if !(MIN_ARENA_MB..=MAX_ARENA_MB).contains(&size_mb) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "size_mb must be between {} and {}",
                    MIN_ARENA_MB, MAX_ARENA_MB
                ),
            ));
        }
        let db_path = path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false) // Do not truncate existing files
            .open(&db_path)?;

        let requested_size_bytes = size_mb
            .checked_mul(1024)
            .and_then(|value| value.checked_mul(1024))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "size_mb overflow"))?;
        let current_len = file.metadata()?.len();
        let absolute_max_bytes = MAX_ARENA_MB
            .checked_mul(1024 * 1024)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "arena limit overflow"))?;
        if current_len > absolute_max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "existing arena exceeds configured maximum",
            ));
        }
        let size_bytes = current_len.max(requested_size_bytes);
        if current_len < requested_size_bytes {
            file.set_len(requested_size_bytes)?;
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize Vector Index (Simple configuration)
        let options = IndexOptions {
            dimensions: DEFAULT_VECTOR_DIMENSIONS,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            multi: false,
        };

        let index = Index::new(&options)
            .map_err(|e| io::Error::other(format!("vector index init error: {}", e)))?;

        let mut engine = Self {
            heads: HashMap::new(),
            file,
            mmap,
            current_offset: 0,
            meta_path: db_path.with_extension("meta"),
            vector_index: Some(index),
            vector_records: HashMap::new(),
            max_size_bytes: size_bytes,
        };

        engine.restore_metadata()?;
        Ok(engine)
    }

    fn restore_metadata(&mut self) -> io::Result<()> {
        if !self.meta_path.exists() {
            return Ok(());
        }

        let metadata_len = std::fs::metadata(&self.meta_path)?.len();
        if metadata_len > MAX_METADATA_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "metadata file exceeds configured limit",
            ));
        }
        let bytes = std::fs::read(&self.meta_path)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let metadata = rkyv::from_bytes::<EngineMetadata, rkyv::rancor::Error>(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        if metadata.current_offset > self.mmap.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "metadata current_offset exceeds mapped arena size",
            ));
        }
        if metadata.heads.len() > MAX_METADATA_HEADS
            || metadata.vector_records.len() > MAX_VECTOR_RECORDS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "metadata entry count exceeds configured limit",
            ));
        }
        for (key, address) in &metadata.heads {
            if key.is_empty()
                || key.len() > MAX_TOKEN_ID_BYTES
                || *address >= metadata.current_offset
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata contains an invalid head",
                ));
            }
        }
        for (address, vector) in &metadata.vector_records {
            if *address >= metadata.current_offset
                || vector.len() != DEFAULT_VECTOR_DIMENSIONS
                || vector.iter().any(|value| !value.is_finite())
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata contains an invalid vector record",
                ));
            }
        }

        self.heads = metadata.heads;
        self.current_offset = metadata.current_offset;
        self.vector_records = metadata.vector_records;

        for address in self.heads.values().chain(self.vector_records.keys()) {
            if self.read_node_at(*address).is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata references an invalid node",
                ));
            }
        }

        if let Some(idx) = &mut self.vector_index {
            for (address, vector) in &self.vector_records {
                idx.add(*address, vector)
                    .map_err(|e| io::Error::other(format!("Vector index restore error: {}", e)))?;
            }
        }
        Ok(())
    }

    fn persist_metadata(&mut self) -> io::Result<()> {
        let metadata = EngineMetadata {
            heads: self.heads.clone(),
            current_offset: self.current_offset,
            vector_records: self.vector_records.clone(),
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&metadata)
            .map_err(|e| io::Error::other(e.to_string()))?;
        if bytes.len() as u64 > MAX_METADATA_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "metadata exceeds configured limit",
            ));
        }
        // Data must reach the mapped file before metadata publishes pointers
        // to it.  This prevents a metadata head from surviving a torn node.
        self.mmap.flush()?;
        self.file.sync_data()?;
        let tmp_path = self.meta_path.with_extension("meta.tmp");
        {
            let mut tmp = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            use std::io::Write;
            tmp.write_all(&bytes)?;
            tmp.sync_all()?;
        }
        std::fs::rename(&tmp_path, &self.meta_path)?;
        Ok(())
    }

    fn ensure_capacity(&mut self, additional_bytes: u64) -> io::Result<()> {
        let required = self
            .current_offset
            .checked_add(additional_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::OutOfMemory, "arena offset overflow"))?;
        if required <= self.mmap.len() as u64 {
            return Ok(());
        }
        if required > self.max_size_bytes {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "write would exceed configured arena size",
            ));
        }

        let mut new_len = (self.mmap.len() as u64).max(1024 * 1024);
        while new_len < required {
            new_len = new_len.saturating_mul(2);
            if new_len > self.max_size_bytes {
                new_len = self.max_size_bytes;
                break;
            }
        }

        self.mmap.flush()?;
        self.file.set_len(new_len)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        Ok(())
    }

    pub fn write(&mut self, token_id: &str, value: Vec<u8>) -> Result<u64, String> {
        // Hybrid Search Check: If value looks like a vector (float32 bytes), index it.
        // For prototype, we don't parse the bytes here, we assume separate method or manual handling.
        // But let's assume if the user provides a "vector" argument (API change needed), we index it.
        // For this function, we stick to the core log.

        let addr = self.write_internal(token_id, value)?;
        self.persist_metadata().map_err(|e| e.to_string())?;
        Ok(addr)
    }

    // New method for Hybrid Write
    pub fn write_with_vector(
        &mut self,
        token_id: &str,
        value: Vec<u8>,
        vector: Vec<f32>,
    ) -> Result<u64, String> {
        if vector.len() != DEFAULT_VECTOR_DIMENSIONS {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                DEFAULT_VECTOR_DIMENSIONS,
                vector.len()
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err("Vector contains non-finite values".to_string());
        }
        let addr = self.write_internal(token_id, value)?;

        if let Some(idx) = &mut self.vector_index {
            // Use the address as the Key in the vector index
            // Note: usearch keys are u64, perfect for our address/offset
            idx.add(addr, &vector)
                .map_err(|e| format!("Vector index error: {}", e))?;
        }

        self.vector_records.insert(addr, vector);
        self.persist_metadata().map_err(|e| e.to_string())?;

        Ok(addr)
    }

    fn write_internal(&mut self, token_id: &str, value: Vec<u8>) -> Result<u64, String> {
        if token_id.is_empty() || token_id.len() > MAX_TOKEN_ID_BYTES {
            return Err(format!("token_id must be 1..{} bytes", MAX_TOKEN_ID_BYTES));
        }
        if value.len() > MAX_VALUE_BYTES {
            return Err(format!("value exceeds {} bytes", MAX_VALUE_BYTES));
        }
        let prev_ptr = self.heads.get(token_id).copied();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis() as u64;

        let node = Node {
            value,
            prev: prev_ptr,
            timestamp,
        };

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&node).map_err(|e| e.to_string())?;

        let len = u64::try_from(bytes.len()).map_err(|_| "serialized node is too large")?;
        if len > MAX_ARCHIVED_NODE_BYTES {
            return Err("serialized node exceeds configured limit".to_string());
        }

        let required = len
            .checked_add(8)
            .ok_or_else(|| "serialized node length overflow".to_string())?;
        self.ensure_capacity(required).map_err(|e| e.to_string())?;

        let len_bytes = len.to_le_bytes();
        self.mmap[self.current_offset as usize..self.current_offset as usize + 8]
            .copy_from_slice(&len_bytes);

        let data_start = self.current_offset + 8;
        self.mmap[data_start as usize..data_start as usize + len as usize].copy_from_slice(&bytes);

        let node_addr = self.current_offset;
        self.current_offset = self
            .current_offset
            .checked_add(required)
            .ok_or_else(|| "arena offset overflow".to_string())?;

        self.heads.insert(token_id.to_string(), node_addr);

        Ok(node_addr)
    }

    pub fn read_latest(&self, token_id: &str) -> Option<Node> {
        let addr = self.heads.get(token_id)?;
        self.read_node_at(*addr)
    }

    pub fn read_node_at(&self, addr: u64) -> Option<Node> {
        if addr >= self.current_offset {
            return None;
        }
        let data_start = addr.checked_add(8)?;
        if data_start > self.current_offset || data_start > self.mmap.len() as u64 {
            return None;
        }

        let addr_usize = usize::try_from(addr).ok()?;
        let data_start_usize = usize::try_from(data_start).ok()?;
        let len_bytes = &self.mmap[addr_usize..data_start_usize];
        let len = u64::from_le_bytes(len_bytes.try_into().unwrap());
        if len == 0 || len > MAX_ARCHIVED_NODE_BYTES {
            return None;
        }

        let data_end = data_start.checked_add(len)?;
        if data_end > self.current_offset || data_end > self.mmap.len() as u64 {
            return None;
        }

        let data_end_usize = usize::try_from(data_end).ok()?;
        let bytes = &self.mmap[data_start_usize..data_end_usize];

        let node = rkyv::from_bytes::<Node, rkyv::rancor::Error>(bytes).ok()?;
        if node.value.len() > MAX_VALUE_BYTES || node.prev.is_some_and(|previous| previous >= addr)
        {
            return None;
        }
        Some(node)
    }

    pub fn recall(&self, token_id: &str, depth: usize) -> Vec<Node> {
        self.recall_bounded(token_id, depth, MAX_RECALL_BYTES)
    }

    pub fn recall_bounded(
        &self,
        token_id: &str,
        depth: usize,
        max_total_bytes: usize,
    ) -> Vec<Node> {
        let mut result = Vec::new();
        let mut curr_addr = self.heads.get(token_id).copied();
        let mut total_bytes = 0usize;
        let mut visited = HashSet::new();

        for _ in 0..depth.min(MAX_RECALL_DEPTH) {
            if let Some(addr) = curr_addr {
                if !visited.insert(addr) {
                    break;
                }
                if let Some(node) = self.read_node_at(addr) {
                    let Some(next_total) = total_bytes.checked_add(node.value.len()) else {
                        break;
                    };
                    if next_total > max_total_bytes.min(MAX_RECALL_BYTES) {
                        break;
                    }
                    total_bytes = next_total;
                    curr_addr = node.prev;
                    result.push(node);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        result
    }

    pub fn search_similar(&self, vector: Vec<f32>, k: usize) -> Vec<Node> {
        let mut results = Vec::new();
        if vector.len() != DEFAULT_VECTOR_DIMENSIONS
            || vector.iter().any(|value| !value.is_finite())
            || k == 0
            || k > MAX_SEARCH_RESULTS
        {
            return results;
        }

        if let Some(idx) = &self.vector_index {
            // USearch `search` returns `Result<Matches, Error>`.
            // `Matches` stores keys and distances in internal vectors.
            if let Ok(matches) = idx.search(&vector, k) {
                // We need to iterate over the keys. Matches struct exposes .keys field.
                for key in matches.keys {
                    // key is the address in our store (u64)
                    if let Some(node) = self.read_node_at(key) {
                        results.push(node);
                    }
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_paths(name: &str) -> (PathBuf, PathBuf) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let db = std::env::temp_dir().join(format!(
            "oxtamem_{name}_{}_{}.db",
            std::process::id(),
            nonce
        ));
        let meta = db.with_extension("meta");
        (db, meta)
    }

    #[test]
    fn rejects_oversized_inputs_and_bounds_recall() {
        let (db, meta) = test_paths("bounds");
        let mut engine = GeodesicEngine::new(&db, 1).unwrap();
        assert!(engine.write("", vec![1]).is_err());
        assert!(engine.write("key", vec![0; MAX_VALUE_BYTES + 1]).is_err());
        assert!(
            engine
                .write_with_vector("key", vec![1], vec![f32::NAN; DEFAULT_VECTOR_DIMENSIONS],)
                .is_err()
        );

        engine.write("key", vec![1; 32]).unwrap();
        engine.write("key", vec![2; 32]).unwrap();
        engine.write("key", vec![3; 32]).unwrap();
        let recalled = engine.recall_bounded("key", 100, 64);
        assert_eq!(recalled.len(), 2);
        assert_eq!(recalled[0].value[0], 3);
        assert_eq!(recalled[1].value[0], 2);

        drop(engine);
        let restored = GeodesicEngine::new(&db, 1).unwrap();
        assert_eq!(restored.recall("key", 10).len(), 3);
        drop(restored);

        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    #[test]
    fn rejects_corrupt_node_length_without_allocating() {
        let (db, meta) = test_paths("corrupt");
        let mut engine = GeodesicEngine::new(&db, 1).unwrap();
        let address = engine.write("key", vec![1, 2, 3]).unwrap();
        engine.mmap[address as usize..address as usize + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(engine.read_node_at(address).is_none());
        drop(engine);

        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }
}
