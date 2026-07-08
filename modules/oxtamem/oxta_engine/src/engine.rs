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

/// usearch needs capacity reserved before `add`; inserting into a zero-capacity
/// index dereferences uninitialised internals and crashes the process (this path
/// was never exercised by a successful write, so the crash shipped latent).  Grow
/// geometrically (double, min 1024) so bulk ingestion amortizes to O(1) reserves
/// instead of reallocating the HNSW graph on every single insert.
fn ensure_index_capacity(idx: &Index, wanted: usize) -> Result<(), String> {
    if wanted > idx.capacity() {
        let target = wanted.max(idx.capacity().saturating_mul(2)).max(1024);
        idx.reserve(target)
            .map_err(|e| format!("Vector index reserve error: {e}"))?;
    }
    Ok(())
}

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
    // When false, per-write metadata persistence is deferred until flush()/Drop.
    sync_on_write: bool,
    // Set on every append, cleared on persist; lets Drop skip a no-op fsync.
    dirty: bool,
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
            // Hard ceiling for arena growth; ensure_capacity doubles the mmap up
            // to this limit instead of pinning the arena to its initial size_mb.
            max_size_bytes: absolute_max_bytes,
            sync_on_write: true,
            dirty: false,
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
            ensure_index_capacity(idx, self.vector_records.len()).map_err(io::Error::other)?;
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
        self.dirty = false;
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

    /// When `false`, `write`/`write_with_vector` skip the per-write metadata
    /// fsync; call `flush()` (or drop the engine) to persist.  Trades recent-write
    /// crash durability for throughput — the node bytes are already durably in the
    /// arena, only the metadata pointer that references them is deferred.
    pub fn set_sync_on_write(&mut self, sync_on_write: bool) {
        self.sync_on_write = sync_on_write;
    }

    /// Durably persist the metadata snapshot (heads + vector records) to disk.
    pub fn flush(&mut self) -> Result<(), String> {
        self.persist_metadata().map_err(|e| e.to_string())
    }

    pub fn write(&mut self, token_id: &str, value: Vec<u8>) -> Result<u64, String> {
        let addr = self.write_internal(token_id, value)?;
        if self.sync_on_write {
            self.persist_metadata().map_err(|e| e.to_string())?;
        }
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
        // Enforce the same cap the restore path checks, atomically before writing
        // anything — otherwise a store can accept vectors it can never reopen with.
        if self.vector_records.len() >= MAX_VECTOR_RECORDS {
            return Err(format!(
                "vector index is full: {MAX_VECTOR_RECORDS} records is the maximum the store can reopen with"
            ));
        }
        let addr = self.write_internal(token_id, value)?;

        if let Some(idx) = &mut self.vector_index {
            // Use the address as the Key in the vector index
            // Note: usearch keys are u64, perfect for our address/offset
            let wanted = idx.size() + 1;
            ensure_index_capacity(idx, wanted)?;
            idx.add(addr, &vector)
                .map_err(|e| format!("Vector index error: {}", e))?;
        }

        self.vector_records.insert(addr, vector);
        if self.sync_on_write {
            self.persist_metadata().map_err(|e| e.to_string())?;
        }

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
        self.dirty = true;

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
        self.search_similar_scored(vector, k)
            .into_iter()
            .map(|(_, node)| node)
            .collect()
    }

    /// Like `search_similar` but returns the cosine distance next to each node
    /// (smaller = closer; ~0 means near-identical direction).  Lets callers
    /// threshold on relevance instead of blindly trusting the top-k ordering.
    pub fn search_similar_scored(&self, vector: Vec<f32>, k: usize) -> Vec<(f32, Node)> {
        let mut results = Vec::new();
        if vector.len() != DEFAULT_VECTOR_DIMENSIONS
            || vector.iter().any(|value| !value.is_finite())
            || k == 0
            || k > MAX_SEARCH_RESULTS
        {
            return results;
        }

        if let Some(idx) = &self.vector_index {
            // usearch returns keys (our store addresses) and distances, ordered
            // closest-first; pair them so callers get a relevance score.
            if let Ok(matches) = idx.search(&vector, k) {
                for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
                    if let Some(node) = self.read_node_at(*key) {
                        results.push((*distance, node));
                    }
                }
            }
        }

        results
    }
}

impl Drop for GeodesicEngine {
    fn drop(&mut self) {
        // Safety net for deferred (sync_on_write = false) writes: persist on a
        // clean shutdown.  Best-effort — a failure here cannot be surfaced.
        if self.dirty {
            let _ = self.persist_metadata();
        }
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

    fn unit_vector(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot % dim] = 1.0;
        v
    }

    // Causal recall must return a key's history newest-first, honour the depth
    // bound, isolate keys from each other, and yield nothing for unknown keys.
    #[test]
    fn causal_recall_orders_and_isolates_keys() {
        let (db, meta) = test_paths("causal");
        let mut engine = GeodesicEngine::new(&db, 2).unwrap();
        for i in 1..=5u8 {
            engine.write("x", vec![i]).unwrap();
        }
        engine.write("y", vec![99]).unwrap();

        let rx = engine.recall("x", 10);
        assert_eq!(rx.len(), 5, "should recall the full chain");
        assert_eq!(rx[0].value[0], 5, "newest first");
        assert_eq!(rx[4].value[0], 1, "oldest last");
        assert!(
            rx.windows(2).all(|w| w[0].timestamp >= w[1].timestamp),
            "timestamps must be non-increasing walking back the chain"
        );

        assert_eq!(engine.recall("x", 2).len(), 2, "depth bound respected");

        let ry = engine.recall("y", 10);
        assert_eq!(ry.len(), 1, "keys are isolated");
        assert_eq!(ry[0].value[0], 99);

        assert!(
            engine.recall("missing", 10).is_empty(),
            "unknown key -> empty"
        );

        drop(engine);
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // A store must survive a restart: causal chains AND the vector index have to
    // come back.  This exercises the restore replay path that re-adds every
    // persisted vector to a fresh usearch index (regression guard for the
    // missing-`reserve` crash — restore adds without a live write in between).
    #[test]
    fn persistence_roundtrip_recovers_chains_and_vectors() {
        let (db, meta) = test_paths("roundtrip");
        let v_a = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 0);
        let v_b = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 1);
        {
            let mut engine = GeodesicEngine::new(&db, 4).unwrap();
            engine.write("k", vec![1; 8]).unwrap();
            engine.write("k", vec![2; 8]).unwrap();
            engine
                .write_with_vector("a", b"AAA".to_vec(), v_a.clone())
                .unwrap();
            engine
                .write_with_vector("b", b"BBB".to_vec(), v_b.clone())
                .unwrap();
            let hit = engine.search_similar(v_a.clone(), 1);
            assert_eq!(hit.len(), 1);
            assert_eq!(hit[0].value, b"AAA", "search works before restart");
        }
        {
            let engine = GeodesicEngine::new(&db, 4).unwrap();
            assert_eq!(engine.recall("k", 10).len(), 2, "chain survived restart");
            let hit = engine.search_similar(v_a.clone(), 1);
            assert_eq!(hit.len(), 1, "vector index rebuilt on restart");
            assert_eq!(hit[0].value, b"AAA", "search works after restart");
        }
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // Writing more than the initial `size_mb` must grow the arena (the doubling
    // logic in `ensure_capacity`), not fail with OutOfMemory.  Guards against the
    // arena being silently pinned to its initial size.
    #[test]
    fn arena_grows_beyond_initial_size() {
        let (db, meta) = test_paths("grow");
        let mut engine = GeodesicEngine::new(&db, 1).unwrap(); // 1 MiB arena
        let value = vec![7u8; 8 * 1024]; // 8 KiB payload
        let count = 300; // ~2.4 MiB of payload -> must grow past 1 MiB
        for i in 0..count {
            engine
                .write(&format!("k{i}"), value.clone())
                .unwrap_or_else(|e| panic!("write {i} should grow the arena, got: {e}"));
        }
        let last = engine.read_latest(&format!("k{}", count - 1)).unwrap();
        assert_eq!(last.value.len(), 8 * 1024, "late write must be readable");

        drop(engine);
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // search_similar must degrade gracefully at the edges instead of crashing or
    // returning garbage.
    #[test]
    fn search_similar_handles_edges() {
        let (db, meta) = test_paths("edges");
        let mut engine = GeodesicEngine::new(&db, 2).unwrap();
        let v = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 3);

        assert!(
            engine.search_similar(v.clone(), 5).is_empty(),
            "empty index -> no results"
        );

        engine
            .write_with_vector("a", b"A".to_vec(), v.clone())
            .unwrap();

        let over_k = engine.search_similar(v.clone(), 100);
        assert_eq!(over_k.len(), 1, "k larger than size returns what exists");

        assert!(
            engine.search_similar(vec![0.0; 10], 5).is_empty(),
            "wrong dimension is rejected, not searched"
        );
        assert!(engine.search_similar(v, 0).is_empty(), "k=0 -> empty");

        drop(engine);
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // Deferred durability keeps writes in-memory until flush(); a clean drop
    // still persists them (the Drop safety net).  A crash before flush would lose
    // the deferred writes but must never corrupt the store.
    #[test]
    fn deferred_durability_persists_on_flush_and_drop() {
        let (db, meta) = test_paths("deferred");
        {
            let mut engine = GeodesicEngine::new(&db, 2).unwrap();
            engine.set_sync_on_write(false);
            engine.write("k", vec![1, 2, 3]).unwrap();
            engine.write("k", vec![4, 5, 6]).unwrap();
            engine.flush().unwrap();
            // written after the flush -> only persisted by the Drop safety net
            engine.write("k", vec![7, 8, 9]).unwrap();
        }
        {
            let engine = GeodesicEngine::new(&db, 2).unwrap();
            assert_eq!(
                engine.recall("k", 10).len(),
                3,
                "flushed + drop-persisted writes all survive restart"
            );
        }
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // Scored search exposes cosine distance: closest first, exact match ~0.
    #[test]
    fn scored_search_orders_by_distance() {
        let (db, meta) = test_paths("scored");
        let mut engine = GeodesicEngine::new(&db, 2).unwrap();
        let a = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 0);
        let b = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 1); // orthogonal to a
        engine
            .write_with_vector("a", b"A".to_vec(), a.clone())
            .unwrap();
        engine
            .write_with_vector("b", b"B".to_vec(), b.clone())
            .unwrap();

        let scored = engine.search_similar_scored(a.clone(), 2);
        assert_eq!(scored.len(), 2);
        assert_eq!(scored[0].1.value, b"A", "exact match ranked first");
        assert!(
            scored[0].0 <= scored[1].0,
            "distances ascending (closest first)"
        );
        assert!(
            scored[0].0 < 1e-3,
            "exact-match cosine distance ~0, got {}",
            scored[0].0
        );

        drop(engine);
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // A torn or corrupt metadata file must be rejected on open — never a panic
    // and never a half-valid store.
    #[test]
    fn torn_metadata_is_rejected_on_open() {
        let (db, meta) = test_paths("torn");
        {
            let mut engine = GeodesicEngine::new(&db, 2).unwrap();
            engine.write("k", vec![1, 2, 3]).unwrap();
        }
        // Truncate the persisted metadata to half its bytes (simulates a crash
        // mid-write); rkyv stores its root at the tail, so a truncated buffer can
        // no longer be resolved.
        let data = std::fs::read(&meta).unwrap();
        assert!(data.len() > 8);
        {
            use std::io::Write;
            let mut file = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&meta)
                .unwrap();
            file.write_all(&data[..data.len() / 2]).unwrap();
        }
        assert!(
            GeodesicEngine::new(&db, 2).is_err(),
            "torn metadata must fail to open, not load a corrupt store"
        );
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // The engine behind an Arc<Mutex<...>> (how the server shares it) must stay
    // consistent under many concurrent writers: every write lands, none corrupt.
    #[test]
    fn concurrent_writers_stay_consistent() {
        use std::sync::{Arc, Mutex};
        let (db, meta) = test_paths("concurrent");
        let engine = Arc::new(Mutex::new(GeodesicEngine::new(&db, 16).unwrap()));
        engine.lock().unwrap().set_sync_on_write(false); // keep the test fast
        let threads = 8usize;
        let per_thread = 50usize;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let eng = engine.clone();
                std::thread::spawn(move || {
                    for i in 0..per_thread {
                        eng.lock()
                            .unwrap()
                            .write(&format!("t{t}_{i}"), vec![t as u8; 4])
                            .unwrap();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        let guard = engine.lock().unwrap();
        for t in 0..threads {
            for i in 0..per_thread {
                let node = guard
                    .read_latest(&format!("t{t}_{i}"))
                    .expect("every concurrent write must be readable");
                assert_eq!(node.value, vec![t as u8; 4]);
            }
        }
        drop(guard);
        drop(engine);
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
    }

    // Quantifies the per-write fsync ceiling vs deferred durability.  Run with:
    //   cargo test -p oxta_mem --release -- --ignored --nocapture write_throughput
    #[test]
    #[ignore = "write-throughput benchmark; run explicitly with --ignored --nocapture"]
    fn write_throughput_sync_vs_deferred() {
        let n = 2000usize;
        let value = vec![0u8; 64];

        let (db1, m1) = test_paths("thr_sync");
        let mut sync_engine = GeodesicEngine::new(&db1, 32).unwrap();
        let t = std::time::Instant::now();
        for i in 0..n {
            sync_engine.write(&format!("k{i}"), value.clone()).unwrap();
        }
        let sync_ms = t.elapsed().as_secs_f64() * 1000.0;
        drop(sync_engine);
        let _ = std::fs::remove_file(&db1);
        let _ = std::fs::remove_file(&m1);

        let (db2, m2) = test_paths("thr_deferred");
        let mut def_engine = GeodesicEngine::new(&db2, 32).unwrap();
        def_engine.set_sync_on_write(false);
        let t = std::time::Instant::now();
        for i in 0..n {
            def_engine.write(&format!("k{i}"), value.clone()).unwrap();
        }
        def_engine.flush().unwrap();
        let def_ms = t.elapsed().as_secs_f64() * 1000.0;
        drop(def_engine);
        let _ = std::fs::remove_file(&db2);
        let _ = std::fs::remove_file(&m2);

        println!(
            "\n[write throughput N={n}] sync={sync_ms:.0}ms ({:.0}/s)  \
             deferred+flush={def_ms:.0}ms ({:.0}/s)  speedup {:.1}x",
            n as f64 / (sync_ms / 1000.0),
            n as f64 / (def_ms / 1000.0).max(1e-9),
            sync_ms / def_ms.max(1e-9)
        );
    }

    // Deterministic splitmix64 PRNG so the benchmark is reproducible without
    // pulling an external `rand` dependency into the engine crate.
    struct SplitMix(u64);
    impl SplitMix {
        fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f32 {
            // 24 random mantissa bits -> [0, 1)
            (self.u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn sym(&mut self) -> f32 {
            self.unit() * 2.0 - 1.0
        }
    }

    fn l2_normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    // OxtaMem Camada 0: isolated retrieval-primitive quality.
    // Stores N synthetic 128-dim unit vectors (value = item index), then queries
    // with clean and noisy versions of stored vectors and measures recall@k over
    // the cosine/usearch(HNSW) index. No model involved — this validates the
    // memory primitive alone: "given a store + a query embedding, does it return
    // the right item?".  Run with:
    //   cargo test -p oxta_mem --release -- --ignored --nocapture recall_at_k
    // Scales overridable via OXTA_RECALL_SCALES="500,2000,10000".
    #[test]
    #[ignore = "recall@k benchmark; run explicitly with --ignored --nocapture"]
    fn recall_at_k_benchmark() {
        let dim = DEFAULT_VECTOR_DIMENSIONS;
        let scales: Vec<usize> = std::env::var("OXTA_RECALL_SCALES")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![500, 2000]);
        let noise_levels = [0.0f32, 0.25, 0.5, 1.0, 2.0];
        let n_queries = 300usize;
        let k = 10usize;

        println!(
            "\n===== OxtaMem Camada 0: recall@k  (dim={}, metric=cosine, usearch HNSW) =====",
            dim
        );
        println!(
            "queries/level={}  k={}  | recall@1 = exact top-1 hit, recall@{} = target within top-{}",
            n_queries, k, k, k
        );

        for &n in &scales {
            let (db, meta) = test_paths(&format!("recall_{n}"));
            // 64 MiB backing file is plenty for N up to ~100k tiny values.
            let mut engine = GeodesicEngine::new(&db, 64).unwrap();

            // ---- populate: N random unit vectors, value = index as u32 LE ----
            let mut rng = SplitMix(0xDEAD_BEEF ^ n as u64);
            let mut bases: Vec<Vec<f32>> = Vec::with_capacity(n);
            let t_w = std::time::Instant::now();
            for i in 0..n {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.sym()).collect();
                l2_normalize(&mut v);
                engine
                    .write_with_vector(
                        &format!("it{i}"),
                        (i as u32).to_le_bytes().to_vec(),
                        v.clone(),
                    )
                    .unwrap();
                bases.push(v);
            }
            let w_ms = t_w.elapsed().as_secs_f64() * 1000.0;
            let per = w_ms / n as f64;
            println!(
                "\n[N={}] write {:.0} ms total | {:.3} ms/op | {:.0} writes/s",
                n,
                w_ms,
                per,
                1000.0 / per.max(1e-9)
            );

            // ---- query: clean + noisy versions of stored vectors ----
            let mut qrng = SplitMix(0x1234_5678 ^ n as u64);
            let mut exact_recall_k = 0.0f64;
            for &eps in &noise_levels {
                let mut hit1 = 0usize;
                let mut hitk = 0usize;
                let mut lat_us = 0.0f64;
                for _ in 0..n_queries {
                    let j = (qrng.u64() as usize) % n;
                    let mut q = bases[j].clone();
                    if eps > 0.0 {
                        for x in q.iter_mut() {
                            *x += eps * qrng.sym();
                        }
                        l2_normalize(&mut q);
                    }
                    let t = std::time::Instant::now();
                    let res = engine.search_similar(q, k);
                    lat_us += t.elapsed().as_secs_f64() * 1e6;
                    let got: Vec<u32> = res
                        .iter()
                        .filter(|nd| nd.value.len() >= 4)
                        .map(|nd| {
                            u32::from_le_bytes([nd.value[0], nd.value[1], nd.value[2], nd.value[3]])
                        })
                        .collect();
                    if got.first() == Some(&(j as u32)) {
                        hit1 += 1;
                    }
                    if got.contains(&(j as u32)) {
                        hitk += 1;
                    }
                }
                let r1 = hit1 as f64 / n_queries as f64;
                let rk = hitk as f64 / n_queries as f64;
                if eps == 0.0 {
                    exact_recall_k = rk;
                }
                println!(
                    "  eps={:.2}  recall@1={:.3}  recall@{}={:.3}  | {:.1} us/query",
                    eps,
                    r1,
                    k,
                    rk,
                    lat_us / n_queries as f64
                );
            }

            // Sanity guard: clean (eps=0) queries must find the exact stored item
            // in the top-k — otherwise the primitive is broken, not just approximate.
            assert!(
                exact_recall_k >= 0.90,
                "exact-query recall@{k} collapsed at N={n}: {exact_recall_k:.3} (< 0.90)"
            );

            drop(engine);
            let _ = std::fs::remove_file(&db);
            let _ = std::fs::remove_file(&meta);
        }
        println!("=====================================================================\n");
    }
}
