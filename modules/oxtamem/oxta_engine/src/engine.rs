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
const METADATA_MAGIC: &[u8; 8] = b"OXTAMT02";
const METADATA_VERSION: u32 = 2;
const METADATA_ENVELOPE_BYTES: usize = 8 + 4 + 8 + 8;
const JOURNAL_MAGIC: &[u8; 8] = b"OXTAJR01";
const JOURNAL_VERSION: u32 = 1;
const JOURNAL_ENVELOPE_BYTES: usize = 8 + 4 + 8 + 8;
const MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;
const MIN_JOURNAL_COMPACTION_RECORDS: usize = 1024;
const NODE_MAGIC: &[u8; 8] = b"OXTAND02";
const NODE_VERSION: u32 = 2;
const NODE_ENVELOPE_BYTES: usize = 8 + 4 + 8 + 8;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211u64);
    }
    hash
}

fn appended_sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
}

#[cfg(unix)]
fn lock_file_exclusive(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;
    unsafe extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    let status = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "OxtaMem arena is already open by another process",
        ))
    }
}

#[cfg(windows)]
fn lock_file_exclusive(_file: &File) -> io::Result<()> {
    // Windows exclusivity is established by OpenOptionsExt::share_mode(0).
    Ok(())
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let source_wide: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination_wide: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let status = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if status == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    std::fs::rename(source, destination)?;
    if let Some(parent) = destination.parent()
        && !parent.as_os_str().is_empty()
    {
        // The rename is already the logical commit point. A directory fsync is
        // best-effort here: returning an error after the commit would invite a
        // caller retry even though the new snapshot is already visible.
        let _ = File::open(parent).and_then(|directory| directory.sync_all());
    }
    Ok(())
}

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

fn valid_search_vector(vector: &[f32]) -> bool {
    if vector.len() != DEFAULT_VECTOR_DIMENSIONS
        || vector.iter().any(|value| !value.is_finite())
    {
        return false;
    }
    let norm_squared = vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    norm_squared.is_finite() && norm_squared > 1e-12
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

#[derive(Archive, Deserialize, Serialize, Debug)]
struct JournalEntry {
    token_id: String,
    address: u64,
    current_offset: u64,
    vector: Option<Vec<f32>>,
}

pub struct GeodesicEngine {
    pub heads: HashMap<String, u64>, // Maps Variable ID -> Offset in File
    #[allow(dead_code)] // File needs to be kept alive for mmap
    file: File,
    mmap: MmapMut,
    current_offset: u64,
    meta_path: PathBuf,
    journal_path: PathBuf,
    // Hybrid Search: In-Memory Vector Index
    vector_index: Option<Index>,
    vector_records: HashMap<u64, Vec<f32>>,
    max_size_bytes: u64,
    // When true, each write appends a checksummed O(1) durability journal
    // record. When false, persistence is deferred until flush()/Drop.
    sync_on_write: bool,
    // Set on every append, cleared on persist; lets Drop skip a no-op fsync.
    dirty: bool,
    journal_records_since_snapshot: usize,
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
        let mut file_options = OpenOptions::new();
        file_options
            .read(true)
            .write(true)
            .create(true)
            .truncate(false); // Do not truncate existing files
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            file_options.share_mode(0);
        }
        let file = file_options.open(&db_path)?;
        lock_file_exclusive(&file)?;

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

        const LEGACY_METADATA_EXTENSION: &str = "meta";
        let preferred_meta_path = appended_sidecar(&db_path, ".meta");
        let legacy_meta_path = db_path.with_extension(LEGACY_METADATA_EXTENSION);
        let meta_path = if preferred_meta_path.exists() || !legacy_meta_path.exists() {
            preferred_meta_path
        } else {
            legacy_meta_path
        };
        let journal_path = appended_sidecar(&meta_path, ".journal");
        let mut engine = Self {
            heads: HashMap::new(),
            file,
            mmap,
            current_offset: 0,
            meta_path,
            journal_path,
            vector_index: Some(index),
            vector_records: HashMap::new(),
            // Hard ceiling for arena growth; ensure_capacity doubles the mmap up
            // to this limit instead of pinning the arena to its initial size_mb.
            max_size_bytes: absolute_max_bytes,
            sync_on_write: true,
            dirty: false,
            journal_records_since_snapshot: 0,
        };

        engine.restore_metadata()?;
        engine.restore_journal()?;
        engine.rebuild_vector_index()?;
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
        let payload: &[u8] = if bytes.starts_with(METADATA_MAGIC) {
            if bytes.len() < METADATA_ENVELOPE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata envelope is truncated",
                ));
            }
            let version = u32::from_le_bytes(
                bytes[8..12]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad metadata version"))?,
            );
            let payload_len = u64::from_le_bytes(
                bytes[12..20]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad metadata length"))?,
            );
            let checksum = u64::from_le_bytes(
                bytes[20..28]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad metadata checksum"))?,
            );
            let payload_len = usize::try_from(payload_len).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "metadata length overflows usize")
            })?;
            if version != METADATA_VERSION
                || payload_len != bytes.len() - METADATA_ENVELOPE_BYTES
                || fnv1a64(&bytes[METADATA_ENVELOPE_BYTES..]) != checksum
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata envelope validation failed",
                ));
            }
            &bytes[METADATA_ENVELOPE_BYTES..]
        } else {
            // Version-1 compatibility: legacy files were raw rkyv payloads.
            &bytes
        };
        let metadata = rkyv::from_bytes::<EngineMetadata, rkyv::rancor::Error>(payload)
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
                || !valid_search_vector(vector)
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

        Ok(())
    }

    fn truncate_journal_to(&self, length: u64) -> io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.journal_path)?;
        file.set_len(length)?;
        file.sync_all()
    }

    fn restore_journal(&mut self) -> io::Result<()> {
        if !self.journal_path.exists() {
            return Ok(());
        }
        let journal_len = std::fs::metadata(&self.journal_path)?.len();
        if journal_len > MAX_JOURNAL_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "metadata journal exceeds configured limit",
            ));
        }
        let bytes = std::fs::read(&self.journal_path)?;
        let mut cursor = 0usize;
        let mut records = 0usize;

        while cursor < bytes.len() {
            let remaining = bytes.len() - cursor;
            if remaining < JOURNAL_ENVELOPE_BYTES {
                self.truncate_journal_to(cursor as u64)?;
                break;
            }
            if &bytes[cursor..cursor + 8] != JOURNAL_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata journal magic is invalid",
                ));
            }
            let version = u32::from_le_bytes(
                bytes[cursor + 8..cursor + 12]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad journal version"))?,
            );
            let payload_len = u64::from_le_bytes(
                bytes[cursor + 12..cursor + 20]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad journal length"))?,
            );
            let checksum = u64::from_le_bytes(
                bytes[cursor + 20..cursor + 28]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad journal checksum"))?,
            );
            if version != JOURNAL_VERSION || payload_len > MAX_JOURNAL_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata journal envelope is invalid",
                ));
            }
            let payload_len = usize::try_from(payload_len).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "journal length overflows usize")
            })?;
            let record_end = cursor
                .checked_add(JOURNAL_ENVELOPE_BYTES)
                .and_then(|value| value.checked_add(payload_len))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "journal record size overflow")
                })?;
            if record_end > bytes.len() {
                // A crash can leave only the final append incomplete. The last
                // fully checksummed record remains the durable commit point.
                self.truncate_journal_to(cursor as u64)?;
                break;
            }
            let payload = &bytes[cursor + JOURNAL_ENVELOPE_BYTES..record_end];
            if fnv1a64(payload) != checksum {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata journal checksum mismatch",
                ));
            }
            let entry = rkyv::from_bytes::<JournalEntry, rkyv::rancor::Error>(payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            cursor = record_end;
            records = records.saturating_add(1);

            // A snapshot may have committed just before a crash prevented the
            // journal truncation. Such records are idempotently skipped.
            if entry.current_offset <= self.current_offset {
                continue;
            }
            if entry.token_id.is_empty()
                || entry.token_id.len() > MAX_TOKEN_ID_BYTES
                || entry.address != self.current_offset
                || entry.current_offset <= entry.address
                || entry.current_offset > self.mmap.len() as u64
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata journal references an invalid node range",
                ));
            }
            let previous_offset = self.current_offset;
            self.current_offset = entry.current_offset;
            let Some(node) = self.read_node_at(entry.address) else {
                self.current_offset = previous_offset;
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata journal references an invalid node",
                ));
            };
            if node.prev != self.heads.get(&entry.token_id).copied() {
                self.current_offset = previous_offset;
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "metadata journal breaks a node history chain",
                ));
            }
            if let Some(vector) = entry.vector {
                if !valid_search_vector(&vector)
                    || self.vector_records.len() >= MAX_VECTOR_RECORDS
                {
                    self.current_offset = previous_offset;
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "metadata journal contains an invalid vector record",
                    ));
                }
                self.vector_records.insert(entry.address, vector);
            }
            self.heads.insert(entry.token_id, entry.address);
        }
        self.journal_records_since_snapshot = records;
        Ok(())
    }

    fn rebuild_vector_index(&mut self) -> io::Result<()> {
        if let Some(index) = &mut self.vector_index {
            ensure_index_capacity(index, self.vector_records.len()).map_err(io::Error::other)?;
            for (address, vector) in &self.vector_records {
                index.add(*address, vector).map_err(|error| {
                    io::Error::other(format!("Vector index restore error: {error}"))
                })?;
            }
        }
        Ok(())
    }

    fn persist_incremental(
        &mut self,
        token_id: &str,
        address: u64,
        vector: Option<&[f32]>,
    ) -> io::Result<()> {
        let entry = JournalEntry {
            token_id: token_id.to_string(),
            address,
            current_offset: self.current_offset,
            vector: vector.map(|values| values.to_vec()),
        };
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
            .map_err(|error| io::Error::other(error.to_string()))?;
        let record_bytes = JOURNAL_ENVELOPE_BYTES
            .checked_add(payload.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::OutOfMemory, "journal size overflow"))?;
        if record_bytes as u64 > MAX_JOURNAL_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "journal record exceeds configured limit",
            ));
        }

        let data_len = self.current_offset.checked_sub(address).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid journal node range")
        })?;
        self.mmap.flush_range(
            usize::try_from(address)
                .map_err(|_| io::Error::other("journal address overflows usize"))?,
            usize::try_from(data_len)
                .map_err(|_| io::Error::other("journal range overflows usize"))?,
        )?;
        self.file.sync_data()?;

        let existing_len = std::fs::metadata(&self.journal_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let journal_would_overflow = match existing_len.checked_add(record_bytes as u64) {
            Some(length) => length > MAX_JOURNAL_BYTES,
            None => true,
        };
        if journal_would_overflow {
            // The full snapshot includes the just-written logical state, so a
            // successful compaction itself is the durable commit.
            return self.persist_metadata();
        }

        let mut bytes = Vec::with_capacity(record_bytes);
        bytes.extend_from_slice(JOURNAL_MAGIC);
        bytes.extend_from_slice(&JOURNAL_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&fnv1a64(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let mut journal = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(true)
            .open(&self.journal_path)?;
        use std::io::Write;
        if let Err(error) = journal.write_all(&bytes).and_then(|_| journal.sync_all()) {
            let _ = journal.set_len(existing_len).and_then(|_| journal.sync_all());
            return Err(error);
        }
        drop(journal);
        self.journal_records_since_snapshot =
            self.journal_records_since_snapshot.saturating_add(1);

        // Snapshot after O(N) new records for O(1) amortized persistence. A
        // failed compaction is harmless: the journal is already durable.
        let compact_after = self
            .heads
            .len()
            .saturating_add(self.vector_records.len())
            .max(MIN_JOURNAL_COMPACTION_RECORDS);
        if self.journal_records_since_snapshot >= compact_after {
            let _ = self.persist_metadata();
        }
        Ok(())
    }

    fn persist_metadata(&mut self) -> io::Result<()> {
        let metadata = EngineMetadata {
            heads: self.heads.clone(),
            current_offset: self.current_offset,
            vector_records: self.vector_records.clone(),
        };
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&metadata)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let total_bytes = METADATA_ENVELOPE_BYTES
            .checked_add(payload.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::OutOfMemory, "metadata size overflow"))?;
        if total_bytes as u64 > MAX_METADATA_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "metadata exceeds configured limit",
            ));
        }
        let mut bytes = Vec::with_capacity(total_bytes);
        bytes.extend_from_slice(METADATA_MAGIC);
        bytes.extend_from_slice(&METADATA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&fnv1a64(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);
        // Data must reach the mapped file before metadata publishes pointers
        // to it.  This prevents a metadata head from surviving a torn node.
        if self.current_offset > 0 {
            self.mmap.flush_range(0, self.current_offset as usize)?;
        }
        self.file.sync_data()?;
        let tmp_path = appended_sidecar(&self.meta_path, ".tmp");
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
        atomic_replace(&tmp_path, &self.meta_path)?;
        // The snapshot is now the commit point. Journal cleanup is idempotent:
        // if truncation fails, replay skips entries already covered by
        // current_offset, so reporting a false write failure would be worse.
        if self.truncate_journal_to(0).is_ok() {
            let _ = std::fs::remove_file(&self.journal_path);
            self.journal_records_since_snapshot = 0;
        }
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

    /// When `false`, `write`/`write_with_vector` skip the per-write journal
    /// fsync; call `flush()` (or drop the engine) to persist. Trades recent-write
    /// crash durability for throughput.
    pub fn set_sync_on_write(&mut self, sync_on_write: bool) {
        self.sync_on_write = sync_on_write;
    }

    /// Durably persist the metadata snapshot (heads + vector records) to disk.
    pub fn flush(&mut self) -> Result<(), String> {
        self.persist_metadata().map_err(|e| e.to_string())
    }

    pub fn write(&mut self, token_id: &str, value: Vec<u8>) -> Result<u64, String> {
        let previous_head = self.heads.get(token_id).copied();
        let previous_offset = self.current_offset;
        let previous_dirty = self.dirty;
        let addr = self.write_internal(token_id, value)?;
        if self.sync_on_write {
            if let Err(error) = self.persist_incremental(token_id, addr, None) {
                self.rollback_logical_write(
                    token_id,
                    previous_head,
                    previous_offset,
                    previous_dirty,
                );
                return Err(error.to_string());
            }
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
        if !valid_search_vector(&vector) {
            return Err("Vector must contain finite values and have non-zero norm".to_string());
        }
        // Enforce the same cap the restore path checks, atomically before writing
        // anything — otherwise a store can accept vectors it can never reopen with.
        if self.vector_records.len() >= MAX_VECTOR_RECORDS {
            return Err(format!(
                "vector index is full: {MAX_VECTOR_RECORDS} records is the maximum the store can reopen with"
            ));
        }
        if let Some(idx) = &mut self.vector_index {
            let wanted = idx.size() + 1;
            ensure_index_capacity(idx, wanted)?;
        }

        let previous_head = self.heads.get(token_id).copied();
        let previous_offset = self.current_offset;
        let previous_dirty = self.dirty;
        let addr = self.write_internal(token_id, value)?;

        if let Some(idx) = &mut self.vector_index
            && let Err(error) = idx.add(addr, &vector)
        {
            let _ = idx.remove(addr);
            self.rollback_logical_write(
                token_id,
                previous_head,
                previous_offset,
                previous_dirty,
            );
            return Err(format!("Vector index error: {error}"));
        }

        self.vector_records.insert(addr, vector.clone());
        if self.sync_on_write {
            if let Err(error) = self.persist_incremental(token_id, addr, Some(&vector)) {
                self.vector_records.remove(&addr);
                if let Some(idx) = &mut self.vector_index {
                    let _ = idx.remove(addr);
                }
                self.rollback_logical_write(
                    token_id,
                    previous_head,
                    previous_offset,
                    previous_dirty,
                );
                return Err(error.to_string());
            }
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
        if !self.heads.contains_key(token_id) && self.heads.len() >= MAX_METADATA_HEADS {
            return Err(format!(
                "head index is full: {MAX_METADATA_HEADS} unique keys is the maximum"
            ));
        }
        let prev_ptr = self.heads.get(token_id).copied();
        let wall_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis() as u64;
        let timestamp = prev_ptr
            .and_then(|address| self.read_node_at(address))
            .map(|previous| wall_timestamp.max(previous.timestamp.saturating_add(1)))
            .unwrap_or(wall_timestamp);

        let node = Node {
            value,
            prev: prev_ptr,
            timestamp,
        };

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&node).map_err(|e| e.to_string())?;
        let total_bytes = NODE_ENVELOPE_BYTES
            .checked_add(payload.len())
            .ok_or_else(|| "serialized node length overflow".to_string())?;
        let mut bytes = Vec::with_capacity(total_bytes);
        bytes.extend_from_slice(NODE_MAGIC);
        bytes.extend_from_slice(&NODE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&fnv1a64(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);

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

    fn rollback_logical_write(
        &mut self,
        token_id: &str,
        previous_head: Option<u64>,
        previous_offset: u64,
        previous_dirty: bool,
    ) {
        self.current_offset = previous_offset;
        self.dirty = previous_dirty;
        match previous_head {
            Some(address) => {
                self.heads.insert(token_id.to_string(), address);
            }
            None => {
                self.heads.remove(token_id);
            }
        }
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
        let payload = if bytes.starts_with(NODE_MAGIC) {
            if bytes.len() < NODE_ENVELOPE_BYTES {
                return None;
            }
            let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
            let payload_len = u64::from_le_bytes(bytes[12..20].try_into().ok()?);
            let checksum = u64::from_le_bytes(bytes[20..28].try_into().ok()?);
            let payload_len = usize::try_from(payload_len).ok()?;
            if version != NODE_VERSION
                || payload_len != bytes.len() - NODE_ENVELOPE_BYTES
                || fnv1a64(&bytes[NODE_ENVELOPE_BYTES..]) != checksum
            {
                return None;
            }
            &bytes[NODE_ENVELOPE_BYTES..]
        } else {
            // Version-1 compatibility: legacy arena records contained the raw
            // rkyv Node payload immediately after the outer u64 length.
            bytes
        };

        let node = rkyv::from_bytes::<Node, rkyv::rancor::Error>(payload).ok()?;
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
        if !valid_search_vector(&vector)
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
        let meta = appended_sidecar(&db, ".meta");
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

    #[test]
    fn metadata_failure_rolls_back_logical_write() {
        let (db, meta) = test_paths("rollback");
        let journal = appended_sidecar(&meta, ".journal");
        let mut engine = GeodesicEngine::new(&db, 2).unwrap();
        engine.write("stable", vec![1]).unwrap();
        let before_offset = engine.current_offset;
        std::fs::remove_file(&journal).unwrap();
        std::fs::create_dir(&journal).unwrap();

        assert!(engine.write("rejected", vec![2]).is_err());
        assert_eq!(engine.current_offset, before_offset);
        assert!(engine.read_latest("rejected").is_none());
        assert_eq!(engine.read_latest("stable").unwrap().value, vec![1]);

        let vector = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 0);
        assert!(
            engine
                .write_with_vector("vector-rejected", vec![3], vector.clone())
                .is_err()
        );
        assert!(engine.read_latest("vector-rejected").is_none());
        assert!(engine.search_similar(vector, 1).is_empty());

        drop(engine);
        let _ = std::fs::remove_dir_all(journal);
        let _ = std::fs::remove_file(meta);
        let _ = std::fs::remove_file(db);
    }

    #[test]
    fn journal_replays_durable_writes_and_recovers_torn_tail() {
        let (db, meta) = test_paths("journal_replay");
        let journal = appended_sidecar(&meta, ".journal");
        let vector = unit_vector(DEFAULT_VECTOR_DIMENSIONS, 4);
        {
            let mut engine = GeodesicEngine::new(&db, 2).unwrap();
            engine.write("plain", b"one".to_vec()).unwrap();
            engine
                .write_with_vector("vector", b"two".to_vec(), vector.clone())
                .unwrap();
            // Simulate an abrupt process exit after journal fsync but before
            // Drop compacts a full metadata snapshot.
            engine.dirty = false;
        }
        assert!(!meta.exists());
        {
            use std::io::Write;
            let mut file = OpenOptions::new().append(true).open(&journal).unwrap();
            file.write_all(b"OXTA").unwrap();
            file.sync_all().unwrap();
        }
        {
            let engine = GeodesicEngine::new(&db, 2).unwrap();
            assert_eq!(engine.read_latest("plain").unwrap().value, b"one");
            assert_eq!(engine.read_latest("vector").unwrap().value, b"two");
            let hits = engine.search_similar(vector, 1);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].value, b"two");
        }
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
        let _ = std::fs::remove_file(journal);
    }

    #[test]
    fn journal_checksum_corruption_is_rejected() {
        let (db, meta) = test_paths("journal_checksum");
        let journal = appended_sidecar(&meta, ".journal");
        {
            let mut engine = GeodesicEngine::new(&db, 2).unwrap();
            engine.write("k", b"value".to_vec()).unwrap();
            engine.dirty = false;
        }
        let mut bytes = std::fs::read(&journal).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x5a;
        std::fs::write(&journal, bytes).unwrap();
        assert!(
            GeodesicEngine::new(&db, 2).is_err(),
            "silently corrupted journal payload must be rejected"
        );
        let _ = std::fs::remove_file(db);
        let _ = std::fs::remove_file(meta);
        let _ = std::fs::remove_file(journal);
    }

    #[test]
    fn arena_rejects_a_second_process_handle() {
        let (db, meta) = test_paths("exclusive");
        let first = GeodesicEngine::new(&db, 2).unwrap();
        assert!(
            GeodesicEngine::new(&db, 2).is_err(),
            "two independent writers must not mmap the same arena"
        );
        drop(first);
        assert!(GeodesicEngine::new(&db, 2).is_ok());
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
