use crate::engine::{GeodesicEngine, MAX_RECALL_DEPTH};
use bytes::{Buf, BytesMut};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::time::{Duration, timeout};

const DEFAULT_MAX_CONNECTIONS: usize = 64;
const DEFAULT_MAX_FRAME_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_BULK_BYTES: usize = 512 * 1024;
const DEFAULT_MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_MAX_ARGS: usize = 16;
const DEFAULT_READ_TIMEOUT_MS: u64 = 10_000;
const ABSOLUTE_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
const ABSOLUTE_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const ABSOLUTE_MAX_ARGS: usize = 1024;
const ABSOLUTE_MAX_CONNECTIONS: usize = 4096;

#[derive(Clone, Debug)]
pub struct RespServerConfig {
    pub host: String,
    pub port: u16,
    pub auth_token: Option<String>,
    pub max_connections: usize,
    pub max_frame_bytes: usize,
    pub max_bulk_bytes: usize,
    pub max_response_bytes: usize,
    pub max_args: usize,
    pub read_timeout_ms: u64,
    pub allow_insecure_remote: bool,
}

impl Default for RespServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 6379,
            auth_token: None,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_bulk_bytes: DEFAULT_MAX_BULK_BYTES,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            max_args: DEFAULT_MAX_ARGS,
            read_timeout_ms: DEFAULT_READ_TIMEOUT_MS,
            allow_insecure_remote: false,
        }
    }
}

pub struct RespServer {
    engine: Arc<Mutex<GeodesicEngine>>,
}

impl RespServer {
    pub fn new(engine: Arc<Mutex<GeodesicEngine>>) -> Self {
        Self { engine }
    }

    pub async fn run(&self, config: RespServerConfig) -> Result<(), Box<dyn std::error::Error>> {
        if config.max_connections == 0
            || config.max_connections > ABSOLUTE_MAX_CONNECTIONS
            || config.max_frame_bytes == 0
            || config.max_frame_bytes > ABSOLUTE_MAX_FRAME_BYTES
            || config.max_bulk_bytes == 0
            || config.max_bulk_bytes > config.max_frame_bytes
            || config.max_response_bytes == 0
            || config.max_response_bytes > ABSOLUTE_MAX_RESPONSE_BYTES
            || config.max_args == 0
            || config.max_args > ABSOLUTE_MAX_ARGS
            || config.read_timeout_ms == 0
        {
            return Err("invalid RESP server resource limits".into());
        }
        if !is_loopback_host(&config.host)
            && (!config.allow_insecure_remote
                || config.auth_token.as_ref().map_or(true, |token| token.len() < 16))
        {
            return Err(
                "remote RESP binding requires allow_insecure_remote=true and an auth token of at least 16 bytes"
                    .into(),
            );
        }
        let listener = TcpListener::bind(format!("{}:{}", config.host, config.port)).await?;
        println!("RESP Server listening on {}:{}", config.host, config.port);
        let connection_limit = Arc::new(Semaphore::new(config.max_connections.max(1)));

        loop {
            let (socket, _) = listener.accept().await?;
            let engine = self.engine.clone();
            let permit = match connection_limit.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    let mut rejected = socket;
                    let _ = rejected
                        .write_all(b"-ERR connection limit reached\r\n")
                        .await;
                    continue;
                }
            };
            let config = config.clone();

            tokio::spawn(async move {
                let _permit = permit;
                if let Err(e) = process_connection(socket, engine, config).await {
                    eprintln!("Connection error: {}", e);
                }
            });
        }
    }
}

async fn process_connection(
    mut socket: TcpStream,
    engine: Arc<Mutex<GeodesicEngine>>,
    config: RespServerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = BytesMut::with_capacity(4096);
    let mut authenticated = config.auth_token.is_none();

    loop {
        let read_future = socket.read_buf(&mut buffer);
        let n = timeout(Duration::from_millis(config.read_timeout_ms), read_future).await??;
        if n == 0 {
            return Ok(());
        }
        if buffer.len() > config.max_frame_bytes {
            socket.write_all(b"-ERR frame too large\r\n").await?;
            buffer.clear();
            continue;
        }

        loop {
            match parse_resp(&buffer, &config) {
                Ok(Some((command, consumed))) => {
                    buffer.advance(consumed);
                    let response =
                        handle_command(command, &engine, &config, &mut authenticated).await;
                    socket.write_all(&response).await?;
                }
                Ok(None) => break,
                Err(error) => {
                    socket
                        .write_all(format!("-ERR {}\r\n", error).as_bytes())
                        .await?;
                    buffer.clear();
                    break;
                }
            }
        }
    }
}

fn is_loopback_host(host: &str) -> bool {
    matches!(
        host.trim_matches(|character| character == '[' || character == ']'),
        "" | "localhost" | "localhost." | "127.0.0.1" | "::1" | "0:0:0:0:0:0:0:1"
    )
}

fn constant_time_equal(lhs: &[u8], rhs: &[u8]) -> bool {
    let mut difference = lhs.len() ^ rhs.len();
    const ZERO: u8 = 0;
    let extent = lhs.len().max(rhs.len());
    for index in 0..extent {
        let left = *lhs.get(index).unwrap_or(&ZERO);
        let right = *rhs.get(index).unwrap_or(&ZERO);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

fn find_crlf(buffer: &[u8], start: usize) -> Option<usize> {
    let mut index = start;
    while index + 1 < buffer.len() {
        if buffer[index] == b'\r' && buffer[index + 1] == b'\n' {
            return Some(index);
        }
        index += 1;
    }
    None
}

type RespFrame = Vec<Vec<u8>>;
type RespParse = Result<Option<(RespFrame, usize)>, String>;

// RESP parser for arrays of bulk strings. Binary-safe and supports multiple frames.
fn parse_resp(buffer: &BytesMut, config: &RespServerConfig) -> RespParse {
    if buffer.is_empty() {
        return Ok(None);
    }
    if buffer.len() > config.max_frame_bytes {
        return Err("frame too large".to_string());
    }
    if buffer[0] != b'*' {
        return Err("expected RESP array".to_string());
    }

    let array_end = match find_crlf(buffer, 1) {
        Some(end) => end,
        None => return Ok(None),
    };
    let count_text =
        std::str::from_utf8(&buffer[1..array_end]).map_err(|_| "invalid array length")?;
    let count: usize = count_text.parse().map_err(|_| "invalid array length")?;
    if count == 0 || count > config.max_args {
        return Err("array length outside configured limit".to_string());
    }

    let mut cursor = array_end + 2;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        if cursor >= buffer.len() {
            return Ok(None);
        }
        if buffer[cursor] != b'$' {
            return Err("expected bulk string".to_string());
        }
        let len_end = match find_crlf(buffer, cursor + 1) {
            Some(end) => end,
            None => return Ok(None),
        };
        let len_text =
            std::str::from_utf8(&buffer[cursor + 1..len_end]).map_err(|_| "invalid bulk length")?;
        let len: usize = len_text.parse().map_err(|_| "invalid bulk length")?;
        if len > config.max_bulk_bytes {
            return Err("bulk string exceeds configured limit".to_string());
        }
        cursor = len_end + 2;
        let frame_end = cursor
            .checked_add(len)
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| "frame length overflow".to_string())?;
        if frame_end > config.max_frame_bytes {
            return Err("frame exceeds configured limit".to_string());
        }
        if buffer.len() < frame_end {
            return Ok(None);
        }
        let payload = buffer[cursor..cursor + len].to_vec();
        cursor += len;
        if buffer[cursor] != b'\r' || buffer[cursor + 1] != b'\n' {
            return Err("bulk string missing terminator".to_string());
        }
        cursor += 2;
        args.push(payload);
    }

    Ok(Some((args, cursor)))
}

fn bulk_response(bytes: &[u8], max_response_bytes: usize) -> Option<Vec<u8>> {
    let header = format!("${}\r\n", bytes.len()).into_bytes();
    let total = header.len().checked_add(bytes.len())?.checked_add(2)?;
    if total > max_response_bytes {
        return None;
    }
    let mut response = Vec::with_capacity(total);
    response.extend_from_slice(&header);
    response.extend_from_slice(bytes);
    response.extend_from_slice(b"\r\n");
    Some(response)
}

fn array_bulk_response(items: &[Vec<u8>], max_response_bytes: usize) -> Option<Vec<u8>> {
    let array_header = format!("*{}\r\n", items.len()).into_bytes();
    let mut total = array_header.len();
    for item in items {
        let item_header_len = format!("${}\r\n", item.len()).len();
        total = total
            .checked_add(item_header_len)?
            .checked_add(item.len())?
            .checked_add(2)?;
        if total > max_response_bytes {
            return None;
        }
    }
    let mut response = Vec::with_capacity(total);
    response.extend_from_slice(&array_header);
    for item in items {
        response.extend_from_slice(format!("${}\r\n", item.len()).as_bytes());
        response.extend_from_slice(item);
        response.extend_from_slice(b"\r\n");
    }
    Some(response)
}

async fn handle_command(
    args: Vec<Vec<u8>>,
    engine: &Arc<Mutex<GeodesicEngine>>,
    config: &RespServerConfig,
    authenticated: &mut bool,
) -> Vec<u8> {
    if args.is_empty() {
        return b"-ERR empty command\r\n".to_vec();
    }
    let cmd = match std::str::from_utf8(&args[0]) {
        Ok(value) => value.to_uppercase(),
        Err(_) => return b"-ERR command must be utf-8\r\n".to_vec(),
    };

    match cmd.as_str() {
        "PING" if args.len() == 1 => b"+PONG\r\n".to_vec(),
        "AUTH" if args.len() == 2 => {
            let Some(expected) = &config.auth_token else {
                *authenticated = true;
                return b"+OK\r\n".to_vec();
            };
            if constant_time_equal(&args[1], expected.as_bytes()) {
                *authenticated = true;
                b"+OK\r\n".to_vec()
            } else {
                b"-ERR invalid auth token\r\n".to_vec()
            }
        }
        _ if !*authenticated => b"-NOAUTH authentication required\r\n".to_vec(),
        "SET" if args.len() == 3 => {
            let key = match std::str::from_utf8(&args[1]) {
                Ok(value) => value.to_string(),
                Err(_) => return b"-ERR key must be utf-8\r\n".to_vec(),
            };
            let val = args[2].clone();
            let engine = Arc::clone(engine);
            let result = tokio::task::spawn_blocking(move || {
                let mut eng = engine
                    .lock()
                    .map_err(|_| "engine lock poisoned".to_string())?;
                eng.write(&key, val)
            })
            .await;
            match result {
                Ok(Ok(_)) => b"+OK\r\n".to_vec(),
                Ok(Err(e)) => {
                    eprintln!("OxtaMem SET failed: {e}");
                    b"-ERR storage operation failed\r\n".to_vec()
                }
                Err(error) => {
                    eprintln!("OxtaMem SET worker failed: {error}");
                    b"-ERR storage worker failed\r\n".to_vec()
                }
            }
        }
        "GET" if args.len() == 2 => {
            let key = match std::str::from_utf8(&args[1]) {
                Ok(value) => value.to_string(),
                Err(_) => return b"-ERR key must be utf-8\r\n".to_vec(),
            };
            let engine = Arc::clone(engine);
            let node = tokio::task::spawn_blocking(move || {
                engine.lock().ok().and_then(|eng| eng.read_latest(&key))
            })
            .await;
            match node {
                Ok(Some(node)) => bulk_response(&node.value, config.max_response_bytes)
                    .unwrap_or_else(|| b"-ERR response exceeds configured limit\r\n".to_vec()),
                Ok(None) => b"$-1\r\n".to_vec(),
                Err(_) => b"-ERR storage worker failed\r\n".to_vec(),
            }
        }
        "RECALL" if args.len() == 3 => {
            let key = match std::str::from_utf8(&args[1]) {
                Ok(value) => value.to_string(),
                Err(_) => return b"-ERR key must be utf-8\r\n".to_vec(),
            };
            let depth = match std::str::from_utf8(&args[2])
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(value) if value <= MAX_RECALL_DEPTH => value,
                None => return b"-ERR depth must be an integer\r\n".to_vec(),
                Some(_) => return b"-ERR depth exceeds configured limit\r\n".to_vec(),
            };
            let engine = Arc::clone(engine);
            let max_response_bytes = config.max_response_bytes;
            let items = tokio::task::spawn_blocking(move || {
                let eng = engine.lock().map_err(|_| ())?;
                Ok::<Vec<Vec<u8>>, ()>(eng
                    .recall_bounded(&key, depth, max_response_bytes)
                    .into_iter()
                    .map(|node| node.value)
                    .collect())
            })
            .await;
            match items {
                Ok(Ok(items)) => array_bulk_response(&items, config.max_response_bytes)
                    .unwrap_or_else(|| b"-ERR response exceeds configured limit\r\n".to_vec()),
                _ => b"-ERR storage worker failed\r\n".to_vec(),
            }
        }
        "GETRAW" if args.len() == 2 => {
            let key = match std::str::from_utf8(&args[1]) {
                Ok(value) => value.to_string(),
                Err(_) => return b"-ERR key must be utf-8\r\n".to_vec(),
            };
            let engine = Arc::clone(engine);
            let node = tokio::task::spawn_blocking(move || {
                engine.lock().ok().and_then(|eng| eng.read_latest(&key))
            })
            .await;
            match node {
                Ok(Some(node)) => bulk_response(&node.value, config.max_response_bytes)
                    .unwrap_or_else(|| b"-ERR response exceeds configured limit\r\n".to_vec()),
                Ok(None) => b"$-1\r\n".to_vec(),
                Err(_) => b"-ERR storage worker failed\r\n".to_vec(),
            }
        }
        _ => b"-ERR unknown command\r\n".to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_builders_enforce_total_limit() {
        assert!(bulk_response(&[0; 32], 16).is_none());
        assert!(bulk_response(&[0; 32], 64).is_some());
        assert!(array_bulk_response(&[vec![0; 20], vec![0; 20]], 32).is_none());
        let response = array_bulk_response(&[vec![0; 20], vec![0; 20]], 128)
            .expect("bounded response should fit");
        assert!(response.len() <= 128);
    }

    // The RESP parser reads untrusted network bytes; it must never panic (index
    // out of bounds, slice overflow, ...) on any input — only ever return
    // Ok(None) (need more), Ok(Some) (frame), or Err (reject).
    #[test]
    fn parse_resp_never_panics_on_arbitrary_input() {
        let config = RespServerConfig::default();
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let len = (rng() % 96) as usize;
            let mut buf = BytesMut::with_capacity(len);
            for _ in 0..len {
                // bias toward RESP-significant bytes so we explore real code paths
                let byte = match rng() % 8 {
                    0 => b'*',
                    1 => b'$',
                    2 => b'\r',
                    3 => b'\n',
                    4 => b'0' + (rng() % 10) as u8,
                    _ => (rng() % 256) as u8,
                };
                buf.extend_from_slice(&[byte]);
            }
            // A panic here fails the test; any Ok/Err is acceptable.
            let _ = parse_resp(&buf, &config);
        }

        // Crafted malformed / truncated frames.
        let cases: [&[u8]; 9] = [
            b"*",
            b"*1\r\n",
            b"*1\r\n$",
            b"*-1\r\n",
            b"*99999999\r\n",
            b"*2\r\n$3\r\nfoo\r\n$",
            b"*1\r\n$3\r\nfo\r\n",
            b"$3\r\nfoo\r\n",
            b"*1\r\n*1\r\n",
        ];
        for case in cases {
            let mut buf = BytesMut::new();
            buf.extend_from_slice(case);
            let _ = parse_resp(&buf, &config);
        }
    }
}
