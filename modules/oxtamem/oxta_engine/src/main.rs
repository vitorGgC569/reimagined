use clap::Parser;
use oxta_mem::engine::GeodesicEngine;
use oxta_mem::server::{RespServer, RespServerConfig};
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 6379)]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, env = "OXTAMEM_AUTH_TOKEN")]
    auth_token: Option<String>,

    #[arg(short, long, default_value = "geodesic.db")]
    db_path: String,

    #[arg(short, long, default_value_t = 100)]
    size_mb: u64,

    #[arg(long, default_value_t = 64)]
    max_connections: usize,

    #[arg(long, default_value_t = 1_048_576)]
    max_frame_bytes: usize,

    #[arg(long, default_value_t = 524_288)]
    max_bulk_bytes: usize,

    #[arg(long, default_value_t = 10_000)]
    read_timeout_ms: u64,
}

fn is_loopback_host(host: &str) -> bool {
    host.is_empty() || host == "localhost" || host == "127.0.0.1" || host == "::1"
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !is_loopback_host(&args.host) && args.auth_token.is_none() {
        return Err(
            "refusing to bind OxtaMem outside loopback without --auth-token/OXTAMEM_AUTH_TOKEN"
                .into(),
        );
    }

    println!("--- Geodesic Engine Starting ---");
    println!("Store: {} ({} MB)", args.db_path, args.size_mb);

    let engine = GeodesicEngine::new(&args.db_path, args.size_mb)?;
    let shared_engine = Arc::new(Mutex::new(engine));

    let server = RespServer::new(shared_engine);
    server
        .run(RespServerConfig {
            host: args.host,
            port: args.port,
            auth_token: args.auth_token,
            max_connections: args.max_connections,
            max_frame_bytes: args.max_frame_bytes,
            max_bulk_bytes: args.max_bulk_bytes,
            read_timeout_ms: args.read_timeout_ms,
            ..RespServerConfig::default()
        })
        .await?;

    Ok(())
}
