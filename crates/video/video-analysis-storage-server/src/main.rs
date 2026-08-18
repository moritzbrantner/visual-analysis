use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "video-analysis-storage-server",
    version,
    about = "Thin HTTP API adapter for video-analysis-storage"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!(
        "video-analysis-storage-server listening on http://{}",
        args.addr
    );
    video_analysis_storage_server::serve(&args.addr)
}
