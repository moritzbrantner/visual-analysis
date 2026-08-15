use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "vision-core-server",
    version,
    about = "Thin HTTP API adapter for vision-core"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!("vision-core-server listening on http://{}", args.addr);
    vision_core_server::serve(&args.addr)
}
