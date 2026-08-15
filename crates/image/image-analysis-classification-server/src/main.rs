use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "image-analysis-classification-server",
    version,
    about = "Thin HTTP API adapter for image-analysis-classification"
)]
struct Args {
    /// Address to bind, for example 127.0.0.1:3000.
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    eprintln!(
        "image-analysis-classification-server listening on http://{}",
        args.addr
    );
    image_analysis_classification_server::serve(&args.addr)
}
