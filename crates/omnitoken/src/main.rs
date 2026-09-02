//! `omnitoken` CLI binary.
//!
//! Subcommands:
//!   encode   — encode stdin lines using a HuggingFace tokenizers.json or .otk model
//!   compile  — compile tokenizers.json into a zero-copy .otk binary blob

#[cfg(target_os = "linux")]
use mimalloc::MiMalloc;
#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::{self, BufRead};

#[derive(Parser)]
#[command(
    name    = "omnitoken",
    version,
    about   = "Universal Hardware-Adaptive Tokenizer for BPE, WordPiece, and Unigram vocabularies.",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Encode text from stdin, one line at a time.
    Encode {
        #[arg(short, long, help = "Path to tokenizers.json or model.otk")]
        vocab: std::path::PathBuf,
    },
    /// Compile a tokenizers.json into a zero-copy .otk binary blob.
    Compile {
        #[arg(short, long, help = "Path to input HuggingFace tokenizers.json")]
        vocab: std::path::PathBuf,

        #[arg(short, long, help = "Path to output .otk file")]
        out: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Encode { vocab } => {
            let ir = if vocab.extension().and_then(|s| s.to_str()) == Some("otk") {
                vocab_ir::load_otk_file(&vocab)?
            } else {
                vocab_ir::load_hf_file(&vocab)?
            };
            let trie = trie_builder::build(&ir)?;
            let dispatcher = omnitoken::dispatcher::AdaptiveDispatcher::new();

            for line in io::stdin().lock().lines() {
                let line = line?;
                let ids = dispatcher.encode(&line, &ir, &trie)?;
                for (idx, id) in ids.iter().enumerate() {
                    if idx > 0 { print!(" "); }
                    print!("{id}");
                }
                println!();
            }
        }
        Cmd::Compile { vocab, out } => {
            let ir = vocab_ir::load_hf_file(&vocab)?;
            ir.save_otk(&out)?;
            println!(" Successfully compiled `{}` -> `{}` ({} tokens)", vocab.display(), out.display(), ir.len());
        }
    }
    Ok(())
}
