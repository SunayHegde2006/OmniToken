//! `omnitoken` CLI binary.
//!
//! Subcommands:
//!   encode   — encode stdin lines using a HuggingFace tokenizers.json vocab
//!
//! **Allocator (plan Section 4.6):** mimalloc replaces the default allocator.
//! On Windows the default allocator is not optimised for the small, high-frequency
//! allocations typical of tokenizer hot paths; mimalloc is a measurable, low-risk win.

#[cfg(not(target_os = "macos"))]
use mimalloc::MiMalloc;
#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::{self, BufRead};

#[derive(Parser)]
#[command(
    name    = "omnitoken",
    version,
    about   = "Universal tokenizer for BPE, WordPiece, and Unigram vocabularies.",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Encode text from stdin, one line at a time, using a HF tokenizers.json vocab.
    Encode {
        #[arg(short, long, help = "Path to tokenizers.json")]
        vocab: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Encode { vocab } => {
            let ir   = vocab_ir::load_hf_file(&vocab)?;
            let mr   = walker::build_merge_rank(&ir);
            let mut cache = hot_cache::HotCache::new();

            for line in io::stdin().lock().lines() {
                let line = line?;
                let mut ids = Vec::new();
                for word in line.split_whitespace() {
                    let bytes = word.as_bytes();
                    if let Some(cached) = cache.get(bytes) {
                        ids.extend_from_slice(cached);
                    } else {
                        let word_ids = walker::bpe_encode(word, &ir.vocab, &mr)?;
                        cache.insert(bytes, word_ids.clone());
                        ids.extend(word_ids);
                    }
                }
                for (idx, id) in ids.iter().enumerate() {
                    if idx > 0 { print!(" "); }
                    print!("{id}");
                }
                println!();
            }
        }
    }
    Ok(())
}
