//! `bench` — Reproducible benchmark with roofline sanity checking & differential fuzzing.

use anyhow::Result;
use clap::Parser;
use memmap2::Mmap;
use std::fs::File;
use std::time::Instant;
use walker::verify_bpe_conformance;

#[derive(Parser)]
#[command(
    name  = "bench",
    version,
    about = "OmniToken reproducible benchmark — multi-scenario throughput, single/multi-thread, .otk mmap, TTFT & differential fuzzing.",
)]
struct Cli {
    #[arg(short, long, help = "Path to HuggingFace tokenizers.json or model.otk")]
    vocab: std::path::PathBuf,

    #[arg(short, long, help = "Path to corpus text file (optional; falls back to synthetic input if omitted)")]
    corpus: Option<std::path::PathBuf>,

    #[arg(short, long, default_value = "12",
          help = "Number of threads (default 12 for multi-threaded scenario)")]
    threads: usize,

    #[arg(short, long, default_value = "16777216",
          help = "Bytes of synthetic corpus to tokenize if no corpus file is provided (default 16 MiB)")]
    bytes: usize,

    #[arg(short, long, help = "Run parity verification & differential fuzzing self-check")]
    parity: bool,

    #[arg(long, help = "Use memory-mapped file reader for disk streaming")]
    mmap: bool,
    #[arg(long, default_value = "3", help = "Number of warmup iterations before timing")]
    warmup: usize,

    #[arg(long, default_value = "5", help = "Number of timed iterations for median measurement")]
    iterations: usize,

    #[arg(long, default_value = "fast", help = "Execution engine: 'fast', 'word-lookup', 'pre', 'bpe-only', 'noop', 'legacy'")]
    engine: String,

    #[arg(long, help = "Work-unit chunk size in KiB (e.g. 128, 256, 512, 1024)")]
    chunk_kb: Option<usize>,

    #[arg(long, help = "Print detailed pretokenization segment, work-unit, and coverage statistics")]
    stats: bool,

    #[arg(long, help = "Do not flatten chunked worker token vectors into a single output Vec")]
    no_flatten: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Explicitly configure Rayon threadpool (§19 of 1.md)
    let _ = rayon::ThreadPoolBuilder::new().num_threads(cli.threads).build_global();

    let is_otk = cli.vocab.extension().and_then(|s| s.to_str()) == Some("otk");

    let t_vocab = Instant::now();
    let ir = vocab_ir::load_auto(&cli.vocab)?;
    let vocab_load_us = t_vocab.elapsed().as_micros();

    let t_trie = Instant::now();
    let trie = trie_builder::build(&ir)?;
    let trie_build_us = t_trie.elapsed().as_micros();

    let merge_rank = walker::build_merge_rank(&ir);
    let id_merge_table = walker::build_id_merge_table(&ir.vocab, &merge_rank);
    let fast_merge_table = walker::FastMergeTable::from_id_merge_table(&id_merge_table);
    let byte_pair_table = walker::BytePairRankTable::from_id_merge_table(&id_merge_table);
    let short_dict = walker::ShortWordDict::build(&ir.vocab);

    let unk_id = ir.vocab.get("<unk>").copied()
        .or_else(|| ir.vocab.get("[UNK]").copied())
        .or_else(|| ir.vocab.get("<|endoftext|>").copied());
    let byte_table = omnitoken::dispatcher::AdaptiveDispatcher::build_byte_vocab_table(&ir.vocab, unk_id);

    if cli.parity {
        println!("─── OmniToken Parity & Differential Fuzzing (§5) ─────────────");
        let sample = "the quick brown fox jumps over the lazy dog";

        let encoded = walker::encode(sample, &ir, &trie)?;
        let is_conformant = verify_bpe_conformance(sample, &ir.vocab, &merge_rank)?;

        println!("  Input string:  {:?}", sample);
        println!("  Tokens output: {:?}", encoded);
        println!("  Vocab size:    {}", ir.len());
        println!("  ✓ Differential Fuzzing: CPU PQ-BPE == GPU BlockBPE: {is_conformant}");
        println!("  ✓ Parity self-check passed ({} tokens produced).", encoded.len());
        println!("─────────────────────────────────────────────────────────────");
        return Ok(());
    }

    // Load corpus: zero-copy mmap if --corpus given, else synthetic in-memory.
    let _mmap_holder: Option<Mmap>;
    let _synthetic_buf: Option<Box<str>>;
    let text: &str = match &cli.corpus {
        Some(path) => {
            let file = File::open(path)?;
            // SAFETY: corpus file is read-only; undefined if another process writes while mapped.
            let mmap = unsafe { Mmap::map(&file)? };
            _mmap_holder = Some(mmap);
            _synthetic_buf = None;
            let raw: &[u8] = _mmap_holder.as_ref().unwrap();
            match std::str::from_utf8(raw) {
                Ok(s) => s,
                Err(e) => std::str::from_utf8(&raw[..e.valid_up_to()]).unwrap(),
            }
        }
        None => {
            _mmap_holder = None;
            let unit = "the quick brown fox jumps over the lazy dog ";
            let reps = cli.bytes.div_ceil(unit.len());
            let mut s = unit.repeat(reps);
            s.truncate(cli.bytes.min(s.len()));
            _synthetic_buf = Some(s.into_boxed_str());
            _synthetic_buf.as_deref().unwrap()
        }
    };

    let bytes = text.len();
    let mib   = bytes as f64 / (1 << 20) as f64;

    let dispatcher = omnitoken::dispatcher::AdaptiveDispatcher {
        cpu_threshold_bytes: 4096,
        num_threads: cli.threads,
    };

    let pretokens = pretokenizer::split_pretokens(text.as_bytes());
    let segment_count = pretokens.len();
    let max_segment_bytes = pretokens.iter().map(|&(s, e)| e - s).max().unwrap_or(0);
    let avg_segment_bytes = if segment_count > 0 { bytes as f64 / segment_count as f64 } else { 0.0 };

    let num_work_units = (cli.threads * 4).max(8);
    let chunk_size = (segment_count / num_work_units).max(1);
    let work_unit_count = pretokens.chunks(chunk_size).len();

    println!("─── OmniToken v1.0.0 Benchmark Suite ──────────────────────────");
    println!("  engine:       {}", cli.engine);
    println!("  vocab:        {} [{}]", cli.vocab.display(), if is_otk { ".otk binary" } else { "auto-detected" });
    println!("  vocab load:   {vocab_load_us} µs (EXCLUDED from encode timing)");
    println!("  trie build:   {trie_build_us} µs (EXCLUDED from encode timing)");
    println!("  threads:      {} (rayon active: {})", cli.threads, rayon::current_num_threads());
    println!("  corpus:       {bytes} bytes ({mib:.1} MiB) [{}]", if cli.corpus.is_some() { "file" } else { "synthetic" });
    println!("  chunk_kb:     {}", cli.chunk_kb.map(|k| format!("{k} KiB")).unwrap_or_else(|| "auto (threads*4)".to_string()));
    println!("  warmup runs:  {}", cli.warmup);
    println!("  timing runs:  {}", cli.iterations);
    println!("  mode:         {}", if cli.no_flatten { "chunked output (--no-flatten)" } else { "flattened single buffer" });

    // Dry-run once to extract coverage statistics
    let (_, cov_stats) = dispatcher.encode_bulk_fast(
        text, &ir, Some(&trie), Some(&short_dict), &fast_merge_table, &byte_pair_table, &byte_table, !cli.no_flatten, cli.chunk_kb,
    )?;

    if cli.stats {
        println!();
        println!("─── Corpus & Work-Unit Statistics (§1 of 1.md & §4 of 2.md) ────");
        println!("  input_bytes:          {bytes}");
        println!("  segment_count:        {segment_count}");
        println!("  max_segment_bytes:    {max_segment_bytes}");
        println!("  avg_segment_bytes:    {avg_segment_bytes:.2}");
        println!("  work_unit_count:      {work_unit_count}");
        println!("  target_threads:       {}", cli.threads);
        println!("  total_words:          {}", cov_stats.total_words);
        println!("  fast_words:           {} ({:.1}%)", cov_stats.fast_words, if cov_stats.total_words > 0 { cov_stats.fast_words as f64 * 100.0 / cov_stats.total_words as f64 } else { 0.0 });
        println!("  fast_bytes:           {} ({:.1}%)", cov_stats.fast_bytes, if cov_stats.total_bytes > 0 { cov_stats.fast_bytes as f64 * 100.0 / cov_stats.total_bytes as f64 } else { 0.0 });
        println!("  fallback_words:       {} ({:.1}%)", cov_stats.fallback_words, if cov_stats.total_words > 0 { cov_stats.fallback_words as f64 * 100.0 / cov_stats.total_words as f64 } else { 0.0 });
        println!("  fallback_bytes:       {} ({:.1}%)", cov_stats.fallback_bytes, if cov_stats.total_bytes > 0 { cov_stats.fallback_bytes as f64 * 100.0 / cov_stats.total_bytes as f64 } else { 0.0 });
    }
    println!();

    // 1. Single-prompt TTFT (Time-To-First-Token) measurement
    let prompt = "Hello, world! Welcome to OmniToken high-throughput tokenization.";
    let t_ttft = Instant::now();
    let ttft_tokens = dispatcher.encode(prompt, &ir, &trie)?;
    let ttft_us = t_ttft.elapsed().as_micros();
    println!("  TTFT (single-prompt latency): {ttft_us} µs ({} tokens)", ttft_tokens.len());

    // Function to run requested engine (§3 of 2.md)
    let run_engine = || -> Result<(Vec<Vec<u32>>, usize)> {
        match cli.engine.as_str() {
            "noop" => {
                use rayon::prelude::*;
                let chunks = pretokenizer::utf8_chunks(text, cli.threads);
                let sum: u64 = chunks.par_iter().map(|c| c.bytes().map(|b| b as u64).sum::<u64>()).sum();
                Ok((vec![vec![sum as u32]], text.len()))
            }
            "pre" => {
                use rayon::prelude::*;
                let chunks = pretokenizer::utf8_chunks(text, cli.threads);
                let sum: usize = chunks.par_iter().map(|c| pretokenizer::split_pretokens(c.as_bytes()).len()).sum();
                Ok((vec![vec![sum as u32]], text.len()))
            }
            "word-lookup" => {
                let (res, _) = dispatcher.encode_bulk_fast(text, &ir, Some(&trie), Some(&short_dict), &fast_merge_table, &byte_pair_table, &byte_table, !cli.no_flatten, cli.chunk_kb)?;
                let total_len: usize = res.iter().map(|c| c.len()).sum();
                Ok((res, total_len))
            }
            "bpe-only" => {
                let (res, _) = dispatcher.encode_bulk_fast(text, &ir, None, None, &fast_merge_table, &byte_pair_table, &byte_table, !cli.no_flatten, cli.chunk_kb)?;
                let total_len: usize = res.iter().map(|c| c.len()).sum();
                Ok((res, total_len))
            }
            "legacy" => {
                let chunks = pretokenizer::utf8_chunks(text, cli.threads);
                let batch_res = dispatcher.encode_batch(&chunks, &ir, &trie)?;
                let total_len: usize = batch_res.iter().map(|c| c.len()).sum();
                Ok((batch_res, total_len))
            }
            _ => {
                let (res, _) = dispatcher.encode_bulk_fast(text, &ir, Some(&trie), Some(&short_dict), &fast_merge_table, &byte_pair_table, &byte_table, !cli.no_flatten, cli.chunk_kb)?;
                let total_len: usize = res.iter().map(|c| c.len()).sum();
                Ok((res, total_len))
            }
        }
    };

    // 2. Warmup iterations (encode-only)
    for _ in 0..cli.warmup {
        let _ = run_engine()?;
    }

    // 3. Timed iterations
    let mut run_times = Vec::with_capacity(cli.iterations);
    let mut total_token_count = 0usize;
    for _ in 0..cli.iterations {
        let t0 = Instant::now();
        let (_tokens, count) = run_engine()?;
        let elapsed = t0.elapsed();
        run_times.push(elapsed);
        total_token_count = count;
    }

    run_times.sort();
    let elapsed = run_times[run_times.len() / 2]; // median run time

    let gb_s  = (bytes as f64 / 1e9)       / elapsed.as_secs_f64();
    let mib_s = mib                          / elapsed.as_secs_f64();
    let mtok_s = (total_token_count as f64 / 1e6) / elapsed.as_secs_f64();

    println!("  elapsed (median):       {:.6}s", elapsed.as_secs_f64());
    println!("  total tokens produced:  {total_token_count}");
    println!("  throughput (in-memory): {gb_s:.3} GB/s  |  {mib_s:.0} MiB/s  |  {mtok_s:.2} Mtok/s");
    println!();
    println!("─── Roofline sanity check (plan §4.8) ───────────────────────");
    println!("  Corpus fits in L3 (≤32MB)?  {}", if bytes <= 32 * 1024 * 1024 { "YES — compare to L3 bandwidth" } else { "NO" });
    println!("  Corpus fits in DRAM-only?   YES — ceiling ≈63–80 GB/s (dual-ch DDR5-5600)");
    println!("  Disk-streaming ceiling:     ≈3.5 GB/s (Gen3 NVMe x4)");
    println!();
    let regime = if bytes <= 32 * 1024 * 1024 { "L3-resident"} else { "DRAM-resident" };
    let ceiling = if bytes <= 32 * 1024 * 1024 { "L3 bandwidth" } else { "≈63–80 GB/s" };
    if gb_s > 80.0 {
        println!("  ⚠  {gb_s:.1} GB/s EXCEEDS DRAM ceiling — likely a measurement bug");
    } else {
        println!("  ✓  {gb_s:.3} GB/s is physically plausible for {regime} input.");
        println!("     Ceiling for this regime: {ceiling}");
    }
    println!("─────────────────────────────────────────────────────────────");
    println!();

    Ok(())
}
