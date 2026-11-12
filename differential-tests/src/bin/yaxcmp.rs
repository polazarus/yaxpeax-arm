//! `yaxcmp`: compare yaxpeax-arm against capstone over a range of encodings.
//!
//! reports every category of agreement/disagreement, keeps going on mismatch,
//! and exits nonzero only if there were mismatches (or, with
//! `--fail-on-rejections`, instructions yaxpeax-arm didn't decode).
//!
//! examples:
//!
//! ```text
//! yaxcmp --arch a64 --range 0x94000000..0x94100000
//! yaxcmp --arch thumb --start 0x4000 --end 0x5000 --threads 8
//! yaxcmp --arch a64 --words regressions.txt --json
//! yaxcmp --arch armv7 --sample 100000000
//! ```

use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};

use differential_tests::report::{Category, Tally};
use differential_tests::{compare_words, run, ArchKind, CompareMode, Config};

const USAGE: &str = "\
usage: yaxcmp [options]

  --arch <a64|armv7|thumb>     architecture to compare (default: a64)
  --range <start..end>         encoding range, inclusive, hex or decimal
  --start <n>                  first encoding, inclusive (default: 0)
  --end <n>                    last encoding, inclusive (default: 0xffffffff)
  --sample <n>                 compare `n` randomly chosen encodings instead
  --seed <n>                   seed for --sample (default: 1)
  --words <file|w1,w2,..>      compare specific encodings (hex, or a file with
                               one word per line, `#` comments allowed)
  --threads <n>                worker threads (default: rayon's choice)
  --chunk-size <n>             encodings per task (default: 65536)
  --max-examples <n>           examples kept per mnemonic pair (default: 3)
  --compare <semantic|text>    compare opcode + typed operands (default), or the
                               rendered disassembly text
  --no-equivalence             only byte-identical text counts as a match
  --fail-on-rejections         also exit nonzero when yaxpeax-arm can't decode
                               something capstone can
  --json                       machine-readable report on stdout
  --quiet                      no progress on stderr
  -h, --help                   this help

exit status is 0 when there are no mismatches, 1 otherwise.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match Options::parse(&args) {
        Ok(Some(opts)) => opts,
        Ok(None) => {
            println!("{}", USAGE);
            return ExitCode::SUCCESS;
        }
        Err(err) => {
            eprintln!("yaxcmp: {}\n\n{}", err, USAGE);
            return ExitCode::from(2);
        }
    };

    let target = opts.config.arch.target();
    if !opts.quiet {
        eprintln!(
            "yaxcmp: {} vs capstone, {}",
            target.name(),
            opts.describe()
        );
    }

    let tally = match &opts.words {
        Some(words) => compare_words(&opts.config, words),
        None => {
            let progress = opts.progress();
            run(&opts.config, progress.as_deref())
        }
    };

    if opts.json {
        print!("{}", report_json(&opts, &tally));
    } else {
        print!("{}", report_text(&opts, &tally));
    }

    let failures = tally.failures()
        + if opts.fail_on_rejections {
            tally.count(Category::YaxRejectUnexpected)
        } else {
            0
        };

    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

struct Options {
    config: Config,
    words: Option<Vec<u32>>,
    json: bool,
    quiet: bool,
    fail_on_rejections: bool,
    /// original range/sample description, for the report header.
    what: String,
}

impl Options {
    fn parse(args: &[String]) -> Result<Option<Options>, String> {
        let mut config = Config::default();
        let mut words = None;
        let mut json = false;
        let mut quiet = false;
        let mut fail_on_rejections = false;
        let mut sample: Option<(u64, u64)> = None;
        let mut have_range = false;
        let mut start_given = false;
        let mut end_given = false;

        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_str();

            match arg {
                "-h" | "--help" => return Ok(None),
                "--arch" => {
                    let value = take(args, &mut i, "--arch")?;
                    config.arch = ArchKind::parse(&value)
                        .ok_or_else(|| format!("unknown architecture: {}", value))?;
                }
                "--range" => {
                    let value = take(args, &mut i, "--range")?;
                    let (start, end) = parse_range(&value)?;
                    config.start = start;
                    config.end = end;
                    have_range = true;
                }
                "--start" => {
                    config.start = parse_number(&take(args, &mut i, "--start")?)?;
                    start_given = true;
                }
                "--end" => {
                    config.end = parse_number(&take(args, &mut i, "--end")?)?;
                    end_given = true;
                }
                "--sample" => {
                    let count = parse_number(&take(args, &mut i, "--sample")?)?;
                    sample = Some((count, 1));
                }
                "--seed" => {
                    let seed = parse_number(&take(args, &mut i, "--seed")?)?;
                    match sample.as_mut() {
                        Some(sample) => sample.1 = seed,
                        None => return Err("--seed only applies to --sample".to_string()),
                    }
                }
                "--words" => {
                    let value = take(args, &mut i, "--words")?;
                    words = Some(load_words(&value)?);
                }
                "--threads" => {
                    let threads = parse_number(&take(args, &mut i, "--threads")?)?;
                    config.threads = threads as usize;
                }
                "--chunk-size" => {
                    let size = parse_number(&take(args, &mut i, "--chunk-size")?)?;
                    config.chunk_size = size.max(1);
                }
                "--max-examples" => {
                    let count = parse_number(&take(args, &mut i, "--max-examples")?)?;
                    config.max_examples = count as usize;
                }
                "--compare" => {
                    config.compare = match take(args, &mut i, "--compare")?.as_str() {
                        "semantic" | "semantics" => CompareMode::Semantic,
                        "text" => CompareMode::Text,
                        other => return Err(format!("unknown comparison mode: {}", other)),
                    };
                }
                "--no-equivalence" => config.use_equivalence = false,
                "--fail-on-rejections" => fail_on_rejections = true,
                "--json" => json = true,
                "--quiet" => quiet = true,
                other => return Err(format!("unknown option: {}", other)),
            }

            i += 1;
        }

        if have_range && (start_given || end_given) {
            return Err("--range conflicts with --start/--end".to_string());
        }
        if words.is_some() && (have_range || start_given || end_given || sample.is_some()) {
            return Err("--words conflicts with --range/--start/--end/--sample".to_string());
        }
        if config.end < config.start {
            return Err(format!(
                "empty range: start {:#x} is after end {:#x}",
                config.start, config.end
            ));
        }
        // encodings are 32-bit words; a larger bound would silently wrap.
        if config.end > u32::MAX as u64 {
            return Err(format!(
                "end {:#x} is past the 32-bit encoding space",
                config.end
            ));
        }

        let what = match (&words, sample) {
            (Some(words), _) => format!("{} encodings from a word list", words.len()),
            (None, Some((count, seed))) => format!("{} sampled encodings (seed {})", count, seed),
            (None, None) => format!(
                "{:#010x}..={:#010x} ({} encodings)",
                config.start,
                config.end,
                config.end - config.start + 1
            ),
        };

        if let (None, Some((count, seed))) = (&words, sample) {
            let words = sample_words(&config, count, seed);
            let mut config = config;
            config.start = 0;
            config.end = 0;
            return Ok(Some(Options {
                config,
                words: Some(words),
                json,
                quiet,
                fail_on_rejections,
                what,
            }));
        }

        Ok(Some(Options {
            config,
            words,
            json,
            quiet,
            fail_on_rejections,
            what,
        }))
    }

    fn describe(&self) -> String {
        let comparison = match self.config.compare {
            CompareMode::Semantic if self.config.arch.target().supports_semantics() => "semantic",
            CompareMode::Semantic => "text (no semantic model for this arch yet)",
            CompareMode::Text => "text",
        };
        format!("{}, comparing {}", self.what, comparison)
    }

    /// progress callback for a full sweep, throttled to one line per 5%.
    fn progress(&self) -> Option<Box<dyn Fn(u64, u64) + Sync>> {
        if self.quiet {
            return None;
        }
        let last_percent = AtomicU64::new(0);
        Some(Box::new(move |done: u64, total: u64| {
            let percent = if total == 0 { 100 } else { done * 100 / total };
            let previous = last_percent.load(Ordering::Relaxed);
            if percent >= previous + 5 {
                last_percent.store(percent, Ordering::Relaxed);
                eprintln!("  {}% ({}/{})", percent, done, total);
            }
        }))
    }
}

fn take(args: &[String], i: &mut usize, name: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{} needs a value", name))
}

fn parse_number(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let parsed = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(&hex.replace('_', ""), 16)
    } else {
        s.replace('_', "").parse::<u64>()
    };
    parsed.map_err(|_| format!("not a number: {:?}", s))
}

fn parse_range(s: &str) -> Result<(u64, u64), String> {
    let (start, end) = s
        .split_once("..")
        .ok_or_else(|| format!("expected `start..end`, got {:?}", s))?;
    let start = parse_number(start)?;
    let end = if end.is_empty() {
        u32::MAX as u64
    } else {
        parse_number(end)?
    };
    Ok((start, end))
}

/// `--words` is either a comma-separated list or a path to a file with one
/// encoding per line. in either case `#` starts a comment.
fn load_words(value: &str) -> Result<Vec<u32>, String> {
    let is_path = !value.contains(',') && std::path::Path::new(value).is_file();
    let text = if is_path {
        std::fs::read_to_string(value).map_err(|e| format!("can't read {}: {}", value, e))?
    } else {
        value.to_string()
    };

    let mut words = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        for item in line.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let word = parse_number(item).map_err(|_| {
                format!(
                    "not an encoding on line {}: {:?}",
                    line_no + 1,
                    item
                )
            })?;
            if word > u32::MAX as u64 {
                return Err(format!(
                    "encoding {:#x} on line {} doesn't fit in 32 bits",
                    word,
                    line_no + 1
                ));
            }
            words.push(word as u32);
        }
    }

    if words.is_empty() {
        return Err("no encodings in --words".to_string());
    }
    Ok(words)
}

/// deterministic pseudo-random sample of the configured range. xorshift-ish, so
/// there is no dependency just for this.
fn sample_words(config: &Config, count: u64, seed: u64) -> Vec<u32> {
    let span = config.end - config.start + 1;
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut words = Vec::with_capacity(count.min(1 << 24) as usize);
    for _ in 0..count.min(span) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        words.push((config.start + state % span) as u32);
    }
    words
}

fn percent(count: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * count as f64 / total as f64
    }
}

fn report_text(opts: &Options, tally: &Tally) -> String {
    let total = tally.total();
    let mut out = String::new();

    out.push_str(&format!(
        "\n{}: {}\n\n",
        opts.config.arch.target().name(),
        opts.describe()
    ));
    out.push_str(&format!("  {:<24}{:>12}\n", "encodings compared", total));
    for category in Category::ALL.iter() {
        let count = tally.count(*category);
        out.push_str(&format!(
            "  {:<24}{:>12}  {:>6.2}%\n",
            category.name(),
            count,
            percent(count, total)
        ));
    }

    let buckets = tally.sorted_buckets();
    if buckets.is_empty() {
        out.push_str("\n  no disagreements\n");
        return out;
    }

    // mismatches first: those are the ones where neither decoder is obviously
    // wrong. everything else is, by construction, an explained divergence.
    for category in [
        Category::Panic,
        Category::OpcodeMismatch,
        Category::OperandMismatch,
        Category::Mismatch,
        Category::YaxRejectUnexpected,
        Category::CsReject,
        Category::OpcodeAlias,
        Category::Unmodeled,
        Category::AcceptedDiff,
    ]
    .iter()
    {
        let of_category: Vec<_> = buckets
            .iter()
            .filter(|bucket| bucket.category == *category)
            .collect();
        if of_category.is_empty() {
            continue;
        }

        out.push_str(&format!("\n  {}:\n", category.name()));
        for bucket in of_category.iter().take(25) {
            out.push_str(&format!(
                "    {:>10}  yax {:<16} capstone {}\n",
                bucket.count, bucket.yax, bucket.cs
            ));
            for example in &bucket.examples {
                let bytes = example.word.to_le_bytes();
                let yax = if example.yax.is_empty() {
                    "<rejected>".to_string()
                } else {
                    example.yax.clone()
                };
                let cs = if example.cs.is_empty() {
                    "<rejected>".to_string()
                } else {
                    example.cs.clone()
                };
                let len = example
                    .cs_len
                    .map(|len| format!("{}B", len))
                    .unwrap_or_else(|| "-".to_string());
                out.push_str(&format!(
                    "              {:08x} [{:02x} {:02x} {:02x} {:02x}] ({})\n",
                    example.word, bytes[0], bytes[1], bytes[2], bytes[3], len
                ));
                out.push_str(&format!("                yax: {}\n", yax));
                out.push_str(&format!("                cs:  {}\n", cs));
                if !example.note.is_empty() {
                    out.push_str(&format!("                cmp: {}\n", example.note));
                }
            }
        }
        if of_category.len() > 25 {
            out.push_str(&format!(
                "    .. and {} more distinct mnemonic pairs\n",
                of_category.len() - 25
            ));
        }
    }

    out
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn report_json(opts: &Options, tally: &Tally) -> String {
    let total = tally.total();
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"arch\": {},\n", json_string(opts.config.arch.name())));
    out.push_str(&format!("  \"what\": {},\n", json_string(&opts.describe())));
    out.push_str("  \"counts\": {\n");
    let counts: Vec<String> = Category::ALL
        .iter()
        .map(|category| {
            format!(
                "    {}: {}",
                json_string(category.name()),
                tally.count(*category)
            )
        })
        .collect();
    out.push_str(&counts.join(",\n"));
    out.push_str("\n  },\n");
    out.push_str(&format!("  \"total\": {},\n", total));
    out.push_str("  \"buckets\": [\n");
    let buckets: Vec<String> = tally
        .sorted_buckets()
        .iter()
        .map(|bucket| {
            let examples: Vec<String> = bucket
                .examples
                .iter()
                .map(|example| {
                    format!(
                        "{{\"word\": \"{:#010x}\", \"yax\": {}, \"cs\": {}, \"note\": {}, \"cs_len\": {}}}",
                        example.word,
                        json_string(&example.yax),
                        json_string(&example.cs),
                        json_string(&example.note),
                        example
                            .cs_len
                            .map(|len| len.to_string())
                            .unwrap_or_else(|| "null".to_string())
                    )
                })
                .collect();
            format!(
                "    {{\"category\": {}, \"yax\": {}, \"cs\": {}, \"count\": {}, \"examples\": [{}]}}",
                json_string(bucket.category.name()),
                json_string(&bucket.yax),
                json_string(&bucket.cs),
                bucket.count,
                examples.join(", ")
            )
        })
        .collect();
    out.push_str(&buckets.join(",\n"));
    out.push_str("\n  ]\n}\n");
    out
}
