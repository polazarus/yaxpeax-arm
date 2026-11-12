//! differential comparison of `yaxpeax-arm` against capstone.
//!
//! this crate is deliberately separate from `yaxpeax-arm` proper: capstone is a
//! C dependency and has no business in the disassembler's dependency tree.
//!
//! the pieces:
//!
//! - [`capstone`]: a per-thread capstone handle.
//! - [`report`]: counts and bounded example retention for a run.
//! - arch modules ([`a64`], [`armv7`], [`armv7_thumb`]): yaxpeax-arm decode
//!   glue, the disassembly-text parser, and the rules that describe which
//!   divergences are *acceptable* rather than bugs on either side.
//! - [`Config`]/[`run`]: a rayon-parallel sweep over an encoding range.
//!
//! `yaxcmp` (in `src/bin`) is the command line front end.

pub mod a64;
pub mod armv7;
pub mod armv7_thumb;
pub mod capstone;
pub mod report;
pub mod semantic;

use capstone::{ArchMode, Capstone};
use rayon::prelude::*;
use report::{Category, Tally};
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchKind {
    A64,
    Armv7,
    Armv7Thumb,
}

impl ArchKind {
    pub fn parse(s: &str) -> Option<ArchKind> {
        match s {
            "a64" | "armv8" | "v8" | "arm64" => Some(ArchKind::A64),
            "armv7" | "v7" | "arm" | "a32" => Some(ArchKind::Armv7),
            "thumb" | "thumbv7" | "t32" => Some(ArchKind::Armv7Thumb),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ArchKind::A64 => "a64",
            ArchKind::Armv7 => "armv7",
            ArchKind::Armv7Thumb => "thumb",
        }
    }

    pub fn target(self) -> &'static dyn Target {
        match self {
            ArchKind::A64 => &a64::TARGET,
            ArchKind::Armv7 => &armv7::TARGET,
            ArchKind::Armv7Thumb => &armv7_thumb::TARGET,
        }
    }
}

/// why yaxpeax-arm did not produce an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectKind {
    /// more bytes were needed. expected when sweeping single words in thumb
    /// mode; a 32-bit thumb encoding is only reachable if a full word is given,
    /// which this tool always does.
    Incomplete,
    /// the encoding is explicitly undefined.
    Undefined,
    /// the encoding has unpredictable behavior.
    Unpredictable,
    /// reserved bits were not as required.
    Nonconforming,
    /// yaxpeax-arm doesn't implement this instruction (yet).
    Unimplemented,
    /// any other decode failure.
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareMode {
    /// compare rendered disassembly text (with per-arch equivalences).
    Text,
    /// compare opcode and typed operands.
    Semantic,
}

/// result of a yaxpeax-arm decode.
pub enum YaxOutcome {
    /// decoded; display text was written to the provided buffer, and the
    /// semantic form is available when the arch module models it.
    Decoded(Option<semantic::Semantics>),
    Rejected(RejectKind),
    /// yaxpeax-arm panicked; the contained string is the panic message. a
    /// decoder that panics on attacker-controlled encodings is a defect, but
    /// one that shouldn't take down a 2^32 sweep.
    Panicked(String),
}

/// one architecture's comparison policy.
pub trait Target: Sync {
    fn name(&self) -> &'static str;

    /// capstone architecture/mode to open for this target.
    fn mode(&self) -> ArchMode;

    /// decode one instruction from `bytes` with yaxpeax-arm, writing its
    /// display text into `out`.
    fn yax_decode(&self, bytes: &[u8], out: &mut String) -> YaxOutcome;

    /// whether this arch has a semantic (opcode + operand) model.
    fn supports_semantics(&self) -> bool {
        false
    }

    /// whether differing text for `word` is a known-acceptable divergence
    /// (aliasing, formatting, documented yaxpeax-arm or capstone quirks).
    fn acceptable_match(&self, word: u32, yax_text: &str, cs_text: &str) -> bool;

    /// whether yaxpeax-arm rejecting `word` while capstone decodes `cs_text` is
    /// a known gap rather than a defect.
    fn expected_rejection(&self, word: u32, kind: RejectKind, cs_text: &str) -> bool;
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: ArchKind,
    /// first encoding to test, inclusive.
    pub start: u64,
    /// last encoding to test, inclusive.
    pub end: u64,
    /// encodings per rayon task. small enough to load-balance, large enough
    /// that per-chunk capstone setup disappears.
    pub chunk_size: u64,
    pub threads: usize,
    /// apply arch `acceptable_match` rules. with this off, only byte-identical
    /// text counts as a match.
    pub use_equivalence: bool,
    /// examples retained per mnemonic-pair bucket.
    pub max_examples: usize,
    pub compare: CompareMode,
}

impl Config {
    /// capstone only needs operand detail for the semantic comparison.
    fn use_detail(&self) -> bool {
        self.compare == CompareMode::Semantic
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            arch: ArchKind::A64,
            start: 0,
            end: u32::MAX as u64,
            chunk_size: 1 << 16,
            threads: 0,
            use_equivalence: true,
            max_examples: 3,
            compare: CompareMode::Semantic,
        }
    }
}

/// sweep `config.start..=config.end` with rayon, returning aggregated results.
///
/// `progress`, if given, is called from worker threads with
/// `(encodings_done, encodings_total)` roughly once per chunk. it must be cheap
/// and must not block.
pub fn run(config: &Config, progress: Option<&(dyn Fn(u64, u64) + Sync)>) -> Tally {
    install_panic_hook();
    let chunks = chunk_ranges(config);
    let total: u64 = chunks.iter().map(|&(start, end)| end - start + 1).sum();
    let done = AtomicU64::new(0);
    let target = config.arch.target();

    let sweep = || {
        chunks
            .par_iter()
            .map(|&(start, end)| {
                let tally = compare_range(target, config, start, end);
                let finished = done.fetch_add(end - start + 1, Ordering::Relaxed) + (end - start + 1);
                if let Some(progress) = progress {
                    progress(finished, total);
                }
                tally
            })
            .reduce(Tally::default, Tally::merge)
    };

    with_pool(config.threads, sweep)
}

/// compare a specific set of encodings (a regression list, `--words FILE`).
pub fn compare_words(config: &Config, words: &[u32]) -> Tally {
    install_panic_hook();
    let target = config.arch.target();
    let compare = || {
        words
            .par_chunks(WORDS_PER_TASK)
            .map(|words| {
                let mut tally = Tally::with_max_examples(config.max_examples);
                // one capstone handle per task, not per encoding.
                let mut cs = Capstone::new_with_detail(target.mode(), config.use_detail())
                    .expect("can open capstone");
                for &word in words {
                    compare_word(target, config, &mut cs, word, &mut tally);
                }
                tally
            })
            .reduce(Tally::default, Tally::merge)
    };

    with_pool(config.threads, compare)
}

/// run `work` inside a rayon pool of `threads` workers (0 = rayon's default).
fn with_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    if threads == 0 {
        work()
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("can build a rayon thread pool")
            .install(work)
    }
}

thread_local! {
    static PANIC_MESSAGE: RefCell<Option<String>> = RefCell::new(None);
}

static PANIC_HOOK: Once = Once::new();

/// yaxpeax-arm panics on some malformed encodings (an out-of-range register
/// number, for example). those are caught per-instruction and reported as
/// [`Category::Panic`]; this hook keeps millions of panics from flooding stderr
/// while still capturing the message for the report.
fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        panic::set_hook(Box::new(|info| {
            let payload = info.payload();
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| message.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked with a non-string payload".to_string());
            let location = info
                .location()
                .map(|location| format!(" at {}:{}", location.file(), location.line()))
                .unwrap_or_default();
            PANIC_MESSAGE.with(|slot| *slot.borrow_mut() = Some(format!("{}{}", message, location)));
        }));
    });
}

fn take_panic_message() -> String {
    PANIC_MESSAGE
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_else(|| "panicked without a message".to_string())
}

/// encodings per rayon task when comparing an explicit word list.
const WORDS_PER_TASK: usize = 4096;

fn chunk_ranges(config: &Config) -> Vec<(u64, u64)> {
    let chunk_size = config.chunk_size.max(1);
    let mut chunks = Vec::new();
    let mut start = config.start;
    while start <= config.end {
        let end = (start + chunk_size - 1).min(config.end);
        chunks.push((start, end));
        start = end + 1;
    }
    chunks
}

fn compare_range(target: &dyn Target, config: &Config, start: u64, end: u64) -> Tally {
    let mut tally = Tally::with_max_examples(config.max_examples);
    // capstone handles are per-thread state, and a rayon task is one thread's
    // work at a time, so one handle per task is both correct and cheapest.
    let mut cs = Capstone::new_with_detail(target.mode(), config.use_detail())
        .unwrap_or_else(|e| panic!("can't open capstone for {}: {}", target.name(), e));
    let mut cs_text = String::new();
    let mut yax_text = String::new();

    for word in start..=end {
        let word = word as u32;
        cs_text.clear();
        let cs_len = cs.disasm(&word.to_le_bytes(), &mut cs_text);
        compare_word_with(target, config, &cs, word, cs_len, &cs_text, &mut yax_text, &mut tally);
    }

    tally
}

fn compare_word(
    target: &dyn Target,
    config: &Config,
    cs: &mut Capstone,
    word: u32,
    tally: &mut Tally,
) {
    let mut cs_text = String::new();
    let mut yax_text = String::new();
    let cs_len = cs.disasm(&word.to_le_bytes(), &mut cs_text);
    compare_word_with(target, config, cs, word, cs_len, &cs_text, &mut yax_text, tally);
}

#[allow(clippy::too_many_arguments)]
fn compare_word_with(
    target: &dyn Target,
    config: &Config,
    cs: &Capstone,
    word: u32,
    cs_len: Option<usize>,
    cs_text: &str,
    yax_text: &mut String,
    tally: &mut Tally,
) {
    yax_text.clear();
    let yax = match panic::catch_unwind(AssertUnwindSafe(|| {
        target.yax_decode(&word.to_le_bytes(), yax_text)
    })) {
        Ok(outcome) => outcome,
        Err(_) => YaxOutcome::Panicked(take_panic_message()),
    };
    let cs_decoded = cs_len.is_some();

    let semantic = config.compare == CompareMode::Semantic && target.supports_semantics();

    let (category, yax_str, note) = match (&yax, cs_decoded) {
        (YaxOutcome::Decoded(semantics), true) => {
            let yax_str: &str = yax_text;
            if semantic {
                match compare_semantics(cs, semantics.as_ref()) {
                    (category, note) => (category, Some(yax_str), Some(note)),
                }
            } else {
                let category = if yax_str == cs_text {
                    Category::Match
                } else if config.use_equivalence && target.acceptable_match(word, yax_str, cs_text) {
                    Category::AcceptedDiff
                } else {
                    Category::Mismatch
                };
                (category, Some(yax_str), None)
            }
        }
        (YaxOutcome::Decoded(_), false) => (Category::CsReject, Some(yax_text.as_str()), None),
        (YaxOutcome::Rejected(RejectKind::Incomplete), _) => (Category::Incomplete, None, None),
        (YaxOutcome::Panicked(message), _) => (Category::Panic, Some(message.as_str()), None),
        (YaxOutcome::Rejected(kind), true) => {
            let category = if target.expected_rejection(word, *kind, cs_text) {
                Category::YaxRejectKnown
            } else {
                Category::YaxRejectUnexpected
            };
            (category, None, None)
        }
        (YaxOutcome::Rejected(_), false) => (Category::BothReject, None, None),
    };

    tally.record_with_note(
        category,
        word,
        yax_str,
        if cs_decoded { Some(cs_text) } else { None },
        cs_len,
        note.as_deref(),
    );
}

/// map a semantic comparison result onto a report category, carrying the
/// verdict text along as the example's note.
fn compare_semantics(cs: &Capstone, yax: Option<&semantic::Semantics>) -> (Category, String) {
    let yax = match yax {
        Some(yax) => yax,
        None => {
            return (
                Category::Unmodeled,
                "unmodeled (yaxpeax-arm has no operand model for this instruction)".to_string(),
            )
        }
    };

    let cs_semantics = match semantic::from_capstone_arm64(cs) {
        Some(cs_semantics) => cs_semantics,
        None => {
            return (
                Category::Unmodeled,
                "unmodeled (no capstone operand detail for this instruction)".to_string(),
            )
        }
    };

    let verdict = semantic::compare(yax, &cs_semantics);
    let category = match &verdict {
        semantic::Verdict::Match => Category::Match,
        semantic::Verdict::OpcodeAlias { .. } => Category::OpcodeAlias,
        semantic::Verdict::OpcodeMismatch { .. } => Category::OpcodeMismatch,
        semantic::Verdict::OperandCount { .. }
        | semantic::Verdict::OperandKind { .. }
        | semantic::Verdict::OperandValue { .. }
        | semantic::Verdict::Condition { .. } => Category::OperandMismatch,
        semantic::Verdict::Unmodeled { .. } => Category::Unmodeled,
    };

    (category, verdict.describe())
}
