//! aggregation for a differential run: per-category counts and per-mnemonic
//! buckets of disagreements, with a bounded number of examples per bucket so a
//! full 2^32 sweep doesn't retain 4 GiB of mismatch text.

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// both decoders produced byte-identical text.
    Match,
    /// text differs, but the divergence is one the arch module knows about and
    /// has a rule for (see the arch modules' `acceptable_match`).
    AcceptedDiff,
    /// both decoders produced text, incompably different.
    Mismatch,
    /// capstone decoded; yaxpeax-arm errored in a way the arch module considers
    /// a known gap (SVE, MTE, coprocessor, ..).
    YaxRejectKnown,
    /// capstone decoded; yaxpeax-arm errored and we have no explanation.
    YaxRejectUnexpected,
    /// yaxpeax-arm decoded; capstone rejected the encoding.
    CsReject,
    /// neither decoder accepted the encoding.
    BothReject,
    /// yaxpeax-arm needs more bytes than were provided.
    Incomplete,
    /// yaxpeax-arm panicked while decoding. always a bug; recorded (with the
    /// panic message) rather than taking down the sweep.
    Panic,
    /// same instruction under a different alias (`mov` vs `orr`).
    OpcodeAlias,
    /// both decoded, but to different opcodes.
    OpcodeMismatch,
    /// same opcode, different operand kinds/values/count.
    OperandMismatch,
    /// at least one side used an operand representation the semantic comparison
    /// doesn't model, so the instruction wasn't compared structurally.
    Unmodeled,
}

impl Category {
    pub const COUNT: usize = 13;

    pub const ALL: [Category; Category::COUNT] = [
        Category::Match,
        Category::AcceptedDiff,
        Category::Mismatch,
        Category::YaxRejectKnown,
        Category::YaxRejectUnexpected,
        Category::CsReject,
        Category::BothReject,
        Category::Incomplete,
        Category::Panic,
        Category::OpcodeAlias,
        Category::OpcodeMismatch,
        Category::OperandMismatch,
        Category::Unmodeled,
    ];

    pub fn index(self) -> usize {
        match self {
            Category::Match => 0,
            Category::AcceptedDiff => 1,
            Category::Mismatch => 2,
            Category::YaxRejectKnown => 3,
            Category::YaxRejectUnexpected => 4,
            Category::CsReject => 5,
            Category::BothReject => 6,
            Category::Incomplete => 7,
            Category::Panic => 8,
            Category::OpcodeAlias => 9,
            Category::OpcodeMismatch => 10,
            Category::OperandMismatch => 11,
            Category::Unmodeled => 12,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Category::Match => "match",
            Category::AcceptedDiff => "accepted_diff",
            Category::Mismatch => "mismatch",
            Category::YaxRejectKnown => "yax_reject_known",
            Category::YaxRejectUnexpected => "yax_reject_unexpected",
            Category::CsReject => "cs_reject",
            Category::BothReject => "both_reject",
            Category::Incomplete => "incomplete",
            Category::Panic => "panic",
            Category::OpcodeAlias => "opcode_alias",
            Category::OpcodeMismatch => "opcode_mismatch",
            Category::OperandMismatch => "operand_mismatch",
            Category::Unmodeled => "unmodeled",
        }
    }

    /// whether a nonzero count here means the tool should exit nonzero.
    ///
    /// disagreements between two decoders are failures; yaxpeax-arm not
    /// implementing an instruction is reported but is a known gap unless the
    /// caller asks to fail on rejections too.
    pub fn is_failure(self) -> bool {
        matches!(
            self,
            Category::Mismatch
                | Category::Panic
                | Category::OpcodeMismatch
                | Category::OperandMismatch
        )
    }

    /// whether this category is only interesting as a diagnostic.
    pub fn is_informational(self) -> bool {
        matches!(self, Category::OpcodeAlias | Category::Unmodeled)
    }

    /// whether this category is yaxpeax-arm failing to decode something
    /// capstone decoded.
    pub fn is_rejection(self) -> bool {
        self == Category::YaxRejectUnexpected
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}



#[derive(Clone, Debug)]
pub struct Example {
    pub word: u32,
    pub yax: String,
    pub cs: String,
    /// bytes capstone consumed, when it decoded. 2 vs 4 is the interesting bit
    /// in thumb mode.
    pub cs_len: Option<usize>,
    /// what the semantic comparison found, when it ran.
    pub note: String,
}

/// one (category, yax mnemonic, capstone mnemonic) group.
#[derive(Clone, Debug)]
pub struct Bucket {
    pub category: Category,
    /// mnemonic from the yaxpeax-arm side, or `<reject>`/`panic`.
    pub yax: String,
    /// mnemonic from the capstone side, or `<reject>`.
    pub cs: String,
    pub count: u64,
    pub examples: Vec<Example>,
}

#[derive(Clone, Debug)]
pub struct Tally {
    pub counts: [u64; Category::COUNT],
    /// grouped results, keyed by a `category\0yax-mnemonic\0cs-mnemonic` string
    /// so that lookups borrow instead of allocating: most encodings in a sweep
    /// land in a bucket that already exists, and this is the hot path.
    buckets: HashMap<String, Bucket>,
    /// maximum examples retained per bucket.
    pub max_examples: usize,
    /// scratch buffer for bucket keys, so `record` doesn't allocate on a hit.
    key: String,
}

impl Default for Tally {
    fn default() -> Self {
        Tally::with_max_examples(3)
    }
}

impl Tally {
    pub fn with_max_examples(max_examples: usize) -> Self {
        Tally {
            counts: [0; Category::COUNT],
            buckets: HashMap::new(),
            max_examples,
            key: String::new(),
        }
    }

    pub fn count(&self, category: Category) -> u64 {
        self.counts[category.index()]
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// record a comparison result. `yax_text`/`cs_text` are the full
    /// disassembly texts; `None` means that decoder rejected the encoding.
    pub fn record(
        &mut self,
        category: Category,
        word: u32,
        yax_text: Option<&str>,
        cs_text: Option<&str>,
        cs_len: Option<usize>,
    ) {
        self.record_with_note(category, word, yax_text, cs_text, cs_len, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_with_note(
        &mut self,
        category: Category,
        word: u32,
        yax_text: Option<&str>,
        cs_text: Option<&str>,
        cs_len: Option<usize>,
        note: Option<&str>,
    ) {
        self.counts[category.index()] += 1;

        // `match` needs no bucketing; everything else is something a human
        // should be able to look at.
        if category == Category::Match {
            return;
        }

        // the "text" for a panic is the panic message, which has no mnemonic to
        // group by.
        let yax = if category == Category::Panic {
            "panic"
        } else {
            yax_text.map(mnemonic).unwrap_or("<reject>")
        };
        let cs = cs_text.map(mnemonic).unwrap_or("<reject>");

        let key = &mut self.key;
        key.clear();
        key.push_str(category.name());
        key.push('\0');
        key.push_str(yax);
        key.push('\0');
        key.push_str(cs);

        match self.buckets.get_mut(key.as_str()) {
            Some(bucket) => {
                bucket.count += 1;
                if bucket.examples.len() < self.max_examples {
                    bucket.examples.push(example(word, yax_text, cs_text, cs_len, note));
                }
            }
            None => {
                // first time this mnemonic pair is seen: this is the only
                // allocation per bucket.
                let mut bucket = Bucket {
                    category,
                    yax: yax.to_string(),
                    cs: cs.to_string(),
                    count: 1,
                    examples: Vec::new(),
                };
                if self.max_examples > 0 {
                    bucket.examples.push(example(word, yax_text, cs_text, cs_len, note));
                }
                self.buckets.insert(key.clone(), bucket);
            }
        }
    }

    pub fn merge(mut self, other: Tally) -> Tally {
        for (i, count) in other.counts.iter().enumerate() {
            self.counts[i] += count;
        }

        let max_examples = self.max_examples.min(other.max_examples);
        self.max_examples = max_examples;

        for (key, mut other_bucket) in other.buckets {
            match self.buckets.get_mut(key.as_str()) {
                Some(bucket) => {
                    bucket.count += other_bucket.count;
                    let room = max_examples.saturating_sub(bucket.examples.len());
                    bucket
                        .examples
                        .extend(other_bucket.examples.drain(..).take(room));
                }
                None => {
                    let room = max_examples;
                    other_bucket.examples.truncate(room);
                    self.buckets.insert(key, other_bucket);
                }
            }
        }

        self
    }

    /// buckets ordered by descending count (ties broken by category and
    /// mnemonics, so output is stable across runs and across thread counts).
    pub fn sorted_buckets(&self) -> Vec<&Bucket> {
        let mut buckets: Vec<_> = self.buckets.values().collect();
        buckets.sort_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| a.category.cmp(&b.category))
                .then_with(|| a.yax.cmp(&b.yax))
                .then_with(|| a.cs.cmp(&b.cs))
        });
        buckets
    }

    pub fn mismatches(&self) -> u64 {
        self.count(Category::Mismatch)
    }

    pub fn failures(&self) -> u64 {
        Category::ALL
            .iter()
            .filter(|category| category.is_failure())
            .map(|category| self.count(*category))
            .sum()
    }
}

fn example(
    word: u32,
    yax_text: Option<&str>,
    cs_text: Option<&str>,
    cs_len: Option<usize>,
    note: Option<&str>,
) -> Example {
    Example {
        word,
        yax: yax_text.unwrap_or("").to_string(),
        cs: cs_text.unwrap_or("").to_string(),
        cs_len,
        note: note.unwrap_or("").to_string(),
    }
}

/// first whitespace-delimited token of a disassembly. no-operand instructions
/// keep capstone's trailing space, so `trim_end` matters here.
pub fn mnemonic(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or("")
}
