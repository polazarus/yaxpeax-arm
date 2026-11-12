differential testing between `yaxpeax-arm` and capstone.

this directory is a separate crate so that capstone (a C library, and a fairly
large one) stays out of `yaxpeax-arm`'s dependency tree.

## `yaxcmp`

`yaxcmp` compares the two decoders over a range of encodings and reports
everything it finds. it keeps going after a disagreement, so one run gives a
complete picture instead of stopping at the first mismatch.

```sh
cargo build --release          # builds the `yaxcmp` binary
cargo run --release --bin yaxcmp -- --arch a64 --range 0x94000000..0x94100000
```

### what gets compared

by default (`--compare semantic`) the tool compares **decode results, not
rendered text**: the opcode and each operand's kind and value, taken from
yaxpeax-arm's `Instruction` and from capstone's operand detail (`CS_OPT_DETAIL`).
`--compare text` falls back to comparing disassembly strings with the
per-architecture equivalence rules.

semantic comparison normalizes representation differences that aren't decode
differences: shift-by-zero vs no shift, `lsl` with an extend style vs capstone's
extender (`uxtx`/`uxtw` as identity), capstone's post-index operand split,
`xzr`/`wzr`/`sp` naming, scalar simd element size from the register name vs `vas`,
lane selectors reported only on the last register of a group, and `mov`-family
immediates that are stored pre-inversion.

alias pairs (`mov`/`orr`, `bfm`/`bfxil`, `subs`/`cmp`, atomics `ldadd`/`stadd`,
...) are reported as `opcode_alias` and are informational: their operand lists
don't correspond one-for-one, so they aren't compared field by field.

anything the model doesn't cover (system registers, prefetch operations, SME
operands) is reported as `unmodeled` rather than as a mismatch — the tool says
"can't compare this" instead of inventing disagreements.

semantic comparison is implemented for a64. armv7 and thumb still compare text;
the tool says which mode it used in the header.

options:

```
--arch <a64|armv7|thumb>     architecture to compare (default: a64)
--range <start..end>         encoding range, inclusive, hex or decimal
--start <n> / --end <n>      range bounds separately (default: 0..0xffffffff)
--sample <n> [--seed <s>]    compare `n` randomly chosen encodings
--words <file|w1,w2,..>      compare specific encodings: a comma-separated list,
                             or a file with one encoding per line (`#` comments)
--threads <n>                worker threads (default: rayon's choice)
--chunk-size <n>             encodings per task (default: 65536)
--max-examples <n>           examples kept per mnemonic pair (default: 3)
--no-equivalence             only byte-identical text counts as a match
--fail-on-rejections         also exit nonzero when yaxpeax-arm can't decode
                             something capstone can
--json                       machine-readable report
--quiet                      no progress on stderr
```

a full 2^32 sweep of a64 takes a few minutes on a modern desktop. a quick
sampled look is `yaxcmp --arch a64 --sample 100000000`.

### what the categories mean

| category | meaning |
| --- | --- |
| `match` | identical disassembly text |
| `accepted_diff` | text differs, but it's a divergence the arch module has a rule for (aliasing, `$` vs `#` on branch targets, register naming, ..) |
| `mismatch` | both decoders produced text and they disagree with no explanation: the interesting failures |
| `yax_reject_known` | yaxpeax-arm did not decode, capstone did, and it's a known gap (SVE, MTE, coprocessor instructions, ..) |
| `yax_reject_unexpected` | yaxpeax-arm did not decode, capstone did, no explanation on file |
| `cs_reject` | yaxpeax-arm decoded, capstone did not |
| `both_reject` | neither decoded |
| `incomplete` | yaxpeax-arm needs more input |
| `opcode_mismatch` | both decoded, different opcodes |
| `operand_mismatch` | same opcode, different operand kind/value/count |
| `opcode_alias` | same instruction under a different alias (`mov` vs `orr`) |
| `unmodeled` | an operand representation the semantic layer doesn't model |
| `panic` | yaxpeax-arm panicked while decoding. always a bug; the panic message is recorded and the sweep continues |

`mismatch`, `opcode_mismatch`, `operand_mismatch` and `panic` are defects: they
make the tool exit nonzero.
unexpected rejections are reported but only fail with `--fail-on-rejections`,
since yaxpeax-arm is knowingly incomplete over large parts of the ISA.

### regression lists

`--words` is the cheap way to re-check specific encodings, such as a list built
from a previous run's mismatches:

```sh
yaxcmp --arch thumb --words mismatches.txt
```

(anything after `#` on a line is a comment, and a comma-separated list works
inline: `--words 0xd503201f,0x94000000`.)

## on batching capstone

capstone can decode a whole buffer in one `cs_disasm` call instead of one
`cs_disasm_iter` per encoding, but only if invalid encodings don't stop it —
which is what `CS_OPT_SKIPDATA` is for (it emits a synthetic `.byte` instruction
instead of stopping). a batched sweep would therefore be: one call per chunk,
one instruction record per decided encoding.

measured with `cargo run --release --example decode_bench`, it is not faster:

| range (1M encodings, one thread) | yax only | `cs_disasm_iter` | batch + manual resync | batch + skipdata |
| --- | --- | --- | --- | --- |
| a64 `0x00000000..0x000fffff` (94% invalid) | 28M/s | 10M/s | 2.0M/s | 1.8M/s |
| a64 `0x94000000..0x941fffff` (all valid) | 5.9M/s | 1.4M/s | 1.5M/s | 1.3M/s |
| thumb `0x0000..0xffff` | 2.9M/s | 1.2M/s | 0.5M/s | 1.1M/s |

the reason is that `cs_disasm` materializes a ~240 byte `cs_insn` per decoded
instruction (mnemonic and `op_str` are inline arrays), so batching writes
hundreds of megabytes per million encodings. `cs_disasm_iter` reuses a single
record. capstone's per-call overhead is small next to the actual decode.

without skipdata, `cs_disasm` returns only the prefix before the first invalid
encoding, and each re-entry allocates a fresh instruction array — which is why
manual resync is the slowest option of the four on invalid-dense input, and only
breaks even on fully-valid input.

batching also changes what a thumb sweep measures: a buffer of 65536 words
yields 124047 decoded instructions because 16-bit instructions put the next
instruction at `+2`, so the unit stops being "one encoding word". the per-word
loop in `src/lib.rs` deliberately keeps `cs_disasm_iter`.

`examples/decode_bench.rs` is kept so this can be re-measured after capstone
updates.

## findings

`REPORT.md` is the full report: what the sweeps found, what each decoder does for
every example, and what the Arm ARM (DDI 0487) says about it — separating
`UNDEFINED` rules (a decoder must reject) from `UNPREDICTABLE` ones (a decoder
may decode).

## tests

`cargo test --release` runs parser unit tests plus a small comparison slice per
architecture. a full sweep exists as an `#[ignore]`d test; the binary is the
better tool for that.

on aarch64 you may want `RUSTFLAGS="-C target-feature=+lse"` (if the target
supports armv8.1), which keeps the progress counter's atomics to a single
instruction rather than an outlined libcall.

## layout

- `src/capstone.rs`: per-worker capstone handle (`cs_disasm_iter`).
- `src/report.rs`: categories, counts, per-mnemonic buckets with bounded examples.
- `src/a64.rs`, `src/armv7.rs`, `src/armv7_thumb.rs`: decode glue, the
  disassembly-text parser, and the acceptable-divergence rules for each
  architecture. these were moved out of the test files so the tool and the tests
  share one implementation.
- `src/lib.rs`: the target trait, rayon sweep, and per-encoding classification.
- `src/bin/yaxcmp.rs`: command line front end.
- `tests/`: parser unit tests and small differential smoke tests.
