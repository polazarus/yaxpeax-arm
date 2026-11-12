//! differential tests against capstone.
//!
//! the sweep tool lives in `src/bin/yaxcmp.rs` and is the thing to reach for
//! when investigating a disagreement; these tests are the CI-sized slice of it,
//! plus the parser unit tests. the actual comparison logic, the text parser and
//! the acceptable-divergence rules are all in the `differential_tests` library
//! so the tool and these tests can never drift apart.

use differential_tests::report::Category;
use differential_tests::{run, ArchKind, Config};

use differential_tests::a64::{ParsedDisassembly, ParsedOperand};

#[test]
fn test_operand_parsing() {
    assert_eq!(ParsedOperand::parse("xzr", 64), (ParsedOperand::Register { size: 'x', num: 32 }, 3));
    assert_eq!(ParsedOperand::parse("wzr", 64), (ParsedOperand::Register { size: 'w', num: 32 }, 3));
    assert_eq!(ParsedOperand::parse("xsp", 64), (ParsedOperand::Register { size: 'x', num: 33 }, 3));
    assert_eq!(ParsedOperand::parse("wsp", 64), (ParsedOperand::Register { size: 'w', num: 33 }, 3));
    assert_eq!(ParsedOperand::parse("w1", 64), (ParsedOperand::Register { size: 'w', num: 1 }, 2));
    assert_eq!(ParsedOperand::parse("x1", 64), (ParsedOperand::Register { size: 'x', num: 1 }, 2));
}

#[test]
fn test_instruction_parsing() {
    let inst = ParsedDisassembly::parse("msub w17, w8, w15, w0");
    assert_eq!(inst, ParsedDisassembly {
        opcode: "msub".to_string(),
        operands: [
            Some(ParsedOperand::Register { size: 'w', num: 17 }),
            Some(ParsedOperand::Register { size: 'w', num: 8 }),
            Some(ParsedOperand::Register { size: 'w', num: 15 }),
            Some(ParsedOperand::Register { size: 'w', num: 0 }),
            None,
            None,
        ]
    });

    let inst = ParsedDisassembly::parse("stlurb w0, [x0, #0x1]");
    assert_eq!(inst, ParsedDisassembly {
        opcode: "stlurb".to_string(),
        operands: [
            Some(ParsedOperand::Register { size: 'w', num: 0 }),
            Some(ParsedOperand::MemoryWithOffset { base: "x0".to_string(), offset: Some(1), writeback: false }),
            None,
            None,
            None,
            None,
        ]
    });
    let inst2 = ParsedDisassembly::parse("stlurb w0, [x0, #1]");
    assert_eq!(inst, inst2);

    let inst = ParsedDisassembly::parse("mov wsp, #0x80000001");
    assert_eq!(inst.opcode, "mov");
    assert_eq!(inst.operands[0], Some(ParsedOperand::Register { size: 'w', num: 33 }));
    assert_eq!(inst.operands[1], Some(ParsedOperand::Immediate(-0x7fffffff)));
}

fn config() -> Config {
    Config {
        arch: ArchKind::A64,
        start: 0x54800000,
        end: 0x54800010,
        max_examples: 2,
        ..Config::default()
    }
}

/// a small slice of the encoding space that both yaxpeax-arm and capstone
/// should agree on. failing here means a real regression: `yaxcmp --arch a64
/// --range 0x54800000..0x54800010` has the details.
#[test]
fn capstone_differential_smoke_test() {
    let tally = run(&config(), None);

    // the tool reports rather than panics; this slice is supposed to be clean,
    // so any failure category is a regression.
    assert_eq!(
        tally.failures(),
        0,
        "{} failing encodings; run `yaxcmp --arch a64 --range 0x54800000..0x54800010` for details",
        tally.failures()
    );
    assert!(tally.total() > 0, "the range should not be empty");
}

/// the full encoding space. this takes a long time and will report known
/// mismatches, so it is `#[ignore]`d; use `yaxcmp` directly when you want to
/// look at what it finds.
#[test]
#[ignore]
fn capstone_differential_full_sweep() {
    let mut config = config();
    config.start = 0;
    config.end = u32::MAX as u64;

    let tally = run(&config, None);
    eprintln!(
        "{} encodings: {} match, {} accepted divergence, {} mismatch, {} panic",
        tally.total(),
        tally.count(Category::Match),
        tally.count(Category::AcceptedDiff),
        tally.count(Category::Mismatch),
        tally.count(Category::Panic),
    );
}
