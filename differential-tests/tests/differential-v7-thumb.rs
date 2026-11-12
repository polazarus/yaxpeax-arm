//! differential tests against capstone.
//!
//! the sweep tool lives in `src/bin/yaxcmp.rs` and is the thing to reach for
//! when investigating a disagreement; these tests are the CI-sized slice of it,
//! plus the parser unit tests. the actual comparison logic, the text parser and
//! the acceptable-divergence rules are all in the `differential_tests` library
//! so the tool and these tests can never drift apart.

use differential_tests::report::Category;
use differential_tests::{run, ArchKind, Config};

use differential_tests::armv7_thumb::{MemOffset, ParsedDisassembly, ParsedOperand};

#[test]
fn test_operand_parsing() {
    assert_eq!(ParsedOperand::parse("r3", 64), (ParsedOperand::Register { size: 'r', num: 3, neg: false }, 2));
    assert_eq!(ParsedOperand::parse("r11", 64), (ParsedOperand::Register { size: 'r', num: 11, neg: false }, 3));
    assert_eq!(ParsedOperand::parse("-r11", 64), (ParsedOperand::Register { size: 'r', num: 11, neg: true }, 4));
    assert_eq!(ParsedOperand::parse("sl", 32), (ParsedOperand::Register { size: 'r', num: 10, neg: false }, 2));
    assert_eq!(ParsedOperand::parse("-sl", 32), (ParsedOperand::Register { size: 'r', num: 10, neg: true }, 3));

    assert_eq!(
        ParsedOperand::parse("[r0, sl, lsl #3]", 32),
        (ParsedOperand::MemoryWithOffset { basereg: 0, offset: MemOffset::Shift { regnum: 10, rest: "lsl #3".to_string() }, writeback: false }, 16)
    );
    assert_eq!(
        ParsedOperand::parse("[r0, r10, lsl #3]", 32),
        (ParsedOperand::MemoryWithOffset { basereg: 0, offset: MemOffset::Shift { regnum: 10, rest: "lsl #3".to_string() }, writeback: false }, 17)
    );

    assert_eq!(
        ParsedDisassembly::parse("ldm.w r10!, {r0, r1, r7, r8, fp, lr}"),
        ParsedDisassembly::parse("ldm.w sl!, {r0, r1, r7, r8, fp, lr}"),
    );

    // 5d, ec 80 70
    assert_eq!(
        ParsedDisassembly::parse("mrrc p0, 8, r7, sp, c0"),
        ParsedDisassembly::parse("mrrc p0, #8, r7, sp, c0"),
    );

    // 45 f6 80 a5
    assert_eq!(
        ParsedDisassembly::parse("bls.w $-0xba500"),
        ParsedDisassembly::parse("bls.w #0xfff45b04"),
    );

    // c2 ea 81 6e
    assert_eq!(
        ParsedDisassembly::parse("pkhbt lr, r2, r1, lsl #26"),
        ParsedDisassembly::parse("pkhbt lr, r2, r1, lsl #0x1a"),
    );

    // d6 fc 00 c2
    assert_eq!(
        ParsedDisassembly::parse("ldc2l p2, c12, [r6], {0x0}"),
        ParsedDisassembly::parse("ldc2l p2, c12, [r6], {0}"),
    );

    // e2 f7 00 81
    assert_eq!(
        ParsedDisassembly::parse("hvc.w #0x2100"),
        ParsedDisassembly::parse("hvc.w #0x2100"),
    );

    // 1a ee ff f4
    assert_eq!(
        ParsedDisassembly::parse("mrc p4, 0, apsr_nzcv, c10, c15, 7"),
        ParsedDisassembly::parse("mrc p4, #0, apsr_nzcv, c10, c15, #7"),
    );
}

#[test]
fn test_instruction_parsing() {
    /*
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
    */
}

fn config() -> Config {
    Config {
        arch: ArchKind::Armv7Thumb,
        start: 0x0,
        end: 0x20,
        max_examples: 2,
        ..Config::default()
    }
}

/// a small slice of the encoding space that both yaxpeax-arm and capstone
/// should agree on. failing here means a real regression: `yaxcmp --arch thumb
/// --range 0x0..0x20` has the details.
#[test]
fn capstone_differential_smoke_test() {
    let tally = run(&config(), None);

    // the tool reports rather than panics; this slice is supposed to be clean,
    // so any failure category is a regression.
    assert_eq!(
        tally.failures(),
        0,
        "{} failing encodings; run `yaxcmp --arch thumb --range 0x00000000..0x00000020` for details",
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
