//! A64 (armv8-a) comparisons: operand/disassembly parsing and the rules that
//! describe *acceptable* divergence between yaxpeax-arm and capstone.
//!
//! the parser and the match rules were originally inline in
//! `tests/capstone-differential.rs`; they live here so the `yaxcmp` tool and the
//! test can share one implementation.

use crate::capstone::ArchMode;
use crate::{RejectKind, Target, YaxOutcome};
use yaxpeax_arch::{Arch, Decoder};

use std::fmt::Write;
use std::num::ParseIntError;

#[derive(Debug)]
pub enum ParsedOperand {
    Register { size: char, num: u8 },
    Memory(String),
    MemoryWithOffset { base: String, offset: Option<i64>, writeback: bool },
    SIMDRegister { size: char, num: u8 },
//    SIMDRegisterElements { num: u8, elems: u8, elem_size: char },
//    SIMDRegisterElement { num: u8, elem_size: char, elem: u8 },
    SIMDElementLane { elem: String, lane_selector: u8 },
    Immediate(i64),
    PCRel(i64),
    Float(f64),
    Other(String),
    RegisterFamily(String),
}

impl PartialEq for ParsedOperand {
    fn eq(&self, other: &Self) -> bool {
        use ParsedOperand::*;

        match (self, other) {
            (Register { size: size_l, num: num_l }, Register { size: size_r, num: num_r }) => {
                size_l == size_r && num_l == num_r
            },
            (Memory(l), Memory(r)) => {
                l == r
            },
            (
                MemoryWithOffset { base: base_l, offset: offset_l, writeback: writeback_l },
                MemoryWithOffset { base: base_r, offset: offset_r, writeback: writeback_r },
            ) => {
                base_l == base_r &&
                offset_l == offset_r &&
                writeback_l == writeback_r
            },
            (Immediate(l), Immediate(r)) => {
                l == r
            },
            (PCRel(l), PCRel(r)) => {
                l == r
            },
            (Immediate(l), PCRel(r)) => {
                // assume pc=0 as capstone does by default
                *l == 0 + r
            },
            (PCRel(l), Immediate(r)) => {
                // assume pc=0 as capstone does by default
                0 + l == *r
            },
            (Float(l), Float(r)) => {
                l.to_ne_bytes() == r.to_ne_bytes()
            },
            (RegisterFamily(l), RegisterFamily(r)) => {
                l == r
            },
            (SIMDRegister { size: size_l, num: num_l }, SIMDRegister { size: size_r, num: num_r }) => {
                size_l == size_r && num_l == num_r
            },
            (SIMDElementLane { elem: elem_l, lane_selector: lane_l }, SIMDElementLane { elem: elem_r, lane_selector: lane_r }) => {
                elem_l == elem_r && lane_l == lane_r
            }
            (Other(l), Other(r)) => {
                // yax prints `asr #0` as just `asr`. is this actually a no-op?
                if (l == "asr" && r == "asr #0") || (l == "asr #0" && r == "asr") {
                    true
                } else if (l == "lsr" && r == "lsr #0") || (l == "lsr #0" && r == "lsr") {
                    true
                } else if (l == "ror" && r == "ror #0") || (l == "ror #0" && r == "ror") {
                    true
                } else {
                    l == r
                }
            }
            (_, _) => {
                false
            }
        }
    }
}

impl ParsedOperand {
    pub fn parse(s: &str, width: u8) -> (Self, usize) {
        let parse_hex_or_dec = |mut s: &str| {
            let mut negate = false;
            if s.as_bytes()[0] == b'-' {
                negate = true;
                s = &s[1..];
            }

            let v = if !s.starts_with("0x") {
                i64::from_str_radix(s, 10).expect("can parse string")
            } else {
                u64::from_str_radix(&s[2..], 16).expect("can parse string") as i64
            };
            if negate {
                -v
            } else {
                v
            }
        };

        if s.as_bytes()[0] == b'#' {
            let end = s.find(',').unwrap_or(s.len());
            let imm_str = &s[1..end];
            if imm_str.contains('.') {
                use std::str::FromStr;
                (ParsedOperand::Float(f64::from_str(imm_str).expect("can parse string")), end)
            } else {
                let imm = parse_hex_or_dec(imm_str);
                let imm = if width == 32 {
                    imm as i32 as i64
                } else {
                    imm
                };
                (ParsedOperand::Immediate(imm), end)
            }
        } else if s.as_bytes()[0] == b'$' {
            let end = s.find(',').unwrap_or(s.len());
            let imm_str = &s[1..end];
            let imm_str = if imm_str.starts_with("+") {
                &imm_str[1..]
            } else {
                imm_str
            };
            let imm = parse_hex_or_dec(imm_str);
            (ParsedOperand::PCRel(imm), end)
        } else if s.as_bytes()[0] == b'[' {
            let brace_end = s.find(']').map(|x| x + 1).unwrap_or(s.len());
            let mut end = brace_end;
            let mut writeback = false;
            if s.as_bytes().get(end) == Some(&b'!') {
                end += 1;
                writeback = true;
            }

            let addr = &s[1..brace_end - 1];

            let offset = addr.rfind(',').map(|comma| {
                addr[comma + 1..].trim()
            }).and_then(|mut offset_str| {
                if offset_str.as_bytes().get(0) == Some(&b'#') {
                    offset_str = &offset_str[1..];

                    Some(parse_hex_or_dec(offset_str))
                } else {
                    None
                }
            });

            let base_end = addr.rfind(',').unwrap_or(addr.len());
            let base = addr[..base_end].trim();

            if writeback || offset.is_some() {
                (ParsedOperand::MemoryWithOffset {
                    base: base.to_string(),
                    offset: offset,
                    writeback,
                }, end)
            } else {
                (ParsedOperand::Memory(base.to_string()), end)
            }
        } else if s.as_bytes()[0] == b'{' {
            let brace_end = s.find('}');
            if let Some(brace_end) = brace_end {
                if s.as_bytes().get(brace_end + 1) == Some(&b'[') {
                    if let Some(end) = s.find(']') {
                        let group = &s[0..brace_end];
                        let lane = &s[brace_end + 2..end];
                        let lane = parse_hex_or_dec(lane);

                        return (ParsedOperand::SIMDElementLane {
                            elem: group.to_string(),
                            lane_selector: lane as u8,
                        }, end);
                    }
                }

                let end = s.find(',').unwrap_or(s.len());
                (ParsedOperand::RegisterFamily(s[0..end].to_string()), end)
            } else {
                let end = s.find(',').unwrap_or(s.len());
                (ParsedOperand::Other(s[0..end].to_string()), end)
            }
        } else {
            let end = s.find(',').unwrap_or(s.len());
            let substr = &s[..end];
            match s.as_bytes()[0] as char {
                sz @ 'w' | sz @ 'x' => {
                    if &s[1..end] == "zr" {
                        return (ParsedOperand::Register { size: sz, num: 32 }, 3);
                    }
                    if &s[1..end] == "sp" {
                        return (ParsedOperand::Register { size: sz, num: 33 }, 3);
                    }
                    let num: Result<u8, ParseIntError> = s[1..end].parse();
                    match num {
                        Ok(num) => {
                            (ParsedOperand::Register { size: sz, num }, end)
                        }
                        Err(_) => {
                            (ParsedOperand::Other(s[..end].to_string()), end)
                        }
                    }
                }
                sz @ 'b' | sz @ 'h' | sz @ 's' | sz @ 'd' | sz @ 'q' => {
                    let num: Result<u8, ParseIntError> = s[1..end].parse();
                    match num {
                        Ok(num) => {
                            (ParsedOperand::SIMDRegister { size: sz, num }, end)
                        }
                        Err(_) => {
                            (ParsedOperand::Other(s[..end].to_string()), end)
                        }
                    }
                }
                'v' => {
                    match substr.find('[') {
                        Some(lane_selector_start) => {
                            let lane_selector_end = substr.find(']').unwrap();
                            let elem = substr[..lane_selector_start].to_string();
                            let lane_selector = parse_hex_or_dec(&substr[lane_selector_start + 1..lane_selector_end]) as u8;
                            (ParsedOperand::SIMDElementLane { elem, lane_selector }, end)
                        }
                        None => {
                            // some kind of simd element that does not include a trailing `[]`.
                            // treat it as an opaque string for now.
                            (ParsedOperand::Other(substr.to_string()), end)
                        }
                    }
                }
                _ => {
                    (ParsedOperand::Other(s[..end].to_string()), end)
                }
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ParsedDisassembly {
    pub opcode: String,
    // arm instructions do not have six operands, but due to parse ambiguity and the rather hackjob
    // parser here, pretend they might.
    pub operands: [Option<ParsedOperand>; 6]
}

impl ParsedDisassembly {
    pub fn parse(s: &str) -> Self {
        let mut operands = [None, None, None, None, None, None];
        if let Some((opcode, mut operands_text)) = s.split_once(' ') {
            let opcode = opcode.to_string();

            let mut i = 0;
            let mut width = 64;

            while operands_text.len() > 0 {
                if operands_text.as_bytes()[0] == b',' {
                    operands_text = &operands_text[1..];
                }
                operands_text = operands_text.trim();
                let (parsed, amount) = ParsedOperand::parse(&operands_text, width);
                operands[i] = Some(parsed);
                if let Some(ParsedOperand::Register { size: 'w', .. }) = &operands[i] {
                    width = 32;
                }
                operands_text = &operands_text[amount..];
                i += 1;
            }

            ParsedDisassembly {
                opcode,
                operands,
            }
        } else {
            ParsedDisassembly {
                opcode: s.to_string(),
                operands,
            }
        }
    }
}

pub fn acceptable_match(yax_text: &str, cs_text: &str) -> bool {
                        if yax_text == cs_text {
                            return true;
                        }

                        let parsed_yax = ParsedDisassembly::parse(yax_text);
                        let parsed_cs = ParsedDisassembly::parse(cs_text);

                        if parsed_yax == parsed_cs {
                            return true;
                        }

                        if parsed_yax.opcode == "sb" && parsed_cs.opcode == "msr" {
                            // capstone doesn't report the `sb` speculation barrier, for whatever
                            // reason?
                            return true;
                        }

                        if false {
                            eprintln!("yax: {} -> {:?}", yax_text, parsed_yax);
                            eprintln!("cs: {} -> {:?}", cs_text, parsed_cs);
                        }

                        if parsed_yax.opcode == parsed_cs.opcode && parsed_yax.opcode == "mrs" && parsed_yax.operands[0] == parsed_cs.operands[0] {
                            if let Some(ParsedOperand::Other(o)) = parsed_yax.operands[1].as_ref() {
                                if o.starts_with("s") {
                                    // capstone knows about more system registers than yaxpeax-arm at the
                                    // moment, so this is likely a case where the disagreement is on the
                                    // name of the system register.
                                    return true;
                                }
                            }
                        }

                        // capstone decodes [00, 40, 20, 0b] as "add w0, w0, w0, uxtw", note
                        // missing shift amount (should be lsl, too). yax decodes this as
                        // "add w0, w0, w0", no shift amount.
                        //
                        // same for x registers and uxtx, so handle that too.
                        if let Some(cs_text) = cs_text.strip_suffix(", uxtw") {
                            if yax_text == cs_text {
                                return true;
                            }
                        } else if let Some(cs_text) = cs_text.strip_suffix(", uxtx") {
                            if yax_text == cs_text {
                                return true;
                            }
                        }

                        // some instructions like `11400000` have an immeidate lsl #12 as their
                        // last operand. yax normalizes this to an unshifted `imm << 12`, capstone
                        // just prints lsl #12.
                        if cs_text.starts_with(yax_text) && cs_text.ends_with(", lsl #12") {
                            return true;
                        }

                        // capstone just shows empty string for unrecognized prf{,u}m immediates,
                        // leaving broken text
                        if cs_text.starts_with("prfum ") && yax_text.starts_with("prfum ") {
                            return true;
                        }
                        if cs_text.starts_with("prfm ") && yax_text.starts_with("prfm ") {
                            return true;
                        }

                        // don't totally understand aliasing rules for `ORR (immediate)` and mov..
                        if cs_text.starts_with("mov ") && yax_text.starts_with("orr ") ||
                            cs_text.starts_with("orr ") && yax_text.starts_with("mov ")
                        {
                            return true;
                        }

                        // yax notmalizes movn to mov
                        if cs_text.starts_with("movn ") && yax_text.starts_with("mov ") {
                            return true;
                        }

                        // yax notmalizes movz to mov
                        if cs_text.starts_with("movz ") && yax_text.starts_with("mov ") {
                            return true;
                        }

                        if parsed_yax.opcode == "mov" && parsed_cs.opcode == "dup" {
                            if parsed_yax.operands == parsed_cs.operands {
                                return true;
                            }
                        }
    //                    if cs_text.starts_with("dup") && yax_text.starts_with("mov ") && cs_text.replace("dup ", "mov ") == yax_text {
    //                        return true;
    //                    }
                        // capstone bug! e0030033 is `bfxil w0, wzr, #0, #1`, but capstone picks
                        // the bfc alias instead. skip these, generally.
                        if yax_text.starts_with("bfxil") && (cs_text.starts_with("bfc") || cs_text.starts_with("bfi")) {
                            return true;
                        }

                        // S being present or not has no bearing on the shift amount, #0 either
                        // way.
                        // yax will not print shift because of its ineffectual nature.
                        if (cs_text.starts_with("strb") || cs_text.starts_with("ldrb") || cs_text.starts_with("ldrsb") || cs_text.starts_with("ldr b") || cs_text.starts_with("str b")) && cs_text.contains(" lsl #0]") {
                            return true;
                        }

                        // yax uses lsl instead of uxtx when the reg size is uxtx. same for
                        // uxtw/w-regs
                        if cs_text.replace("uxtx", "lsl") == yax_text ||
                            cs_text.replace("uxtw", "lsl") == yax_text {
                            return true;
                        }

                        // yax shows dcps{1,2} operand, capstone does not?
                        if yax_text.starts_with("dcps") {
                            return true;
                        }

                        // yax does not handle aliases for msr instructions yet
                        if yax_text.starts_with("msr ") {
                            return true;
                        }

                        // some kinda bug to deal with hint value width
                        if cs_text.starts_with("hint ") {
                            return true;
                        }
                        if cs_text.starts_with("dsb ") {
                            return true;
                        }
                        if cs_text.starts_with("clrex ") {
                            return true;
                        }
                        if yax_text.starts_with("sys ") {
                            return true;
                        }
                        if cs_text.starts_with("yield ") {
                            return true;
                        }
                        if cs_text.starts_with("wfe ") {
                            return true;
                        }
                        if cs_text.starts_with("wfi ") {
                            return true;
                        }
                        if cs_text.starts_with("sev ") {
                            return true;
                        }
                        if yax_text.starts_with("hint ") {
                            return true;
                        }

                        // fmlal v0.2s, v0.4h, v0.h[0] != fmlal v0.2s, v0.2h, v0.h[0]. bytes: [0,
                        // 0, 80, f]
                        // .. i think capstone is wrong on this one
                        if yax_text.starts_with("fmlal ") || yax_text.starts_with("fmlsl ") {
                            return true;
                        }

                        if cs_text.starts_with("irg") && yax_text.starts_with(cs_text) && yax_text.ends_with(", xzr") {
                            // a trailing xzr is implicit in irg if omitted in the instruction
                            // text. yax includes xzr in this case, capstone does not.
                            return true;
                        }

                        const TAG_INSTRUCTIONS: [&'static str; 8] = [
                            "stzgm", "stg", "ldg", "stzg",
                            "stgm", "st2g", "ldgm", "stz2g"
                        ];

                        for mnemonic in TAG_INSTRUCTIONS {
                            if parsed_yax.opcode != mnemonic || parsed_cs.opcode != mnemonic {
                                continue;
                            }

                            for (yax_op, cs_op) in parsed_yax.operands.iter().zip(parsed_cs.operands.iter()) {
                                if yax_op == cs_op {
                                    continue;
                                }
                                match (yax_op, cs_op) {
                                    (
                                        Some(ParsedOperand::MemoryWithOffset {
                                            base: cs_base, offset: Some(cs_offset), writeback: cs_writeback
                                        }),
                                        Some(ParsedOperand::MemoryWithOffset {
                                            base: yax_base, offset: Some(yax_offset), writeback: yax_writeback
                                        })
                                    ) => {
                                        if cs_base != yax_base || cs_writeback != yax_writeback {
                                            // a pretty fundamental decode mismatch..
                                            return false;
                                        }

                                        if *cs_offset != *yax_offset && *yax_offset < 0 {
                                            // capstone decodes offsets as an unsigned integer, not
                                            // clamping to the signed range -4096 to 4080. yaxpeax
                                            // decodes the operand into this range.
                                            if (!*yax_offset + 1) & 0x1fff != *cs_offset {
                                                return false;
                                            }
                                        }
                                    }
                                    _ => {
                                        // some operands differ
                                        return false;
                                    }
                                }
                            }

                            // operand matching above looks OK, opcodes are the same, we're good
                            // here.
                            return true;
                        }

                        return false;
                    }

/// capstone decodes encodings that `yaxpeax-arm` does not implement yet (SVE,
/// SVE2, MTE, the `cpy` family, and friends), and decodes the UNDEFINED
/// encodings of C5.1.2 as `mrs`/`msr`. these are known gaps in yaxpeax-arm, not
/// disagreements about an instruction both decoders claim to know, so they are
/// counted separately from real mismatches.
pub fn known_gap(cs_text: &str) -> bool {
    cs_text.starts_with("mrs ") || cs_text.starts_with("msr ") ||
    cs_text.contains("dot ") ||
    cs_text.contains("fmlal ") ||
    cs_text.contains("bfmlalb ") ||
    cs_text.contains("bfmlalt ") ||
    cs_text.contains("fmlal2 ") ||
    cs_text.contains("fmlsl ") ||
    cs_text.contains("fmlsl2 ") ||
    cs_text.contains("addg ") ||
    cs_text.contains("subg ") ||
    cs_text.contains("cpye ") ||
    cs_text.contains("cpyet ") ||
    cs_text.contains("cpyewt ") ||
    cs_text.contains("cpyfprt ") ||
    cs_text.contains("cpyfpwt ") ||
    cs_text.contains("cpyp ") ||
    cs_text.contains("cpypt ") ||
    cs_text.contains("cpypwt ") ||
    cs_text.contains("cpyprtwn ") ||
    cs_text.contains("cpypwtwn ") ||
    cs_text.contains("cpypwn ") ||
    cs_text.contains("cpyprt ") ||
    cs_text.contains("cpyert ") ||
    cs_text.contains("cpyfert ") ||
    cs_text.contains("cpyfp ") ||
    cs_text.contains("cpyfpt ") ||
    cs_text.contains("cpyfe ") ||
    cs_text.contains("cpyfet ") ||
    cs_text.contains("cpyfewt ") ||
    cs_text.contains("wfet ") ||
    cs_text.contains("wfit ") ||
    cs_text.contains("ttest ") ||
    cs_text.contains("tstart ") ||
    cs_text.contains("tcommit ") ||
    cs_text.contains("tcancel ") ||
    cs_text.contains("setp ") ||
    cs_text.contains("setgp ") ||
    cs_text.contains("setpt ") ||
    cs_text.contains("setpn ") ||
    cs_text.contains("setge ") ||
    cs_text.contains("setget ") ||
    cs_text.contains("setgen ") ||
    cs_text.contains("setgetn ") ||
    cs_text.contains("setptn ") ||
    cs_text.contains("setm ") ||
    cs_text.contains("setmt ") ||
    cs_text.contains("setmn ") ||
    cs_text.contains("setmtn ") ||
    cs_text.contains("sete ") ||
    cs_text.contains("setet ") ||
    cs_text.contains("seten ") ||
    cs_text.contains("setetn ") ||
    cs_text.contains("setgm ") ||
    cs_text.contains("setgmn ") ||
    cs_text.contains("setgmt ") ||
    cs_text.contains("setgmtn ") ||
    cs_text.contains("setgpt ") ||
    cs_text.contains("setgpn ") ||
    cs_text.contains("setgptn ") ||
    cs_text.contains("bfcvtn2 ") ||
    cs_text.contains("bfcvtn ") ||
    cs_text.contains("bfcvt ") ||
    cs_text.contains("bfmmla ") ||
    cs_text.contains("frint32x ") ||
    cs_text.contains("frint64x ") ||
    cs_text.contains("ld64b ") ||
    cs_text.contains("st64b ") ||
    cs_text.contains("st64bv ") ||
    cs_text.contains("st64bv0 ") ||
    cs_text.contains("smmla ") ||
    cs_text.contains("ummla ") ||
    cs_text.contains("dsb ") ||
    cs_text.contains("dfb ") ||
    cs_text.starts_with("cpy")
}

pub static TARGET: A64 = A64;

pub struct A64;

impl Target for A64 {
    fn name(&self) -> &'static str {
        "a64"
    }

    fn mode(&self) -> ArchMode {
        ArchMode::A64
    }

    fn yax_decode(&self, bytes: &[u8], out: &mut String) -> YaxOutcome {
        let decoder = <yaxpeax_arm::armv8::a64::ARMv8 as Arch>::Decoder::default();
        match decoder.decode(&mut yaxpeax_arch::U8Reader::new(bytes)) {
            Ok(inst) => {
                write!(out, "{}", inst).expect("can format instruction");
                YaxOutcome::Decoded(crate::semantic::from_yax_a64(&inst))
            }
            Err(e) => YaxOutcome::Rejected(classify(e)),
        }
    }

    fn acceptable_match(&self, _word: u32, yax_text: &str, cs_text: &str) -> bool {
        // yaxpeax-arm and capstone disagree on the `dfb` encodings; a standing
        // TODO, not a new regression.
        acceptable_match(yax_text, cs_text) || cs_text.contains("dfb ")
    }

    fn expected_rejection(&self, _word: u32, _kind: RejectKind, cs_text: &str) -> bool {
        known_gap(cs_text)
    }

    fn supports_semantics(&self) -> bool {
        true
    }
}

fn classify(e: yaxpeax_arm::armv8::a64::DecodeError) -> RejectKind {
    use yaxpeax_arm::armv8::a64::DecodeError;
    match e {
        DecodeError::ExhaustedInput => RejectKind::Incomplete,
        DecodeError::IncompleteDecoder => RejectKind::Unimplemented,
        DecodeError::InvalidOpcode | DecodeError::InvalidOperand => RejectKind::Invalid,
    }
}
