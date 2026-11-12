//! armv7 thumb (t32) comparisons: operand/disassembly parsing and the rules
//! that describe acceptable divergence between yaxpeax-arm and capstone.
//!
//! the parser and match rules were moved here from
//! `tests/differential-v7-thumb.rs` so the `yaxcmp` tool and the test share one
//! implementation.

use crate::capstone::ArchMode;
use crate::{RejectKind, Target, YaxOutcome};
use yaxpeax_arch::{Arch, Decoder};

use std::fmt::Write;
use std::num::ParseIntError;

#[derive(Debug, PartialEq, Eq)]
pub enum MemOffset {
    Imm(i64),
    Shift { regnum: u8, rest: String },
    Reg { regnum: u8 },
}

#[derive(Debug)]
pub enum ParsedOperand {
    Register { size: char, num: u8, neg: bool },
    Memory(String),
    MemoryWithOffset { basereg: u8, offset: MemOffset, writeback: bool },
    SIMDRegister { size: char, num: u8 },
//    SIMDRegisterElements { num: u8, elems: u8, elem_size: char },
//    SIMDRegisterElement { num: u8, elem_size: char, elem: u8 },
    SIMDElementLane { elem: String, lane_selector: u8 },
    // a shift/rotate by an immediate, like `lsl #4`
    ShiftImm { kind: String, imm: u8 },
    // a shift/rotate by a register, like `lsl r4`
    ShiftReg { kind: String, regnum: u8 },
    Immediate(i64),
    PCRel(i64),
    Float(f64),
    CoprocOption(u64),
    Other(String),
    RegisterFamily(String),
}

impl PartialEq for ParsedOperand {
    fn eq(&self, other: &Self) -> bool {
        use ParsedOperand::*;

        match (self, other) {
            (Register { size: size_l, num: num_l, neg: neg_l }, Register { size: size_r, num: num_r, neg: neg_r }) => {
                size_l == size_r && num_l == num_r && neg_l == neg_r
            },
            (Memory(l), Memory(r)) => {
                if l == "r10" && r == "sl" {
                    true
                } else {
                    l == r
                }
            },
            (
                MemoryWithOffset { basereg: base_l, offset: offset_l, writeback: writeback_l },
                MemoryWithOffset { basereg: base_r, offset: offset_r, writeback: writeback_r },
            ) => {
                base_l == base_r ||
                    offset_l == offset_r &&
                    writeback_l == writeback_r
            },
            // smooth over yax printing `[rN]` rather than `[rN, #0]` like capstone.
            (Memory(l), MemoryWithOffset { basereg, offset: MemOffset::Imm(0), writeback: false }) => {
                if let Some(lreg) = ParsedOperand::parse_reg(l) {
                    lreg == *basereg
                } else {
                    false
                }
            },
            // and make equality reflexive.
            (MemoryWithOffset { basereg, offset: MemOffset::Imm(0), writeback: false }, Memory(r)) => {
                if let Some(rreg) = ParsedOperand::parse_reg(r) {
                    rreg == *basereg
                } else {
                    false
                }
            },
            (Immediate(l), Immediate(r)) => {
                l == r || (*l == *r as u32 as i64) || (
                    // capstone prints 17f1004c as adds.w ip, r7, -0x8000000 which is .. wrong?
                    *l == 0x8000000 && *r == -0x80000000
                )
            },
            (PCRel(l), PCRel(r)) => {
                l == r
            },
            // TODO: don't actually know if this is thumb, 32-bit thumb, arm, .. so try a few
            // things.
            (Immediate(l), PCRel(r)) => { *l == 2 + r || *l == 4 + r },
            (PCRel(l), Immediate(r)) => {
                if (*l >= 0) == (*r >= 0x8000000) {
                    2 + l == *r || 4 + l == *r
                } else {
                    let delta = l.wrapping_sub(*r);
                    delta.wrapping_add(2) as u32 == 0 ||
                        delta.wrapping_add(4) as u32 == 0
                }
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
            (ShiftImm { kind: lkind, imm: limm }, ShiftImm { kind: rkind, imm: rimm }) => {
                lkind == rkind && limm == rimm
            },
            (ShiftReg { kind: lkind, regnum: lregnum }, ShiftReg { kind: rkind, regnum: rregnum }) => {
                lkind == rkind && lregnum == rregnum
            },
            (CoprocOption(l), CoprocOption(r)) => {
                l == r
            },
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
        let end = s.find(',').unwrap_or(s.len());
        if let Some(imm) = ParsedOperand::try_parse_imm(&s[..end]) {
            (ParsedOperand::Immediate(imm), end)
        } else if s.as_bytes()[0] == b'#' {
            let mut imm_str = &s[1..end];
            // TODO: improve the following hack, useful to parse `[reg], -1!`
            if imm_str.ends_with('!') {
                imm_str = &s[1..end - 1];
            }
            if imm_str.contains('.') {
                use std::str::FromStr;
                (ParsedOperand::Float(f64::from_str(imm_str).expect("can parse string")), end)
            } else {
                let imm = ParsedOperand::parse_hex_or_dec(imm_str);
                let imm = if width == 32 {
                    imm as i32 as i64
                } else {
                    imm
                };
                (ParsedOperand::Immediate(imm), end)
            }
        } else if s.as_bytes()[0] == b'$' {
            let imm_str = &s[1..end];
            let imm_str = if imm_str.starts_with("+") {
                &imm_str[1..]
            } else {
                imm_str
            };
            let imm = ParsedOperand::parse_hex_or_dec(imm_str);
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

            let offset = addr.find(',').map(|comma| {
                addr[comma + 1..].trim()
            }).map(|offset_str| {
                if let Some((reg, shift)) = ParsedOperand::parse_shift(offset_str) {
                    MemOffset::Shift { regnum: reg, rest: shift.to_string() }
                } else if let Some(reg) = ParsedOperand::parse_reg(offset_str) {
                    MemOffset::Reg { regnum: reg }
                } else {
                    MemOffset::Imm(ParsedOperand::parse_imm(offset_str))
                }
            });

            let base_end = addr.find(',').unwrap_or(addr.len());
            let base = addr[..base_end].trim();

            if let Some(offset) = offset {
                (ParsedOperand::MemoryWithOffset {
                    basereg: ParsedOperand::parse_reg(base).expect("base is reg"),
                    offset: offset,
                    writeback,
                }, end)
            } else if writeback {
                (ParsedOperand::MemoryWithOffset {
                    basereg: ParsedOperand::parse_reg(base).expect("base is reg"),
                    offset: MemOffset::Imm(0),
                    writeback,
                }, end)
            } else {
                (ParsedOperand::Memory(base.to_string()), end)
            }
        } else if s.as_bytes()[0] == b'{' {
            let brace_end = s.find('}');
            if let Some(brace_end) = brace_end {
                if let Some(imm) = ParsedOperand::try_parse_imm(&s[1..brace_end]) {
                    return (ParsedOperand::CoprocOption(imm as u64), end);
                }
                if s.as_bytes().get(brace_end + 1) == Some(&b'[') {
                    if let Some(end) = s.find(']') {
                        let group = &s[0..brace_end];
                        let lane = &s[brace_end + 2..end];
                        let lane = ParsedOperand::parse_hex_or_dec(lane);

                        return (ParsedOperand::SIMDElementLane {
                            elem: group.to_string(),
                            lane_selector: lane as u8,
                        }, end);
                    }
                }

                let end = s[brace_end..].find(',').unwrap_or(s.len() - brace_end) + brace_end;
                let regs = s[0..end].to_string();
                // TODO: parse register numbers more reasonably...
                (ParsedOperand::RegisterFamily(regs.replace("sl", "r10")), end)
            } else {
                (ParsedOperand::Other(s[0..end].to_string()), end)
            }
        } else {
            let mut start = 0;
            let mut substr = &s[..end];
            let mut neg = false;
            if substr.as_bytes()[0] == b'-' {
                start += 1;
                neg = true;
                substr = &substr[1..];
            }

            // may be a boring register, as in `r11`, or a named form like `sl`, `lr`, `pc`:
            if let Ok(Some(num)) = Self::try_parse_reg(substr) {
                return (ParsedOperand::Register { size: 'r', num, neg }, end);
            }

            // could be a shift description, like `ror #4` or `lsl r4`:
            if let Ok(shift) = Self::try_parse_shift(substr) {
                return (shift, end);
            }

            // or it may be a writeback memory operand (like `ldm r10!, {reglist}`)
            if substr.as_bytes()[end - 1] == b'!' {
                if let Ok(Some(num)) = Self::try_parse_reg(&substr[..end - 1]) {
                    return (ParsedOperand::MemoryWithOffset { basereg: num, offset: MemOffset::Imm(0), writeback: true }, end);
                }
            }
            match s.as_bytes()[start] as char {
                sz @ 'r' => {
                    if &s[start + 1..end] == "zr" {
                        return (ParsedOperand::Register { size: sz, num: 32, neg }, end);
                    }
                    if &s[start + 1..end] == "sp" {
                        return (ParsedOperand::Register { size: sz, num: 33, neg }, end);
                    }
                    let num: Result<u8, ParseIntError> = s[start + 1..end].parse();
                    match num {
                        Ok(num) => {
                            (ParsedOperand::Register { size: sz, num, neg }, end)
                        }
                        Err(_) => {
                            (ParsedOperand::Other(s[start..end].to_string()), end)
                        }
                    }
                }
                sz @ 'b' | sz @ 'h' | sz @ 's' | sz @ 'd' | sz @ 'q' => {
                    let num: Result<u8, ParseIntError> = s[start + 1..end].parse();
                    match num {
                        Ok(num) => {
                            (ParsedOperand::SIMDRegister { size: sz, num }, end)
                        }
                        Err(_) => {
                            (ParsedOperand::Other(s[start..end].to_string()), end)
                        }
                    }
                }
                'v' => {
                    match substr.find('[') {
                        Some(lane_selector_start) => {
                            let lane_selector_end = substr.find(']').unwrap();
                            let elem = substr[..lane_selector_start].to_string();
                            let lane_selector = ParsedOperand::parse_hex_or_dec(&substr[lane_selector_start + 1..lane_selector_end]) as u8;
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
                    (ParsedOperand::Other(s[start..end].to_string()), end)
                }
            }
        }
    }

    fn parse_hex_or_dec(s: &str) -> i64 {
        Self::try_parse_hex_or_dec(s).unwrap_or_else(|| panic!("{}", "failed to parse {s}"))
    }

    fn try_parse_hex_or_dec(mut s: &str) -> Option<i64> {
        let mut negate = false;
        if s.as_bytes()[0] == b'-' {
            negate = true;
            s = &s[1..];
        }

        let v = if !s.starts_with("0x") {
            i64::from_str_radix(s, 10).ok()?
        } else {
            u64::from_str_radix(&s[2..], 16).ok()? as i64
        };
        if negate {
            Some(-v)
        } else {
            Some(v)
        }
    }

    fn parse_imm(s: &str) -> i64 {
        Self::try_parse_imm(s).unwrap_or_else(|| {
            panic!("{}", "failed to parse imm: {s}")
        })
    }

    fn try_parse_imm(s: &str) -> Option<i64> {
        if s.starts_with("#") {
            ParsedOperand::try_parse_hex_or_dec(&s[1..])
        } else {
            ParsedOperand::try_parse_hex_or_dec(s)
        }
    }

    fn try_parse_shift(s: &str) -> Result<ParsedOperand, &'static str> {
        if let Some((lhs, rhs)) = s.split_once(' ') {
            if lhs == "lsl" || lhs == "lsr" || lhs == "asr" || lhs == "ror" {
                if let Some(imm) = Self::try_parse_imm(rhs) {
                    let imm = imm as u8;
                    Ok(ParsedOperand::ShiftImm { kind: lhs.to_string(), imm })
                } else if let Some(num) = Self::try_parse_reg(rhs)? {
                    Ok(ParsedOperand::ShiftReg { kind: lhs.to_string(), regnum: num })
                } else {
                    Err("unrecognized shift amount")
                }
            } else {
                Err("not a recognized shift type")
            }
        } else {
            if s == "rrx" {
                Ok(ParsedOperand::ShiftImm { kind: "rrx".to_string(), imm: 1 })
            } else {
                Err("unrecognized parameterless shift")
            }
        }
    }

    fn try_parse_reg(s: &str) -> Result<Option<u8>, &'static str> {
        if s.starts_with("r") {
            let num = s[1..].parse()
                .map_err(|_| "can parse regnum")?;
            Ok(Some(num))
        } else {
            match s {
                "sb" => Ok(Some(9)),
                "sl" => Ok(Some(10)),
                "fp" => Ok(Some(11)),
                "ip" => Ok(Some(12)),
                "sp" => Ok(Some(13)),
                "lr" => Ok(Some(14)),
                "pc" => Ok(Some(15)),
                _ => {
                    Ok(None)
                }
            }
        }
    }

    fn parse_reg(s: &str) -> Option<u8> {
        Self::try_parse_reg(s).unwrap()
    }

    fn parse_shift(s: &str) -> Option<(u8, &str)> {
        if let Some(comma) = s.find(",") {
            let reg = s[..comma].trim();
            let rest = s[comma + 1..].trim();
            assert!(
                rest.starts_with("lsl") ||
                   rest.starts_with("lsr") ||
                   rest.starts_with("asr") ||
                   rest.starts_with("ror"));
            let regnum = ParsedOperand::parse_reg(reg).unwrap_or_else(|| {
                panic!("shift base should be reg, was not in: {}", s);
            });
            Some((regnum, rest))
        } else {
            None
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

    fn operand_count(&self) -> u8 {
        let mut i = 0;

        for op in self.operands.iter() {
            if op.is_none() {
                break;
            }
            i += 1;
        }

        i
    }
}


pub fn acceptable_match(word: u32, yax_text: &str, cs_text: &str) -> bool {
                        if yax_text == cs_text {
                            return true;
                        }

                        // TODO: capstone prints `blx #0x...`, yax prints `blx.w $+0x...`
                        if yax_text.starts_with("blx.w ") && cs_text.starts_with("blx ") {
                            return true;
                        }

                        // TODO: capstone prints `bl #0x...`, yax prints `bl.w $+0x...`
                        if yax_text.starts_with("bl.w ") && cs_text.starts_with("bl ") {
                            return true;
                        }

                        if yax_text == "udf #0xfe" && cs_text == "trap " {
                            // TODO:
                            return true;
                        }

                        let parsed_yax = ParsedDisassembly::parse(yax_text);
                        let parsed_cs = ParsedDisassembly::parse(cs_text);

                        if parsed_yax == parsed_cs {
                            return true;
                        }

                        // yax shows the alias out of the box, capstone undoes it:
                        // > pop.w {sb} != ldm.w sp!, {sb}. bytes: [bd, e8, 0, 2]
                        if parsed_yax.opcode == "pop.w" && parsed_cs.opcode == "ldm.w" && parsed_yax.operands[0] == parsed_cs.operands[1] {
                            return true;
                        }

                        // yax shows the alias out of the box, capstone undoes it:
                        // > push {sb} != stmdb sp!, {sb}. bytes: [2d, e9, 0, 2]
                        if parsed_yax.opcode == "push" && parsed_cs.opcode == "stmdb" && parsed_yax.operands[0] == parsed_cs.operands[1] {
                            return true;
                        }

                        // more aliasing defaults..
                        // > push.w {r1} != str r1, [sp, #-0x4]!. bytes: [4d, f8, 4, 1d]
                        if parsed_yax.opcode == "push.w" && parsed_cs.opcode == "str" {
                            return true;
                        }

                        if (parsed_yax.opcode == "add" &&
                            parsed_cs.opcode == "add") ||
                            (parsed_yax.opcode == "adds" &&
                             parsed_cs.opcode == "adds") {
                            // capstone prints the T2 encoding of `ADD (register, Thumb)` as if
                            // it is the T1 encoding with three registers.
                            if parsed_yax.operand_count() == 2 && parsed_cs.operand_count() == 3 {
                                if parsed_yax.operands[0] == parsed_cs.operands[0] &&
                                    parsed_yax.operands[1] == parsed_cs.operands[1] &&
                                    parsed_cs.operands[0] == parsed_cs.operands[2] {
                                    return true;
                                }
                            }

                            // capstone prints the T2 encoding of `ADD (SP plus immediate)` with
                            // two operands instead of the three from the manual.
                            if word & 0xff80 == 0xb000 &&
                                parsed_yax.opcode == "add" &&
                                parsed_cs.opcode == "add" &&
                                parsed_yax.operands[0] == parsed_cs.operands[0] &&
                                parsed_yax.operands[0] == parsed_yax.operands[1] &&
                                parsed_cs.operands[1] == parsed_yax.operands[2] {
                                return true;
                            }
                        }

                        if parsed_yax.opcode == "sub" &&
                            parsed_cs.opcode == "sub" {
                            // capstone prints the T1 encoding of `SUB (SP minus immediate)` with
                            // two operands instead of the three from the manual.
                            if word & 0xff80 == 0xb080 &&
                                parsed_yax.opcode == "sub" &&
                                parsed_cs.opcode == "sub" &&
                                parsed_yax.operands[0] == parsed_cs.operands[0] &&
                                parsed_yax.operands[0] == parsed_yax.operands[1] &&
                                parsed_cs.operands[1] == parsed_yax.operands[2] {
                                return true;
                            }
                        }

                        // TODO: yaxpeax-arm doesn't know about armv8-m yet, which gets `bxns` to
                        // replace `bx` in some encodings.
                        if parsed_yax.opcode == "bx" && parsed_cs.opcode == "bxns" {
                            if parsed_yax.operands == parsed_cs.operands {
                                return true;
                            }
                        }

                        // TODO: same for blx/blxns.
                        if parsed_yax.opcode == "blx" && parsed_cs.opcode == "blxns" {
                            if parsed_yax.operands == parsed_cs.operands {
                                return true;
                            }
                        }

                        if parsed_yax.operands == parsed_cs.operands {
                            // TODO: yax probably should simply write `stm` in this case like the
                            // manual implies and capstone does.
                            if parsed_yax.opcode == "stmia" && parsed_cs.opcode == "stm" {
                                    return true;
                            }

                            // TODO: yax says .w when it doesn't need to, allegedly?
                            // TODO: also omits a w when capstone adds one? (mov vs movw: "mov sp, #0x1183 != movw sp, #0x1183. bytes: [41, f2, 83, 1d]")
                            if parsed_yax.opcode == format!("{}{}", parsed_cs.opcode, ".w") || parsed_yax.opcode.clone() + "w" == parsed_cs.opcode {
                                return true;
                            }

                            if parsed_yax.opcode.replace(".w", "w") == parsed_cs.opcode {
                                // TODO: yax prints wide sub-immediate/add-immediate as `sub.w`, capstone
                                // says `subw` (same for add). this probably could use fixing.
                                // comparisons:
                                // > sub.w r4, r4, #0x483 != subw r4, r4, #0x483. bytes: [a4, f2, 83, 44]
                                // > add.w r10, r1, #0x985 != addw sl, r1, #0x985. bytes: [1, f6, 85, 1a]
                                // > pld.w pc, [fp, #0xf04] != pldw [fp, #0xf04]. bytes: [bb, f8, 4, ff]
                                return true;
                            }

                            if parsed_yax.opcode == "pli.w" && parsed_cs.opcode == "pli" {
                                return true;
                            }

                            // TODO: so many signed multiply-related mishaps (usually around the
                            // M/R bits.
                            if parsed_yax.opcode.starts_with("sm") && parsed_cs.opcode.starts_with("sm") {
                                return true;
                            }
                        }

                        // TODO: a weirder mishap with smmls{r}
                        // > smmls.w pc, r3, lr != smmlsr pc, r3, lr, pc. bytes: [63, fb, 1e, ff]
                        if parsed_yax.opcode.starts_with("smmls.w") && parsed_cs.opcode.starts_with("smmlsr") {
                            return true;
                        }

                        if parsed_yax.opcode == parsed_cs.opcode {
                            let mut last_operand = 0;
                            for op in parsed_yax.operands.iter() {
                                if op.is_none() {
                                    break;
                                }

                                last_operand += 1;
                            }
                            if parsed_yax.operands[..last_operand] == parsed_cs.operands[..last_operand] &&
                                parsed_cs.operands[last_operand] == Some(ParsedOperand::Immediate(0)) {
                                return true;
                            }
                        }

                        /*
                        if parsed_yax.opcode.replace(".w", "") == parsed_cs.opcode.replace(".w", "") {
                            // TODO: yax prints garbage like `b.wgt` instead of `bgt.w`. yikes.
                            return true;
                        }
                        */

                        static BRANCHES: &'static [&'static str] = &[
                            "bgt", "bhi", "b", "ble", "bge", "blt", "bge",
                            "bhs", "blo", "beq", "bne", "bpl", "bmi", "bvc",
                            "bvs", "bls", "bfi", "b.w","blx.w",
                        ];
                        if BRANCHES.contains(&parsed_yax.opcode.as_str()) && parsed_yax.opcode == parsed_cs.opcode {
                            // TODO: the harness doesn't relativeizie branch targets?
                            return true;
                        }

                        if false {
                            eprintln!("parsed yax: {:?}", parsed_yax);
                            eprintln!("parsed cs: {:?}", parsed_cs);
                            eprintln!("yax: {} -> {:?}", yax_text, parsed_yax);
                            eprintln!("cs: {} -> {:?}", cs_text, parsed_cs);
                        }

                        false
                    }

pub static TARGET: Thumb = Thumb;

pub struct Thumb;

impl Target for Thumb {
    fn name(&self) -> &'static str {
        "thumb"
    }

    fn mode(&self) -> ArchMode {
        ArchMode::Thumb
    }

    fn yax_decode(&self, bytes: &[u8], out: &mut String) -> YaxOutcome {
        let decoder = <yaxpeax_arm::armv7::ARMv7 as Arch>::Decoder::default()
            .with_thumb_mode(true)
            .allow_nonconforming(true);
        match decoder.decode(&mut yaxpeax_arch::U8Reader::new(bytes)) {
            Ok(inst) => {
                write!(out, "{}", inst).expect("can format instruction");
                YaxOutcome::Decoded(None)
            }
            Err(e) => YaxOutcome::Rejected(classify(e)),
        }
    }

    fn acceptable_match(&self, word: u32, yax_text: &str, cs_text: &str) -> bool {
        acceptable_match(word, yax_text, cs_text) || known_gap(cs_text)
    }

    fn expected_rejection(&self, word: u32, kind: RejectKind, cs_text: &str) -> bool {
        // capstone accepts IT with firstcond=1111, though the encoding is
        // UNPREDICTABLE.
        if word >> 16 & 0xf0ff == 0xf0bf
            && cs_text.starts_with("it")
            && kind == RejectKind::Nonconforming
        {
            return true;
        }

        // yaxpeax-arm reports the undefined encodings capstone calls `udf` as an
        // error.
        if cs_text.starts_with("udf") && kind == RejectKind::Undefined {
            return true;
        }

        // TODO: some better way of verifying unpredictable encodings.
        if kind == RejectKind::Unpredictable {
            return true;
        }

        known_gap(cs_text)
    }
}

fn classify(e: yaxpeax_arm::armv7::DecodeError) -> RejectKind {
    use yaxpeax_arm::armv7::DecodeError;
    match e {
        DecodeError::ExhaustedInput => RejectKind::Incomplete,
        DecodeError::Undefined => RejectKind::Undefined,
        DecodeError::Unpredictable => RejectKind::Unpredictable,
        DecodeError::Nonconforming => RejectKind::Nonconforming,
        DecodeError::Incomplete => RejectKind::Unimplemented,
        DecodeError::InvalidOpcode | DecodeError::InvalidOperand => RejectKind::Invalid,
    }
}

/// instructions yaxpeax-arm does not decode in thumb mode yet.
fn known_gap(cs_text: &str) -> bool {
    cs_text.starts_with("stc")
        || cs_text.starts_with("stlex")
        || cs_text.starts_with("ldrex")
        || cs_text.starts_with("usada8")
        || cs_text.starts_with("usad8")
}
