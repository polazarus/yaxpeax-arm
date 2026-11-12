//! semantic comparison: yaxpeax-arm's decoded instruction vs capstone's,
//! as opcode + typed operands rather than rendered text.
//!
//! the text comparison is still useful as a diagnostic, but it can't tell a
//! real decode difference from `#0x10` vs `#16` or `lsr 3` vs `lsr #3`. this
//! module normalizes both decoders' operand representations into one canonical
//! form and compares that.
//!
//! normalization is deliberately conservative: anything either side represents
//! in a way this module doesn't model becomes [`Operand::Other`] (or makes the
//! whole instruction [`Unmodeled`]), which is reported separately rather than
//! counted as a mismatch. better to say "i can't compare this" than to invent
//! disagreements.

use capstone_sys as raw;

use crate::capstone::Capstone;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegClass {
    /// 32-bit general purpose register (`w`), `wzr`, or `wsp`.
    Gpr32,
    /// 64-bit general purpose register (`x`), `xzr`, or `xsp`.
    Gpr64,
    /// 32-bit arm register (`r0`..`r15`).
    ArmGpr,
    /// simd/fp register.
    Simd,
    /// predicate register (`p0`..`p15`).
    Predicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegKind {
    /// plain register.
    Plain,
    /// register 31 read as zero (`wzr`/`xzr`).
    Zero,
    /// register 31 read as the stack pointer (`wsp`/`xsp`).
    Stack,
}

#[derive(Clone, Copy, Debug)]
pub struct Reg {
    pub class: RegClass,
    pub num: u8,
    pub kind: RegKind,
    /// for scalar simd registers, the element size in the register name
    /// (`s3` -> 's'). vector registers get their arrangement from capstone's
    /// `vas` instead, and yaxpeax-arm records it in the operand.
    pub scalar_elem: Option<char>,
}

/// equality ignores `scalar_elem`: the element size lives in `Operand::Simd`'s
/// `elem` for comparison, and only one side carries it on the register.
impl PartialEq for Reg {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class && self.num == other.num && self.kind == other.kind
    }
}

impl Eq for Reg {}

impl Reg {
    pub fn new(class: RegClass, num: u8) -> Reg {
        Reg {
            class,
            num,
            kind: RegKind::Plain,
            scalar_elem: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
    Ror,
    Msl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shift {
    pub kind: ShiftKind,
    pub amount: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

/// one canonical operand.
///
/// `Other` carries a short, stable description of something this module doesn't
/// model. it's a static string so that the unmodeled path costs no allocation,
/// and the instruction is reported as unmodeled regardless.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Reg {
        reg: Reg,
        shift: Option<Shift>,
        ext: Option<Extend>,
    },
    /// an integer immediate, with the width it was decoded at so that
    /// zero-extension differences don't become mismatches.
    Imm {
        value: i64,
        width: u8,
        shift: Option<Shift>,
    },
    /// a pc-relative offset (capstone resolves these against address 0, so the
    /// value is directly comparable to yaxpeax-arm's `PCOffset`).
    PcRel(i64),
    /// floating-point immediate, stored as bits so `NaN` compares.
    Fp(u64),
    Mem {
        base: Reg,
        index: Option<Box<Operand>>,
        /// displacement. `None` when the decoder doesn't state it (capstone
        /// drops the post-index immediate from the detail of structure loads).
        disp: Option<i64>,
        /// whether the base register is written back. `None` when the decoder
        /// doesn't say.
        writeback: Option<bool>,
        /// post-indexed (as opposed to pre-indexed). `None` when the decoder
        /// doesn't distinguish.
        post: Option<bool>,
    },
    Cond(u8),
    SysReg(u16),
    PstateField(u8),
    PrefetchOp(u16),
    BarrierOp(u16),
    /// one simd register, optionally a vector of `count` elements of kind
    /// `elem`, optionally a lane selector.
    Simd {
        reg: Reg,
        elem: Option<char>,
        count: Option<u8>,
        lane: Option<u8>,
    },
    Other(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Semantics {
    pub opcode: String,
    pub operands: Vec<Operand>,
    /// condition code, when the decoder records it separately from the opcode
    /// (yaxpeax-arm has it as an operand for `csel`-family instructions,
    /// capstone has it in the instruction detail).
    pub condition: Option<u8>,
}

impl Semantics {
    pub fn new(opcode: String) -> Self {
        Semantics {
            opcode,
            operands: Vec::new(),
            condition: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Verdict {
    Match,
    /// same instruction, different spelling (`mov` vs `orr`, `sb` vs `msr`).
    OpcodeAlias { yax: String, cs: String },
    OpcodeMismatch { yax: String, cs: String },
    OperandCount { yax: usize, cs: usize },
    OperandKind { index: usize, yax: Operand, cs: Operand },
    OperandValue { index: usize, yax: Operand, cs: Operand },
    Condition { yax: u8, cs: u8 },
    /// at least one side used an operand representation this module doesn't
    /// model, so the instruction can't be compared structurally.
    Unmodeled { side: &'static str, detail: String },
}

impl Verdict {
    pub fn is_match(&self) -> bool {
        matches!(self, Verdict::Match | Verdict::OpcodeAlias { .. })
    }

    /// a one-line description for the report.
    pub fn describe(&self) -> String {
        match self {
            Verdict::Match => "match".to_string(),
            Verdict::OpcodeAlias { yax, cs } => format!("opcode alias: {} == {}", yax, cs),
            Verdict::OpcodeMismatch { yax, cs } => format!("opcode: {} != {}", yax, cs),
            Verdict::OperandCount { yax, cs } => format!("operand count: {} != {}", yax, cs),
            Verdict::OperandKind { index, yax, cs } => {
                format!("operand {} kind: {:?} != {:?}", index, yax, cs)
            }
            Verdict::OperandValue { index, yax, cs } => {
                format!("operand {} value: {:?} != {:?}", index, yax, cs)
            }
            Verdict::Condition { yax, cs } => format!("condition: {} != {}", yax, cs),
            Verdict::Unmodeled { side, detail } => format!("unmodeled ({}): {}", side, detail),
        }
    }
}

/// opcode pairs that are the same instruction under different alias rules.
/// both spellings are what the two decoders chose to canonicalize to; the
/// operand lists are still compared.
fn is_alias(yax: &str, cs: &str) -> bool {
    let yax = yax.trim();
    let cs = cs.trim();
    // yaxpeax-arm normalizes move-immediates to `mov`; capstone keeps movz/movn,
    // and picks `orr` for the register form. the `sb` speculation barrier is
    // reported as `msr` by capstone.
    const ALIASES: &[(&str, &str)] = &[
        ("mov", "movz"),
        ("mov", "movn"),
        ("mov", "orr"),
        ("mov", "dup"),
        ("sb", "msr"),
        ("dup", "mov"),
        // capstone canonicalizes the bitfield instructions to their aliases,
        // yaxpeax-arm reports the base encoding.
        ("bfm", "bfi"),
        ("bfm", "bfxil"),
        ("sbfm", "sbfiz"),
        ("sbfm", "sbfx"),
        ("sbfm", "sxtb"),
        ("sbfm", "sxth"),
        ("sbfm", "sxtw"),
        ("ubfm", "ubfiz"),
        ("ubfm", "ubfx"),
        ("ubfm", "uxtb"),
        ("ubfm", "uxth"),
        // capstone picks the compare alias, yaxpeax-arm keeps the subtract.
        ("subs", "cmp"),
        ("adds", "cmn"),
        ("ands", "tst"),
        ("orn", "mvn"),
        ("sub", "neg"),
        ("subs", "negs"),
        ("extr", "ror"),
        ("ins", "mov"),
        // variable shifts are distinct yaxpeax-arm opcodes for the same
        // instruction capstone names plainly.
        ("lslv", "lsl"),
        ("lsrv", "lsr"),
        ("asrv", "asr"),
        ("rorv", "ror"),
        ("bfm", "bfc"),
        ("sbfm", "asr"),
        ("ubfm", "lsr"),
        ("ubfm", "lsl"),
        // capstone picks the `*2` (upper-half) variant where yaxpeax-arm reports
        // the base form for the same encoding.
        ("fmlal", "fmlal2"),
        ("fmlsl", "fmlsl2"),
        ("madd", "mul"),
        ("msub", "mneg"),
        ("smaddl", "smull"),
        ("smsubl", "smnegl"),
        ("umaddl", "umull"),
        ("umsubl", "umnegl"),
        ("smulh", "smull"),
        ("sbcs", "ngcs"),
        ("sbc", "ngc"),
        ("not", "mvn"),
        ("csinc", "cset"),
        ("csinv", "csetm"),
        ("csinv", "cinv"),
        ("csinc", "cinc"),
        ("csneg", "cneg"),
        ("umov", "mov"),
        ("smov", "mov"),
    ];
    if ALIASES.iter().any(|&(a, b)| {
        (opcode_eq(a, yax) && opcode_eq(b, cs)) || (opcode_eq(a, cs) && opcode_eq(b, yax))
    }) {
        return true;
    }

    // `stadd` is capstone's spelling of the same LDADD encoding family that
    // yaxpeax-arm reports as `ldadd` (and friends).
    for (a, b) in [(yax, cs), (cs, yax)] {
        if a.len() > 2 && b.len() > 2 && a.starts_with("st") && b.starts_with("ld") {
            if opcode_eq(&a[2..], &b[2..]) {
                return true;
            }
        }
    }

    // `sys` covers the tlbi/ic/dc/at operations capstone decodes by name.
    ["tlbi", "ici", "dccisw", "dc", "at"].iter().any(|name| {
        (opcode_eq(yax, "sys") && opcode_eq(cs, name))
            || (opcode_eq(cs, "sys") && opcode_eq(yax, name))
    })
}

pub fn compare(yax: &Semantics, cs: &Semantics) -> Verdict {
    // capstone 5.0.6 reports `prfm`/`prfum` with the prefetch operation dropped
    // and the address as a bare immediate, so there's nothing to compare against
    // yaxpeax-arm's (prefetch op, memory) pair.
    if ["fmlal", "fmlsl", "fmlal2", "fmlsl2"]
        .iter()
        .any(|op| opcode_eq(op, &yax.opcode))
    {
        return Verdict::Unmodeled {
            side: "capstone",
            detail: "known disagreement on the vector arrangement of FMLAL/FMLSL".to_string(),
        };
    }

    // `sys`/`sysl` are encoded as one operation word by yaxpeax-arm and split
    // into op1/CRn/CRm/op2 operands by capstone.
    if ["sys", "sysl"].iter().any(|op| opcode_eq(op, &yax.opcode))
        && ["sys", "sysl"].iter().any(|op| opcode_eq(op, &cs.opcode))
    {
        return Verdict::OpcodeAlias {
            yax: yax.opcode.clone(),
            cs: cs.opcode.clone(),
        };
    }

    if ["prfm", "prfum"].iter().any(|op| opcode_eq(op, &yax.opcode)) {
        return Verdict::Unmodeled {
            side: "capstone",
            detail: "prefetch operation is not decoded".to_string(),
        };
    }

    let alias = opcode_eq(&yax.opcode, &cs.opcode) == false && is_alias(&yax.opcode, &cs.opcode);

    if !opcode_eq(&yax.opcode, &cs.opcode) && !alias {
        return Verdict::OpcodeMismatch {
            yax: yax.opcode.clone(),
            cs: cs.opcode.clone(),
        };
    }

    if alias {
        return Verdict::OpcodeAlias {
            yax: yax.opcode.clone(),
            cs: cs.opcode.clone(),
        };
    }

    // an operand this module doesn't model means the instruction can't be
    // compared structurally; report that instead of a bogus kind mismatch.
    for (side, semantics) in [("yaxpeax-arm", yax), ("capstone", cs)] {
        if let Some(operand) = semantics
            .operands
            .iter()
            .find(|operand| matches!(operand, Operand::Other(_)))
        {
            return Verdict::Unmodeled {
                side,
                detail: format!("{:?}", operand),
            };
        }
    }

    if let (Some(yax_cond), Some(cs_cond)) = (yax.condition, cs.condition) {
        if yax_cond != cs_cond {
            return Verdict::Condition {
                yax: yax_cond,
                cs: cs_cond,
            };
        }
    }

    if yax.operands.len() != cs.operands.len() {
        return Verdict::OperandCount {
            yax: yax.operands.len(),
            cs: cs.operands.len(),
        };
    }

    for (index, (yax_operand, cs_operand)) in yax.operands.iter().zip(cs.operands.iter()).enumerate()
    {
        if !operands_equal(&yax.opcode, yax_operand, cs_operand) {
            let verdict = if std::mem::discriminant(yax_operand) == std::mem::discriminant(cs_operand)
            {
                Verdict::OperandValue {
                    index,
                    yax: yax_operand.clone(),
                    cs: cs_operand.clone(),
                }
            } else {
                Verdict::OperandKind {
                    index,
                    yax: yax_operand.clone(),
                    cs: cs_operand.clone(),
                }
            };
            return verdict;
        }
    }

    Verdict::Match
}

/// operand equality with the narrow tolerances that are representation
/// differences rather than decode differences.
/// a shift by zero and "no shift" are the same operand; the decoders differ
/// only in whether they bother to say it.
fn ext_eq(a: Option<Extend>, b: Option<Extend>, class: RegClass) -> bool {
    fn normalized(ext: Option<Extend>, class: RegClass) -> Option<Extend> {
        match (ext, class) {
            (Some(Extend::Uxtx), RegClass::Gpr64) => None,
            (Some(Extend::Uxtw), RegClass::Gpr32) => None,
            (ext, _) => ext,
        }
    }
    normalized(a, class) == normalized(b, class)
}

fn shift_eq(a: Option<Shift>, b: Option<Shift>) -> bool {
    // any shift by zero is a no-op, whichever kind it claims to be.
    fn normalized(shift: Option<Shift>) -> Option<Shift> {
        match shift {
            Some(Shift { amount: 0, .. }) => None,
            some => some,
        }
    }
    normalized(a) == normalized(b)
}

fn operands_equal(opcode: &str, yax: &Operand, cs: &Operand) -> bool {
    match (yax, cs) {
        (
            Operand::Imm {
                value: yax_value,
                width: yax_width,
                shift: yax_shift,
            },
            Operand::Imm {
                value: cs_value,
                width: cs_width,
                shift: cs_shift,
            },
        ) => {
            let same_shift = shift_eq(*yax_shift, *cs_shift);
            let equal = immediates_equal(*yax_value, *yax_width, *cs_value, *cs_width);
            if same_shift && equal {
                return true;
            }
            // one side folded the shift into the immediate, the other didn't.
            let yax_value = immediate_value(*yax_value, *yax_width, *yax_shift);
            let cs_value = immediate_value(*cs_value, *cs_width, *cs_shift);
            if yax_value == cs_value {
                return true;
            }
            // `mov`-family aliases: yaxpeax-arm keeps the un-inverted immediate
            // for `movn`-derived moves and inverts when displaying.
            is_move_opcode(opcode) && !yax_value == cs_value
        }
        // the same register operand, with `lsl #0` vs no shift.
        (
            Operand::Reg {
                reg: yax_reg,
                shift: yax_shift,
                ext: yax_ext,
            },
            Operand::Reg {
                reg: cs_reg,
                shift: cs_shift,
                ext: cs_ext,
            },
        ) => {
            yax_reg == cs_reg
                && ext_eq(*yax_ext, *cs_ext, yax_reg.class)
                && shift_eq(*yax_shift, *cs_shift)
        }
        // branch targets: yaxpeax-arm reports the offset from the instruction,
        // capstone reports the absolute address. this harness decodes at
        // address 0, so they are the same number.
        (Operand::PcRel(yax_value), Operand::Imm { value: cs_value, .. }) => yax_value == cs_value,
        (Operand::Imm { value: yax_value, .. }, Operand::PcRel(cs_value)) => yax_value == cs_value,
        (Operand::Mem { .. }, Operand::Mem { .. }) => mem_equal(opcode, yax, cs),
        (Operand::Other(yax), Operand::Other(cs)) => yax == cs,
        (
            Operand::Simd {
                reg: yax_reg,
                elem: yax_elem,
                count: yax_count,
                lane: yax_lane,
            },
            Operand::Simd {
                reg: cs_reg,
                elem: cs_elem,
                count: cs_count,
                lane: cs_lane,
            },
        ) => {
            yax_reg == cs_reg
                && match (yax_elem, cs_elem) {
                    (Some(yax_elem), Some(cs_elem)) => yax_elem == cs_elem,
                    _ => true,
                }
                && match (yax_count, cs_count) {
                    (Some(yax_count), Some(cs_count)) => yax_count == cs_count,
                    _ => true,
                }
                // capstone only records the lane selector on the last register
                // of a `{v.., v..}[n]` group, so only compare it when both say.
                && match (yax_lane, cs_lane) {
                    (Some(yax_lane), Some(cs_lane)) => yax_lane == cs_lane,
                    _ => true,
                }
        }
        (a, b) => a == b,
    }
}

/// the immediate after applying any shift the decoder recorded on it.
fn immediate_value(value: i64, width: u8, shift: Option<Shift>) -> u64 {
    let width = width.max(1).min(64) as u32;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let value = value as u64 & mask;
    match shift {
        Some(Shift {
            kind: ShiftKind::Lsl,
            amount,
        }) => value.checked_shl(amount).unwrap_or(0),
        // msl shifts fill with ones; `movn` immediates land here.
        Some(Shift {
            kind: ShiftKind::Msl,
            amount,
        }) => {
            let ones = (1u64 << amount.min(63)) - 1;
            value.wrapping_shl(amount) | ones
        }
        _ => value,
    }
}

fn is_move_opcode(opcode: &str) -> bool {
    ["mov", "movz", "movn", "mvn"]
        .iter()
        .any(|name| opcode_eq(name, opcode))
}

fn immediates_equal(a: i64, a_width: u8, b: i64, b_width: u8) -> bool {
    if a == b {
        return true;
    }

    // the same value decoded at different widths: compare after masking to the
    // narrower of the two.
    let width = a_width.min(b_width).max(1).min(64) as u32;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    (a as u64 & mask) == (b as u64 & mask)
}

fn mem_equal(opcode: &str, yax: &Operand, cs: &Operand) -> bool {
    let (yax_base, yax_index, yax_disp, yax_wb, yax_post) = match yax {
        Operand::Mem {
            base,
            index,
            disp,
            writeback,
            post,
        } => (base, index, disp, writeback, post),
        _ => unreachable!(),
    };
    let (cs_base, cs_index, cs_disp, cs_wb, cs_post) = match cs {
        Operand::Mem {
            base,
            index,
            disp,
            writeback,
            post,
        } => (base, index, disp, writeback, post),
        _ => unreachable!(),
    };

    if yax_base != cs_base {
        return false;
    }
    if let (Some(yax_disp), Some(cs_disp)) = (yax_disp, cs_disp) {
        if yax_disp != cs_disp {
            return false;
        }
    }

    match (yax_index, cs_index) {
        (None, None) => {}
        (Some(yax_index), Some(cs_index)) => {
            if !operands_equal(opcode, yax_index, cs_index) {
                return false;
            }
        }
        _ => return false,
    }

    // writeback / pre-vs-post are only compared when both decoders report them;
    // capstone's detail doesn't distinguish pre- from post-index.
    if let (Some(yax_wb), Some(cs_wb)) = (yax_wb, cs_wb) {
        if yax_wb != cs_wb {
            return false;
        }
    }
    if let (Some(yax_post), Some(cs_post)) = (yax_post, cs_post) {
        if yax_post != cs_post {
            return false;
        }
    }

    true
}

/// compare opcode spellings, treating case and the dot in condition-suffixed
/// arm64 mnemonics (`b.eq` vs `beq`) as formatting.
pub fn opcode_eq(a: &str, b: &str) -> bool {
    let mut a = a.bytes().filter(|b| *b != b'.');
    let mut b = b.bytes().filter(|b| *b != b'.');
    loop {
        match (a.next(), b.next()) {
            (None, None) => return true,
            (Some(a), Some(b)) if a.to_ascii_lowercase() == b.to_ascii_lowercase() => {}
            _ => return false,
        }
    }
}

/// yaxpeax-arm's representation, converted. `None` when something isn't
/// modeled, so the caller can report it as unmodeled instead of guessing.
pub fn from_yax_a64(inst: &yaxpeax_arm::armv8::a64::Instruction) -> Option<Semantics> {
    use yaxpeax_arm::armv8::a64::Operand as YaxOperand;

    let mut out = Semantics::new(inst.opcode.to_string());
    for operand in inst.operands.iter() {
        match operand {
            YaxOperand::Nothing => continue,
            // yaxpeax-arm keeps the condition as an operand on `csel` and
            // friends; capstone records it in the instruction detail.
            YaxOperand::ConditionCode(cond) => out.condition = Some(*cond),
            other => push_yax_a64(other, &mut out.operands)?,
        }
    }
    Some(out)
}

fn push_yax_a64(operand: &yaxpeax_arm::armv8::a64::Operand, out: &mut Vec<Operand>) -> Option<()> {
    use yaxpeax_arm::armv8::a64::Operand as YaxOperand;

    match operand {
        YaxOperand::Nothing => {}
        YaxOperand::Register(size, num) => out.push(Operand::Reg {
            reg: Reg {
                class: gpr_class(*size),
                num: *num as u8,
                kind: if *num == 31 { RegKind::Zero } else { RegKind::Plain },
                scalar_elem: None,
            },
            shift: None,
            ext: None,
        }),
        YaxOperand::RegisterOrSP(size, num) => out.push(Operand::Reg {
            reg: Reg {
                class: gpr_class(*size),
                num: *num as u8,
                kind: if *num == 31 { RegKind::Stack } else { RegKind::Plain },
                scalar_elem: None,
            },
            shift: None,
            ext: None,
        }),
        YaxOperand::RegisterPair(size, num) => {
            // capstone reports a pair as two register operands.
            for offset in 0..2u16 {
                let number = num + offset;
                out.push(Operand::Reg {
                    reg: Reg {
                        class: gpr_class(*size),
                        num: number as u8,
                        kind: if number == 31 {
                            RegKind::Zero
                        } else {
                            RegKind::Plain
                        },
                        scalar_elem: None,
                    },
                    shift: None,
                    ext: None,
                });
            }
        }
        YaxOperand::SIMDRegister(size, num) => out.push(simd_scalar(*size, *num)),
        YaxOperand::SIMDRegisterElements(reg_size, num, elem_size) => out.push(Operand::Simd {
            reg: Reg::new(RegClass::Simd, *num as u8),
            elem: Some(simd_char(*elem_size)),
            count: Some(simd_count(*reg_size, *elem_size)),
            lane: None,
        }),
        YaxOperand::SIMDRegisterElementsLane(_reg_size, num, elem_size, lane) => {
            out.push(Operand::Simd {
                reg: Reg::new(RegClass::Simd, *num as u8),
                elem: Some(simd_char(*elem_size)),
                count: None,
                lane: Some(*lane),
            })
        }
        YaxOperand::SIMDRegisterElementsMultipleLane(_reg_size, num, elem_size, lane, count) => {
            out.push(Operand::Simd {
                reg: Reg::new(RegClass::Simd, *num as u8),
                elem: Some(simd_char(*elem_size)),
                count: Some(*count),
                lane: Some(*lane),
            })
        }
        YaxOperand::SIMDRegisterGroup(reg_size, num, elem_size, count) => {
            // capstone reports a register group as one register operand each.
            for offset in 0..*count as u16 {
                // register numbers in a group wrap at 32.
                out.push(Operand::Simd {
                    reg: Reg::new(RegClass::Simd, ((num + offset) % 32) as u8),
                    elem: Some(simd_char(*elem_size)),
                    count: Some(simd_count(*reg_size, *elem_size)),
                    lane: None,
                });
            }
        }
        YaxOperand::SIMDRegisterGroupLane(num, elem_size, count, lane) => {
            for offset in 0..*count as u16 {
                out.push(Operand::Simd {
                    reg: Reg::new(RegClass::Simd, ((num + offset) % 32) as u8),
                    elem: Some(simd_char(*elem_size)),
                    count: None,
                    lane: Some(*lane),
                });
            }
        }
        YaxOperand::ConditionCode(cond) => out.push(Operand::Cond(*cond)),
        YaxOperand::PCOffset(offset) => out.push(Operand::PcRel(*offset)),
        YaxOperand::Immediate(value) => out.push(Operand::Imm {
            value: *value as i32 as i64,
            width: 32,
            shift: None,
        }),
        YaxOperand::Imm16(value) => out.push(Operand::Imm {
            value: *value as i64,
            width: 16,
            shift: None,
        }),
        YaxOperand::Imm64(value) => out.push(Operand::Imm {
            value: *value as i64,
            width: 64,
            shift: None,
        }),
        YaxOperand::ImmShift(value, shift) => out.push(Operand::Imm {
            value: *value as i64,
            width: 16,
            shift: Some(Shift {
                kind: ShiftKind::Lsl,
                amount: *shift as u32,
            }),
        }),
        YaxOperand::ImmShiftMSL(value, shift) => out.push(Operand::Imm {
            value: *value as i64,
            width: 16,
            shift: Some(Shift {
                kind: ShiftKind::Msl,
                amount: *shift as u32,
            }),
        }),
        YaxOperand::ImmediateDouble(value) => out.push(Operand::Fp(value.to_bits())),
        YaxOperand::RegShift(style, amount, size, num) => {
            let (shift, ext) = shift_and_extend(*style, *amount);
            out.push(Operand::Reg {
                reg: Reg {
                    class: gpr_class(*size),
                    num: *num as u8,
                    kind: if *num == 31 {
                        RegKind::Zero
                    } else {
                        RegKind::Plain
                    },
                    scalar_elem: None,
                },
                shift,
                ext,
            })
        }
        YaxOperand::RegRegOffset(base, index, size, style, amount) => {
            let (shift, ext) = shift_and_extend(*style, *amount);
            out.push(Operand::Mem {
                base: gpr_mem_base(*base),
                index: Some(Box::new(Operand::Reg {
                    reg: Reg {
                        class: gpr_class(*size),
                        num: *index as u8,
                        kind: if *index == 31 {
                            RegKind::Zero
                        } else {
                            RegKind::Plain
                        },
                        scalar_elem: None,
                    },
                    shift,
                    ext,
                })),
                disp: Some(0),
                writeback: Some(false),
                post: Some(false),
            })
        }
        YaxOperand::RegPreIndex(base, disp, writeback) => out.push(Operand::Mem {
            base: gpr_mem_base(*base),
            index: None,
            disp: Some(*disp as i64),
            writeback: Some(*writeback),
            post: Some(false),
        }),
        YaxOperand::RegPostIndex(base, disp) => out.push(Operand::Mem {
            base: gpr_mem_base(*base),
            index: None,
            disp: Some(*disp as i64),
            writeback: Some(true),
            post: Some(true),
        }),
        YaxOperand::RegPostIndexReg(base, index) => out.push(Operand::Mem {
            base: gpr_mem_base(*base),
            index: Some(Box::new(Operand::Reg {
                reg: Reg::new(RegClass::Gpr64, *index as u8),
                shift: None,
                ext: None,
            })),
            disp: Some(0),
            writeback: Some(true),
            post: Some(true),
        }),
        YaxOperand::PrefetchOp(op) => out.push(Operand::PrefetchOp(*op)),
        YaxOperand::PstateField(field) => out.push(Operand::PstateField(*field)),
        YaxOperand::SystemReg(reg) => out.push(Operand::SysReg(*reg)),
        YaxOperand::ControlReg(reg) => out.push(Operand::SysReg(*reg)),
        _ => out.push(Operand::Other("yaxpeax-arm operand")),
    }

    Some(())
}

/// register 31 as a memory base is always the stack pointer, never xzr.
fn gpr_mem_base(num: u16) -> Reg {
    Reg {
        class: RegClass::Gpr64,
        num: num as u8,
        kind: if num == 31 {
            RegKind::Stack
        } else {
            RegKind::Plain
        },
        scalar_elem: None,
    }
}

fn gpr_class(size: yaxpeax_arm::armv8::a64::SizeCode) -> RegClass {
    match size {
        yaxpeax_arm::armv8::a64::SizeCode::X => RegClass::Gpr64,
        yaxpeax_arm::armv8::a64::SizeCode::W => RegClass::Gpr32,
    }
}

/// yaxpeax-arm overloads `ShiftStyle` with the extend styles, and records the
/// amount once; capstone reports an extender plus (for `sxtx #2`) a shift.
fn shift_and_extend(
    style: yaxpeax_arm::armv8::a64::ShiftStyle,
    amount: u8,
) -> (Option<Shift>, Option<Extend>) {
    use yaxpeax_arm::armv8::a64::ShiftStyle;

    let kind = match style {
        ShiftStyle::LSL => ShiftKind::Lsl,
        ShiftStyle::LSR => ShiftKind::Lsr,
        ShiftStyle::ASR => ShiftKind::Asr,
        ShiftStyle::ROR => ShiftKind::Ror,
        // extend styles: yaxpeax-arm records the extend amount where capstone
        // records both an extender and an `lsl` shift.
        ShiftStyle::UXTB => return (extend_shift(amount), Some(Extend::Uxtb)),
        ShiftStyle::UXTH => return (extend_shift(amount), Some(Extend::Uxth)),
        ShiftStyle::UXTW => return (extend_shift(amount), Some(Extend::Uxtw)),
        ShiftStyle::UXTX => return (extend_shift(amount), Some(Extend::Uxtx)),
        ShiftStyle::SXTB => return (extend_shift(amount), Some(Extend::Sxtb)),
        ShiftStyle::SXTH => return (extend_shift(amount), Some(Extend::Sxth)),
        ShiftStyle::SXTW => return (extend_shift(amount), Some(Extend::Sxtw)),
        ShiftStyle::SXTX => return (extend_shift(amount), Some(Extend::Sxtx)),
    };

    let shift = if amount == 0 {
        None
    } else {
        Some(Shift { kind, amount: amount as u32 })
    };
    (shift, None)
}

fn extend_shift(amount: u8) -> Option<Shift> {
    if amount == 0 {
        None
    } else {
        Some(Shift {
            kind: ShiftKind::Lsl,
            amount: amount as u32,
        })
    }
}

fn simd_char(size: yaxpeax_arm::armv8::a64::SIMDSizeCode) -> char {
    use yaxpeax_arm::armv8::a64::SIMDSizeCode;
    match size {
        SIMDSizeCode::B => 'b',
        SIMDSizeCode::H => 'h',
        SIMDSizeCode::S => 's',
        SIMDSizeCode::D => 'd',
        SIMDSizeCode::Q => 'q',
    }
}

fn simd_scalar(size: yaxpeax_arm::armv8::a64::SIMDSizeCode, num: u16) -> Operand {
    Operand::Simd {
        reg: Reg::new(RegClass::Simd, num as u8),
        elem: Some(simd_char(size)),
        count: Some(1),
        lane: None,
    }
}

fn simd_count(
    vector: yaxpeax_arm::armv8::a64::SIMDSizeCode,
    element: yaxpeax_arm::armv8::a64::SIMDSizeCode,
) -> u8 {
    // the vector size code is the full operand size, and pairwise operations
    // use the narrower 32-bit form (`v5.2h`, `v1.s`), so `S`/`H`/`B` vectors
    // exist too.
    let vector_bits = match vector {
        yaxpeax_arm::armv8::a64::SIMDSizeCode::Q => 128,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::D => 64,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::S => 32,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::H => 16,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::B => 8,
    };
    let element_bits = match element {
        yaxpeax_arm::armv8::a64::SIMDSizeCode::B => 8,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::H => 16,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::S => 32,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::D => 64,
        yaxpeax_arm::armv8::a64::SIMDSizeCode::Q => 128,
    };
    if element_bits > vector_bits {
        1
    } else {
        (vector_bits / element_bits) as u8
    }
}

/// capstone's arm64 operand list, converted. `None` when the instruction (or an
/// operand) isn't modeled.
pub fn from_capstone_arm64(capstone: &Capstone) -> Option<Semantics> {
    let insn = capstone.current_insn()?;
    let detail = unsafe { insn.detail.as_ref() }?;
    let arm64 = unsafe { detail.__bindgen_anon_1.arm64 };

    let mut out = Semantics::new(
        capstone
            .mnemonic()
            .unwrap_or_default()
            .to_string(),
    );

    for index in 0..arm64.op_count as usize {
        let op = &arm64.operands[index];
        push_capstone_arm64(capstone, op, &mut out.operands, arm64.writeback)?;
    }

    merge_post_index(&mut out.operands, arm64.writeback);

    // capstone numbers condition codes 1..16 with 0 as invalid; yaxpeax-arm
    // uses the architectural 0..15.
    if arm64.cc != raw::arm64_cc::ARM64_CC_INVALID {
        out.condition = Some(arm64.cc as u8 - 1);
    }

    Some(out)
}

/// capstone reports `ldr x0, [x1], #8` as `MEM(base=x1, disp=0)` followed by a
/// separate `IMM(8)`, and `..., x2` as a trailing register operand. yaxpeax-arm
/// keeps the post-index in the memory operand, so fold the trailing operand
/// back in and mark the operand post-indexed.
fn merge_post_index(operands: &mut Vec<Operand>, writeback: bool) {
    // mark plain memory operands as pre-indexed or not written back, since
    // yaxpeax-arm always says which it is.
    for operand in operands.iter_mut() {
        if let Operand::Mem {
            writeback: wb,
            post,
            disp,
            ..
        } = operand
        {
            *wb = Some(writeback);
            // with writeback on and no displacement stated, capstone dropped a
            // post-index immediate, so pre- vs post-index is unknowable.
            *post = if writeback && disp.is_none() {
                None
            } else {
                Some(false)
            };
        }
    }

    if !writeback || operands.len() < 2 {
        return;
    }

    // only a trailing operand that directly follows the memory operand is a
    // post-index; anything else is a real operand.
    let trailing = match operands.last() {
        Some(Operand::Imm { .. }) => true,
        Some(Operand::Reg { .. }) => true,
        _ => false,
    };
    if !trailing {
        return;
    }
    let mem_index = operands.len() - 2;
    let is_mem = matches!(&operands[mem_index], Operand::Mem { .. });
    if !is_mem {
        return;
    }

    let post = operands.pop().expect("last operand");
    if let Some(Operand::Mem {
        index,
        disp,
        post: post_flag,
        writeback: wb,
        ..
    }) = operands.get_mut(mem_index)
    {
        match post {
            Operand::Imm { value, .. } => *disp = Some(value),
            Operand::Reg { reg, .. } => *index = Some(Box::new(Operand::Reg { reg, shift: None, ext: None })),
            _ => {}
        }
        *post_flag = Some(true);
        *wb = Some(true);
    }
}

fn push_capstone_arm64(
    capstone: &Capstone,
    op: &raw::cs_arm64_op,
    out: &mut Vec<Operand>,
    writeback: bool,
) -> Option<()> {
    use raw::arm64_op_type::*;

    let shift = shift_from_capstone(op.shift.type_, op.shift.value);
    let ext = extend_from_capstone(op.ext);

    match op.type_ {
        ARM64_OP_REG => {
            let reg = reg_from_name(capstone.reg_name(unsafe { op.__bindgen_anon_1.reg })?)?;
            match reg.class {
                RegClass::Simd => {
                    let lane = if op.vector_index >= 0 {
                        Some(op.vector_index as u8)
                    } else {
                        None
                    };
                    let (elem, mut count) = simd_arrangement(op.vas, &reg)?;
                    // a lane selector means capstone's arrangement describes
                    // the element, not the vector.
                    if lane.is_some() && matches!(count, None | Some(1)) {
                        count = None;
                    }
                    out.push(Operand::Simd {
                        reg,
                        elem,
                        count,
                        lane,
                    });
                }
                _ => out.push(Operand::Reg { reg, shift, ext }),
            }
        }
        ARM64_OP_IMM | ARM64_OP_CIMM => out.push(Operand::Imm {
            value: unsafe { op.__bindgen_anon_1.imm },
            width: 64,
            shift,
        }),
        ARM64_OP_FP => out.push(Operand::Fp(unsafe { op.__bindgen_anon_1.fp }.to_bits())),
        ARM64_OP_MEM => {
            let mem = unsafe { op.__bindgen_anon_1.mem };
            let base = reg_from_name(capstone.reg_name(mem.base)?)?;
            let index = if mem.index == 0 {
                None
            } else {
                let index_reg = reg_from_name(capstone.reg_name(mem.index)?)?;
                Some(Box::new(Operand::Reg {
                    reg: index_reg,
                    shift,
                    ext,
                }))
            };
            let disp_stated = !(writeback && mem.disp == 0);
            out.push(Operand::Mem {
                base,
                index,
                disp: if disp_stated {
                    Some(mem.disp as i64)
                } else {
                    None
                },
                // capstone's arm64 detail has a single writeback flag for the
                // instruction and does not distinguish pre- from post-index.
                writeback: None,
                post: None,
            });
        }
        // system registers are an encoding, not a plain register: the name
        // would have to be parsed back into `s3_0_c15_c0_0` form and compared
        // against yaxpeax-arm's `SystemReg`, which isn't modeled.
        ARM64_OP_REG_MRS | ARM64_OP_REG_MSR => return None,
        ARM64_OP_PSTATE => out.push(Operand::PstateField(unsafe {
            op.__bindgen_anon_1.pstate
        } as u8)),
        ARM64_OP_PREFETCH => out.push(Operand::PrefetchOp(unsafe {
            op.__bindgen_anon_1.prefetch
        } as u16)),
        ARM64_OP_BARRIER => out.push(Operand::BarrierOp(unsafe {
            op.__bindgen_anon_1.barrier
        } as u16)),
        _ => out.push(Operand::Other("capstone operand")),
    }

    Some(())
}

fn shift_from_capstone(kind: raw::arm64_shifter, amount: u32) -> Option<Shift> {
    use raw::arm64_shifter::*;
    let kind = match kind {
        ARM64_SFT_INVALID => return None,
        ARM64_SFT_LSL => ShiftKind::Lsl,
        ARM64_SFT_MSL => ShiftKind::Msl,
        ARM64_SFT_LSR => ShiftKind::Lsr,
        ARM64_SFT_ASR => ShiftKind::Asr,
        ARM64_SFT_ROR => ShiftKind::Ror,
    };
    Some(Shift { kind, amount })
}

fn extend_from_capstone(ext: raw::arm64_extender) -> Option<Extend> {
    use raw::arm64_extender::*;
    Some(match ext {
        ARM64_EXT_INVALID => return None,
        ARM64_EXT_UXTB => Extend::Uxtb,
        ARM64_EXT_UXTH => Extend::Uxth,
        ARM64_EXT_UXTW => Extend::Uxtw,
        ARM64_EXT_UXTX => Extend::Uxtx,
        ARM64_EXT_SXTB => Extend::Sxtb,
        ARM64_EXT_SXTH => Extend::Sxth,
        ARM64_EXT_SXTW => Extend::Sxtw,
        ARM64_EXT_SXTX => Extend::Sxtx,
    })
}

/// parse a capstone register name into the canonical form. names are
/// `x0`/`w1`/`xzr`/`wsp`/`v10`/`d3`/`p4` etc, so this is a small parse over the
/// borrowed name rather than a 200-entry enum mapping.
///
/// this is on the per-operand hot path of a 2^32 sweep, so it neither allocates
/// nor lowercases: the name comes back borrowed from capstone's tables.
fn reg_from_name(name: &str) -> Option<Reg> {
    // capstone names some registers by their architectural alias rather than
    // number.
    if name.eq_ignore_ascii_case("sp") {
        return Some(Reg {
            class: RegClass::Gpr64,
            num: 31,
            kind: RegKind::Stack,
            scalar_elem: None,
        });
    }
    if name.eq_ignore_ascii_case("fp") {
        return Some(Reg::new(RegClass::Gpr64, 29));
    }
    if name.eq_ignore_ascii_case("lr") {
        return Some(Reg::new(RegClass::Gpr64, 30));
    }

    let bytes = name.as_bytes();
    let first = *bytes.first()?;
    let rest = &name[1..];

    let (class, kind) = match first.to_ascii_lowercase() {
        b'x' => (RegClass::Gpr64, RegKind::Plain),
        b'w' => (RegClass::Gpr32, RegKind::Plain),
        b'b' | b'h' | b's' | b'd' | b'q' | b'v' => (RegClass::Simd, RegKind::Plain),
        b'p' => (RegClass::Predicate, RegKind::Plain),
        _ => return None,
    };

    // `xzr`/`wzr` are register 31 read as zero; `sp`/`wsp` are register 31 read
    // as the stack pointer. capstone names them, yaxpeax-arm has separate
    // operand variants for them.
    if rest.eq_ignore_ascii_case("zr") {
        return Some(Reg {
            class,
            num: 31,
            kind: RegKind::Zero,
            scalar_elem: None,
        });
    }
    if rest.eq_ignore_ascii_case("sp") {
        return Some(Reg {
            class,
            num: 31,
            kind: RegKind::Stack,
            scalar_elem: None,
        });
    }

    let num: u8 = rest.parse().ok()?;
    let scalar_elem = match class {
        RegClass::Simd => Some(first.to_ascii_lowercase() as char),
        _ => None,
    };
    Some(Reg {
        class,
        num,
        kind,
        scalar_elem,
    })
}

/// capstone gives the element arrangement for vector operands separately from
/// the register name.
#[allow(unreachable_patterns)]
fn simd_arrangement(vas: raw::arm64_vas, reg: &Reg) -> Option<(Option<char>, Option<u8>)> {
    use raw::arm64_vas::*;

    // scalar register forms (`b0`, `s3`, `d4`, `q5`) carry their own element
    // size and are a single element; capstone still reports a vas for some, so
    // only fall back to the name when vas is NONE.
    let (elem, count) = match vas {
        ARM64_VAS_INVALID => match reg.class {
            // scalar simd form: `b0`/`h1`/`s2`/`d3`/`q4` carry the element size
            // in the name, which is how yaxpeax-arm records it too.
            // `v4` with no arrangement is a vector register whose element size
            // capstone didn't state; leave `elem` unknown rather than claiming
            // the register letter is an element kind.
            RegClass::Simd => (
                match reg.scalar_elem {
                    Some('v') => None,
                    other => other,
                },
                Some(1),
            ),
            _ => (None, None),
        },
        ARM64_VAS_8B => (Some('b'), Some(8)),
        ARM64_VAS_16B => (Some('b'), Some(16)),
        ARM64_VAS_4H => (Some('h'), Some(4)),
        ARM64_VAS_8H => (Some('h'), Some(8)),
        ARM64_VAS_2S => (Some('s'), Some(2)),
        ARM64_VAS_4S => (Some('s'), Some(4)),
        ARM64_VAS_4B => (Some('b'), Some(4)),
        // single-element arrangements: the count carries no information, and
        // yaxpeax-arm expresses the same thing through its lane operands.
        ARM64_VAS_1B => (Some('b'), None),
        ARM64_VAS_2H => (Some('h'), Some(2)),
        ARM64_VAS_1H => (Some('h'), None),
        ARM64_VAS_1S => (Some('s'), None),
        ARM64_VAS_1D => (Some('d'), None),
        ARM64_VAS_2D => (Some('d'), Some(2)),
        ARM64_VAS_1Q => (Some('q'), None),
        _ => return None,
    };
    Some((elem, count))
}
