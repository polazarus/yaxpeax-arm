//! dump capstone's structured operands for specific encodings, for debugging
//! the semantic comparison.
//!
//! ```text
//! cargo run --release --example capstone_detail -- a64 0xada693e8 0xa9400000
//! ```

use capstone_sys as raw;
use differential_tests::capstone::{ArchMode, Capstone};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.first().map(|s| s.as_str()) {
        Some("a64") => ArchMode::A64,
        Some("armv7") => ArchMode::Arm,
        Some("thumb") => ArchMode::Thumb,
        _ => {
            eprintln!("usage: capstone_detail <a64|armv7|thumb> <word>...");
            return;
        }
    };

    let mut capstone = Capstone::new_with_detail(mode, true).expect("can open capstone");
    let mut text = String::new();

    for word in args.iter().skip(1) {
        let word = u32::from_str_radix(word.trim_start_matches("0x"), 16).expect("hex word");
        let bytes = word.to_le_bytes();
        let decoded = capstone.disasm(&bytes, &mut text).is_some();
        println!("{:08x} decoded={} text={:?}", word, decoded, text);
        if !decoded {
            continue;
        }

        let insn = capstone.current_insn().expect("insn");
        let detail = unsafe { insn.detail.as_ref() }.expect("detail");
        if mode == ArchMode::A64 {
            let arm64 = unsafe { detail.__bindgen_anon_1.arm64 };
            println!(
                "  op_count={} writeback={} cc={:?}",
                arm64.op_count, arm64.writeback, arm64.cc
            );
            for index in 0..arm64.op_count as usize {
                let op = &arm64.operands[index];
                let extra = match op.type_ {
                    raw::arm64_op_type::ARM64_OP_REG => {
                        let id = unsafe { op.__bindgen_anon_1.reg };
                        format!("reg={:?}", capstone.reg_name(id))
                    }
                    raw::arm64_op_type::ARM64_OP_IMM => {
                        format!("imm={}", unsafe { op.__bindgen_anon_1.imm })
                    }
                    raw::arm64_op_type::ARM64_OP_MEM => {
                        let mem = unsafe { op.__bindgen_anon_1.mem };
                        format!(
                            "base={:?} index={:?} disp={}",
                            capstone.reg_name(mem.base),
                            capstone.reg_name(mem.index),
                            mem.disp
                        )
                    }
                    other => format!("{:?}", other),
                };
                println!(
                    "  op[{}] type={:?} vas={:?} shift={:?}/{} ext={:?} vecidx={} {}",
                    index,
                    op.type_,
                    op.vas,
                    op.shift.type_,
                    op.shift.value,
                    op.ext,
                    op.vector_index,
                    extra
                );
            }
        }
    }
}
