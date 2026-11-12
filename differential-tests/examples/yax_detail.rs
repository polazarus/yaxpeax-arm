//! decode specific encodings with yaxpeax-arm and print the structured result.
//! panics are left uncaught, so `RUST_BACKTRACE=1` shows where a decoder panic
//! comes from.
//!
//! ```text
//! cargo run --release --example yax_detail -- armv7 0x71ebf3f2
//! cargo run --release --example yax_detail -- thumb 0xdd214d0b
//! ```

use std::fmt::Debug;
use yaxpeax_arch::{Arch, Decoder};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arch = args.first().map(|s| s.as_str()).unwrap_or("a64");

    for word in args.iter().skip(1) {
        let word = u32::from_str_radix(word.trim_start_matches("0x"), 16).expect("hex word");
        let bytes = word.to_le_bytes();
        println!("{:08x}", word);

        match arch {
            "a64" => {
                let decoder = <yaxpeax_arm::armv8::a64::ARMv8 as Arch>::Decoder::default();
                match decoder.decode(&mut yaxpeax_arch::U8Reader::new(&bytes)) {
                    Ok(inst) => print_inst(inst.opcode, &inst.operands, Some(inst.to_string())),
                    Err(e) => println!("  error: {:?}", e),
                }
            }
            "armv7" | "thumb" => {
                let decoder = <yaxpeax_arm::armv7::ARMv7 as Arch>::Decoder::default()
                    .with_thumb_mode(arch == "thumb")
                    .allow_nonconforming(true);
                match decoder.decode(&mut yaxpeax_arch::U8Reader::new(&bytes)) {
                    Ok(inst) => {
                        println!("  condition: {:?}", inst.condition);
                        print_inst(inst.opcode, &inst.operands, Some(inst.to_string()))
                    }
                    Err(e) => println!("  error: {:?}", e),
                }
            }
            other => {
                eprintln!("unknown arch: {}", other);
                return;
            }
        }
    }
}

fn print_inst<O, P>(opcode: O, operands: &[P], text: Option<String>)
where
    O: Debug + std::fmt::Display,
    P: Debug,
{
    println!("  opcode: {:?} ({})", opcode, opcode);
    for (index, operand) in operands.iter().enumerate() {
        println!("  operand[{}]: {:?}", index, operand);
    }
    if let Some(text) = text {
        println!("  text: {}", text);
    }
}
