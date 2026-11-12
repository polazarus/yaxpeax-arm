//! measures where differential sweep time goes, and what batching capstone buys.
//!
//! ```text
//! cargo run --release --example decode_bench -- a64 0x00000000 0x00ffffff
//! cargo run --release --example decode_bench -- thumb 0x0000 0xffff
//! ```
//!
//! phases:
//! - `yax`: yaxpeax-arm decode + formatting, no capstone. the floor.
//! - `cs-iter`: current approach, one cs_disasm_iter per encoding.
//! - `cs-batch-noskip`: cs_disasm over a chunk, resyncing by hand after each
//!   failure.
//! - `cs-batch-skipdata`: cs_disasm over a chunk with CS_OPT_SKIPDATA on.
//!
//! chunk size is fixed at 16384 encodings.

use std::ffi::CStr;
use std::fmt::Write;
use std::time::Instant;

use capstone_sys as raw;
use yaxpeax_arch::{Arch, Decoder};

const CHUNK_WORDS: usize = 16384;

#[derive(Clone, Copy)]
enum Mode {
    A64,
    Arm,
    Thumb,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.get(0).map(|s| s.as_str()) {
        Some("a64") => Mode::A64,
        Some("armv7") | Some("arm") => Mode::Arm,
        Some("thumb") => Mode::Thumb,
        _ => {
            eprintln!("usage: decode_bench <a64|armv7|thumb> <start> <end>");
            return;
        }
    };
    let start = parse(&args[1]);
    let end = parse(&args[2]);
    let words: Vec<u32> = (start..=end).map(|w| w as u32).collect();
    println!("{} encodings {:#x}..={:#x}", words.len(), start, end);

    bench("yax (decode + format only)", words.len(), || yax_phase(mode, &words));
    bench("capstone iter, per encoding", words.len(), || iter_phase(mode, &words));
    bench("capstone batch, manual resync", words.len(), || batch_phase(mode, &words, false));
    bench("capstone batch, skipdata", words.len(), || batch_phase(mode, &words, true));
}

fn bench(name: &str, words: usize, mut f: impl FnMut() -> u64) {
    let start = Instant::now();
    let decoded = f();
    let elapsed = start.elapsed();
    println!(
        "  {:<36} {:>8.3}s  {:>6.1}M enc/s  ({} decoded)",
        name,
        elapsed.as_secs_f64(),
        words as f64 / elapsed.as_secs_f64() / 1e6,
        decoded
    );
}

fn parse(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).unwrap()
    } else {
        s.parse().unwrap()
    }
}

fn yax_phase(mode: Mode, words: &[u32]) -> u64 {
    let a64 = <yaxpeax_arm::armv8::a64::ARMv8 as Arch>::Decoder::default();
    let arm = <yaxpeax_arm::armv7::ARMv7 as Arch>::Decoder::default()
        .with_thumb_mode(matches!(mode, Mode::Thumb))
        .allow_nonconforming(true);
    let mut text = String::new();
    let mut decoded = 0;
    for &word in words {
        let bytes = word.to_le_bytes();
        text.clear();
        let ok = match mode {
            Mode::A64 => a64
                .decode(&mut yaxpeax_arch::U8Reader::new(&bytes))
                .map(|inst| write!(text, "{}", inst).unwrap())
                .is_ok(),
            _ => arm
                .decode(&mut yaxpeax_arch::U8Reader::new(&bytes))
                .map(|inst| write!(text, "{}", inst).unwrap())
                .is_ok(),
        };
        if ok {
            decoded += 1;
        }
    }
    decoded
}

fn open(mode: Mode, skipdata: bool) -> raw::csh {
    let (arch, bits) = match mode {
        Mode::A64 => (raw::cs_arch::CS_ARCH_ARM64, 0),
        Mode::Arm => (raw::cs_arch::CS_ARCH_ARM, 0),
        Mode::Thumb => (raw::cs_arch::CS_ARCH_ARM, 1 << 4),
    };
    let mut handle: raw::csh = 0;
    unsafe {
        assert_eq!(raw::cs_open(arch, raw::cs_mode(bits), &mut handle), raw::cs_err::CS_ERR_OK);
        assert_eq!(raw::cs_option(handle, raw::cs_opt_type::CS_OPT_DETAIL, 0), raw::cs_err::CS_ERR_OK);
        if skipdata {
            assert_eq!(
                raw::cs_option(handle, raw::cs_opt_type::CS_OPT_SKIPDATA, raw::cs_opt_value::CS_OPT_ON as usize),
                raw::cs_err::CS_ERR_OK
            );
            // null callback: capstone's built-in skip size for the arch (4 for
            // arm64/arm, 2 for thumb).
            let setup = raw::cs_opt_skipdata {
                mnemonic: std::ptr::null(),
                callback: None,
                user_data: std::ptr::null_mut(),
            };
            let mut setup = setup;
            assert_eq!(
                raw::cs_option(
                    handle,
                    raw::cs_opt_type::CS_OPT_SKIPDATA_SETUP,
                    &mut setup as *mut _ as usize,
                ),
                raw::cs_err::CS_ERR_OK
            );
        }
    }
    handle
}

fn iter_phase(mode: Mode, words: &[u32]) -> u64 {
    let mut handle = open(mode, false);
    let insn = unsafe { raw::cs_malloc(handle) };
    let mut text = String::new();
    let mut decoded = 0;
    for &word in words {
        let bytes = word.to_le_bytes();
        let mut ptr = bytes.as_ptr();
        let mut size = bytes.len();
        let mut address = 0u64;
        let ok = unsafe { raw::cs_disasm_iter(handle, &mut ptr, &mut size, &mut address, insn) };
        if ok {
            unsafe {
                let mnemonic = CStr::from_ptr((*insn).mnemonic.as_ptr()).to_str().unwrap();
                if (*insn).id != 0 {
                    decoded += 1;
                }
                text.clear();
                write!(text, "{}", mnemonic).unwrap();
            }
        }
    }
    unsafe {
        raw::cs_free(insn, 1);
        raw::cs_close(&mut handle);
    }
    decoded
}

fn batch_phase(mode: Mode, words: &[u32], skipdata: bool) -> u64 {
    let mut handle = open(mode, skipdata);
    let mut code = Vec::with_capacity(CHUNK_WORDS * 4);
    let mut decoded = 0;
    let mut text = String::new();

    for chunk in words.chunks(CHUNK_WORDS) {
        code.clear();
        for &word in chunk {
            code.extend_from_slice(&word.to_le_bytes());
        }

        let mut offset = 0usize;
        while offset < code.len() {
            let mut insns: *mut raw::cs_insn = std::ptr::null_mut();
            let count = unsafe {
                raw::cs_disasm(
                    handle,
                    code[offset..].as_ptr(),
                    code.len() - offset,
                    0,
                    0,
                    &mut insns,
                )
            };

            if count == 0 {
                // nothing decodable from here; step one minimum instruction and
                // retry, which is what skipdata would do internally.
                offset += min_size(mode);
                continue;
            }

            let slice = unsafe { std::slice::from_raw_parts(insns, count) };
            let mut end = offset;
            for insn in slice {
                if insn.id != 0 {
                    decoded += 1;
                }
                text.clear();
                unsafe {
                    let mnemonic = CStr::from_ptr(insn.mnemonic.as_ptr()).to_str().unwrap();
                    write!(text, "{}", mnemonic).unwrap();
                }
                end = offset + insn.address as usize + insn.size as usize;
            }

            unsafe { raw::cs_free(insns, count) };

            if end <= offset {
                break;
            }
            // a short batch means capstone stopped on an undecodable encoding at
            // `end`; if skipdata is off, step over it and continue.
            offset = end;
        }
    }

    unsafe { raw::cs_close(&mut handle) };
    decoded
}

fn min_size(mode: Mode) -> usize {
    match mode {
        Mode::Thumb => 2,
        _ => 4,
    }
}
