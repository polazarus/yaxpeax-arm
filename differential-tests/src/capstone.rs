//! thin wrapper over `capstone-sys` for the bits `yaxcmp` needs: open a handle
//! for one architecture/mode, disassemble a single instruction, and get its
//! textual form back.
//!
//! capstone handles are not `Sync` and carry internal state, so every worker
//! thread gets its own `Capstone`. see `lib.rs`.

use capstone_sys as raw;
use std::ffi::CStr;
use std::fmt::Write;

/// `CS_MODE_THUMB` from `arm.h`. capstone-sys doesn't export the mode bits as
/// named constants.
const CS_MODE_THUMB: u32 = 1 << 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchMode {
    /// `CS_ARCH_ARM64`, a64.
    A64,
    /// `CS_ARCH_ARM` in arm mode.
    Arm,
    /// `CS_ARCH_ARM` in thumb mode.
    Thumb,
}

pub struct Capstone {
    handle: raw::csh,
    insn: *mut raw::cs_insn,
    mode: ArchMode,
    detailed: bool,
}

impl Capstone {
    pub fn new(mode: ArchMode) -> Result<Self, String> {
        Capstone::new_with_detail(mode, false)
    }

    /// `detail` enables operand/register detail, which the semantic comparison
    /// needs and the text comparison doesn't (and pays for, so it's opt-in).
    pub fn new_with_detail(mode: ArchMode, detail: bool) -> Result<Self, String> {
        let (arch, cs_mode_bits) = match mode {
            ArchMode::A64 => (raw::cs_arch::CS_ARCH_ARM64, 0),
            ArchMode::Arm => (raw::cs_arch::CS_ARCH_ARM, 0),
            ArchMode::Thumb => (raw::cs_arch::CS_ARCH_ARM, CS_MODE_THUMB),
        };

        let mut handle: raw::csh = 0;
        let err = unsafe { raw::cs_open(arch, raw::cs_mode(cs_mode_bits), &mut handle) };
        if err != raw::cs_err::CS_ERR_OK {
            return Err(format!("cs_open failed for {:?}: {:?}", mode, err));
        }

        // with detail on, capstone writes through `cs_insn::detail`; `cs_malloc`
        // below allocates it for us, so this has to be set before allocation.
        let detail_value = if detail {
            raw::cs_opt_value::CS_OPT_ON as usize
        } else {
            raw::cs_opt_value::CS_OPT_OFF as usize
        };
        let err = unsafe { raw::cs_option(handle, raw::cs_opt_type::CS_OPT_DETAIL, detail_value) };
        if err != raw::cs_err::CS_ERR_OK {
            unsafe { raw::cs_close(&mut handle) };
            return Err(format!("cs_option failed for {:?}: {:?}", mode, err));
        }

        let insn = unsafe { raw::cs_malloc(handle) };
        if insn.is_null() {
            unsafe { raw::cs_close(&mut handle) };
            return Err(format!("cs_malloc failed for {:?}", mode));
        }

        Ok(Capstone {
            handle,
            insn,
            mode,
            detailed: detail,
        })
    }

    /// the instruction from the last successful `disasm`/`disasm_detailed`.
    pub fn current_insn(&self) -> Option<&raw::cs_insn> {
        unsafe { self.insn.as_ref() }
    }

    /// mnemonic from the last successful disassembly, borrowed from capstone's
    /// instruction buffer.
    pub fn mnemonic(&self) -> Option<&str> {
        let insn = self.current_insn()?;
        unsafe { CStr::from_ptr(insn.mnemonic.as_ptr()).to_str().ok() }
    }

    pub fn has_detail(&self) -> bool {
        self.detailed
    }

    /// name of `reg` (a capstone register id), borrowed from capstone's static
    /// tables. no allocation, so this can be used per-operand in a 2^32 sweep.
    pub fn reg_name(&self, reg: u32) -> Option<&str> {
        let ptr = unsafe { raw::cs_reg_name(self.handle, reg) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// capstone's arm disassembler keeps IT-block state in the handle, so a
    /// decode of `it eq` changes how the *next* decode is printed (`ldreq`)
    /// even when the next call is an unrelated encoding. the sweep decodes every
    /// encoding independently, so the state has to be dropped whenever an IT
    /// block is opened; otherwise every encoding after an `it` is reported with
    /// a borrowed condition.
    fn reset(&mut self) -> Result<(), String> {
        unsafe {
            raw::cs_free(self.insn, 1);
            raw::cs_close(&mut self.handle);
        }
        self.insn = std::ptr::null_mut();

        // adopt the new handle's resources; `forget` keeps its `Drop` from
        // freeing what self now owns.
        let fresh = Capstone::new_with_detail(self.mode, self.detailed)?;
        self.handle = fresh.handle;
        self.insn = fresh.insn;
        std::mem::forget(fresh);
        Ok(())
    }

    /// disassemble exactly one instruction from the front of `bytes`, writing
    /// `"<mnemonic> <op_str>"` into `out`. returns the number of bytes consumed,
    /// or `None` if capstone rejects this encoding.
    pub fn disasm(&mut self, bytes: &[u8], out: &mut String) -> Option<usize> {
        let mut ptr: *const u8 = bytes.as_ptr();
        let mut size: usize = bytes.len();
        let mut address: u64 = 0;

        let ok = unsafe {
            raw::cs_disasm_iter(
                self.handle,
                &mut ptr,
                &mut size,
                &mut address,
                self.insn,
            )
        };
        if !ok {
            return None;
        }

        out.clear();
        // both fields are fixed-size, nul-terminated buffers that capstone fills
        // in; they are always valid utf-8 for arm.
        unsafe {
            let mnemonic = CStr::from_ptr((*self.insn).mnemonic.as_ptr()).to_str().ok()?;
            let op_str = CStr::from_ptr((*self.insn).op_str.as_ptr()).to_str().ok()?;
            // matches the formatting the original inline tests used, including
            // the trailing space for instructions with no operands.
            write!(out, "{} {}", mnemonic, op_str).ok()?;
            let size = (*self.insn).size as usize;
            let opens_it_block = self.mode == ArchMode::Thumb
                && mnemonic.len() >= 2
                && mnemonic[..2].eq_ignore_ascii_case("it");
            if opens_it_block {
                // ignore failure here: a handle that can't be reopened will
                // fail loudly on the next decode anyway.
                let _ = self.reset();
            }
            Some(size)
        }
    }
}

impl Drop for Capstone {
    fn drop(&mut self) {
        unsafe {
            raw::cs_free(self.insn, 1);
            raw::cs_close(&mut self.handle);
        }
    }
}
