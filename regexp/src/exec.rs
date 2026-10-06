//! The matcher, ported from regexec.c: the backtracking VM (`match_at`),
//! the match cache that makes `Regexp.linear_time?` patterns run in linear
//! time, and the search loop (`onig_search`) with its string-search
//! shortcuts. Names follow the C code.
//!
//! Positions are byte offsets into the subject held in `isize`, with -1
//! standing for the NULL pointer of C, so that comparisons keep their C
//! meaning (NULL is below every position). Every read of the subject and
//! of the program is bounds checked.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::ast::*;
use crate::bytecode::*;
use crate::compile::*;
use crate::enc::*;
use crate::error::*;
use crate::parser::bit_status_at;
use crate::syntax::*;

pub type Pos = isize;
pub const NULL_POS: Pos = -1;
pub const ONIG_REGION_NOTPOS: isize = -1;

/// The address of the `OP_FINISH` at the bottom of the stack (`FinishCode`).
const FINISH_CODE: isize = isize::MAX;
const INVALID_STACK_INDEX: isize = -1;
const INIT_MATCH_STACK_SIZE: usize = 160;

static MATCH_STACK_LIMIT_SIZE: AtomicU32 = AtomicU32::new(0);

pub fn get_match_stack_limit_size() -> u32 {
    MATCH_STACK_LIMIT_SIZE.load(Ordering::Relaxed)
}

pub fn set_match_stack_limit_size(size: u32) {
    MATCH_STACK_LIMIT_SIZE.store(size, Ordering::Relaxed);
}

/* stack type */
/* used by normal-POP */
const STK_ALT: u32 = 0x0001;
const STK_LOOK_BEHIND_NOT: u32 = 0x0002;
const STK_POS_NOT: u32 = 0x0003;
/* handled by normal-POP */
const STK_MEM_START: u32 = 0x0100;
const STK_MEM_END: u32 = 0x8200;
const STK_REPEAT_INC: u32 = 0x0300;
/* avoided by normal-POP */
const STK_NULL_CHECK_START: u32 = 0x3000;
const STK_NULL_CHECK_END: u32 = 0x5000; /* for recursive call */
const STK_MEM_END_MARK: u32 = 0x8400;
const STK_POS: u32 = 0x0500; /* used when POP-POS */
const STK_STOP_BT: u32 = 0x0600; /* mark for "(?>...)" */
const STK_REPEAT: u32 = 0x0700;
const STK_CALL_FRAME: u32 = 0x0800;
const STK_RETURN: u32 = 0x0900;
const STK_VOID: u32 = 0x0a00; /* for fill a blank */
const STK_ABSENT_POS: u32 = 0x0b00; /* for absent */
const STK_ABSENT: u32 = 0x0c00; /* absent inner loop marker */
const STK_MATCH_CACHE_POINT: u32 = 0x0d00; /* for the match cache optimization */
const STK_ATOMIC_MATCH_CACHE_POINT: u32 = 0x0e00;

/* stack type check mask */
const STK_MASK_POP_USED: u32 = 0x00ff;
const STK_MASK_TO_VOID_TARGET: u32 = 0x10ff;
const STK_MASK_MEM_END_OR_MARK: u32 = 0x8000; /* MEM_END or MEM_END_MARK */

const NUM_CACHE_OPCODES_UNINIT: i64 = 1;
const NUM_CACHE_OPCODES_IMPOSSIBLE: i64 = -1;
const MATCH_CACHE_STATUS_UNINIT: i32 = 1;
const MATCH_CACHE_STATUS_INIT: i32 = 2;
const MATCH_CACHE_STATUS_DISABLED: i32 = -1;
const MATCH_CACHE_STATUS_ENABLED: i32 = 0;

/// `OnigStackType`. The union of C is laid over four words:
///
/// | type          | w[0]       | w[1]   | w[2]      | w[3]  | num   |
/// |---------------|------------|--------|-----------|-------|-------|
/// | state         | pcode      | pstr   | pstr_prev | pkeep |       |
/// | repeat        | pcode      | count  |           |       | id    |
/// | repeat_inc    | si         |        |           |       |       |
/// | mem           |            | pstr   | start     | end   | mem   |
/// | null_check    |            | pstr   |           |       | id    |
/// | call_frame    | ret_addr   |        |           |       |       |
/// | absent_pos    | abs_pstr   | end    |           |       |       |
/// | cache point   | index      |        |           |       | mask  |
#[derive(Clone, Copy, Default, Debug)]
struct Stk {
    typ: u32,
    num: i32,
    null_check: isize,
    w: [isize; 4],
}

#[derive(Clone, Copy, Debug)]
struct CacheOpcode {
    addr: usize,
    cache_point: i64,
    outer_repeat_mem: i32,
    num_cache_points_at_outer_repeat: i64,
    num_cache_points_in_outer_repeat: i64,
    lookaround_nesting: i32,
    match_addr: Option<usize>,
}

/// The registers of a match: `struct re_registers` seen from Rust.
pub struct Region<'a> {
    pub beg: &'a mut [isize],
    pub end: &'a mut [isize],
}

impl Region<'_> {
    pub fn clear(&mut self) {
        for v in self.beg.iter_mut() {
            *v = ONIG_REGION_NOTPOS;
        }
        for v in self.end.iter_mut() {
            *v = ONIG_REGION_NOTPOS;
        }
    }
}

/// `OnigMatchArg`: state shared by the `match_at` calls of one search.
pub struct MatchArg<'c> {
    stack: Vec<Stk>,
    options: u32,
    gpos: Pos,
    best_len: isize,
    best_s: Pos,
    counter: i32,
    /// Called every 128 steps (`CHECK_INTERRUPT_IN_MATCH_AT`). Returns 0 to
    /// go on, or the code to stop with (ONIGERR_TIMEOUT, RB_REGEXP_INTERRUPTED).
    check: &'c mut dyn FnMut() -> i32,
    match_cache_status: i32,
    num_fails: i64,
    num_cache_opcodes: i64,
    cache_opcodes: Vec<CacheOpcode>,
    num_cache_points: i64,
    match_cache_buf: Vec<u8>,
}

impl<'c> MatchArg<'c> {
    pub fn new(options: u32, gpos: Pos, check: &'c mut dyn FnMut() -> i32) -> Self {
        MatchArg {
            stack: Vec::new(),
            options,
            gpos,
            best_len: ONIG_MISMATCH as isize,
            best_s: 0,
            counter: 0,
            check,
            match_cache_status: MATCH_CACHE_STATUS_UNINIT,
            num_fails: 0,
            num_cache_opcodes: NUM_CACHE_OPCODES_UNINIT,
            cache_opcodes: Vec::new(),
            num_cache_points: 0,
            match_cache_buf: Vec::new(),
        }
    }
}

// ---- match cache: which opcodes can be memoized (regexec.c 234-880) ----

struct Prog<'a> {
    reg: &'a Regex,
    p: &'a [u8],
}

impl Prog<'_> {
    #[inline]
    fn len_at(&self, p: usize) -> i32 {
        Reader { p: self.p }.i32(p)
    }
    #[inline]
    fn memnum_at(&self, p: usize) -> i32 {
        Reader { p: self.p }.i16(p) as i32
    }
    /// The program is ours, so a negative operand is a compiler bug.
    #[inline]
    fn adv(p: usize, n: i64) -> usize {
        usize::try_from(p as i64 + n).expect("bad operand in compiled program")
    }

    fn skip_simple(&self, op: u8, p: &mut usize) -> Option<()> {
        let pend = self.p.len();
        match op {
            OP_FINISH | OP_END => {}
            OP_EXACT1 => *p += 1,
            OP_EXACT2 => *p += 2,
            OP_EXACT3 => *p += 3,
            OP_EXACT4 => *p += 4,
            OP_EXACT5 => *p += 5,
            OP_EXACTN => {
                let len = self.len_at(*p);
                *p = Self::adv(*p + 4, len as i64);
            }
            OP_EXACTMB2N1 => *p += 2,
            OP_EXACTMB2N2 => *p += 4,
            OP_EXACTMB2N3 => *p += 6,
            OP_EXACTMB2N => {
                let len = self.len_at(*p);
                *p = Self::adv(*p + 4, len as i64 * 2);
            }
            OP_EXACTMB3N => {
                let len = self.len_at(*p);
                *p = Self::adv(*p + 4, len as i64 * 3);
            }
            OP_EXACTMBN => {
                let mb_len = self.len_at(*p);
                let len = self.len_at(*p + 4);
                *p = Self::adv(*p + 8, mb_len as i64 * len as i64);
            }
            OP_EXACT1_IC => {
                let len = self.reg.enc.enclen(self.p, *p, pend);
                *p += len;
            }
            OP_EXACTN_IC => {
                let len = self.len_at(*p);
                *p = Self::adv(*p + 4, len as i64);
            }
            OP_CCLASS | OP_CCLASS_NOT => *p += SIZE_BITSET as usize,
            OP_CCLASS_MB | OP_CCLASS_MB_NOT => {
                let len = self.len_at(*p);
                *p = Self::adv(*p + 4, len as i64);
            }
            OP_CCLASS_MIX | OP_CCLASS_MIX_NOT => {
                *p += SIZE_BITSET as usize;
                let len = self.len_at(*p);
                *p = Self::adv(*p + 4, len as i64);
            }
            OP_ANYCHAR | OP_ANYCHAR_ML => {}
            OP_WORD | OP_NOT_WORD | OP_WORD_BOUND | OP_NOT_WORD_BOUND | OP_WORD_BEGIN | OP_WORD_END => {}
            OP_ASCII_WORD
            | OP_NOT_ASCII_WORD
            | OP_ASCII_WORD_BOUND
            | OP_NOT_ASCII_WORD_BOUND
            | OP_ASCII_WORD_BEGIN
            | OP_ASCII_WORD_END => {}
            OP_BEGIN_BUF | OP_END_BUF | OP_BEGIN_LINE | OP_END_LINE | OP_SEMI_END_BUF | OP_BEGIN_POSITION => {}
            OP_KEEP | OP_FAIL | OP_POP => {}
            OP_JUMP => *p += SIZE_RELADDR as usize,
            OP_NULL_CHECK_START | OP_NULL_CHECK_END | OP_NULL_CHECK_END_MEMST_PUSH | OP_NULL_CHECK_END_MEMST => {
                *p += SIZE_MEMNUM as usize
            }
            OP_LOOK_BEHIND => *p += SIZE_LENGTH as usize,
            OP_SET_OPTION_PUSH | OP_SET_OPTION => *p += SIZE_OPTION as usize,
            _ => return None,
        }
        Some(())
    }

    /// `count_num_cache_opcodes_inner`. Returns 0 or an error code.
    fn count_num_cache_opcodes_inner(
        &self,
        current_repeat_mem: i32,
        lookaround_nesting: i32,
        pp: &mut usize,
        num_cache_opcodes_ptr: &mut i64,
    ) -> isize {
        let mut p = *pp;
        let pend = self.p.len();
        let mut num_cache_opcodes = *num_cache_opcodes_ptr;

        macro_rules! impossible {
            () => {{
                *num_cache_opcodes_ptr = NUM_CACHE_OPCODES_IMPOSSIBLE;
                return 0;
            }};
        }
        macro_rules! fail {
            ($r:expr) => {{
                *num_cache_opcodes_ptr = num_cache_opcodes;
                return $r;
            }};
        }

        while p < pend {
            let op = self.p[p];
            p += 1;
            match op {
                OP_ANYCHAR_STAR | OP_ANYCHAR_ML_STAR => num_cache_opcodes += 1,
                OP_ANYCHAR_STAR_PEEK_NEXT | OP_ANYCHAR_ML_STAR_PEEK_NEXT => {
                    p += 1;
                    num_cache_opcodes += 1;
                }
                OP_BACKREF1 | OP_BACKREF2 | OP_BACKREFN | OP_BACKREFN_IC | OP_BACKREF_MULTI | OP_BACKREF_MULTI_IC
                | OP_BACKREF_WITH_LEVEL => impossible!(),
                OP_MEMORY_START | OP_MEMORY_START_PUSH | OP_MEMORY_END_PUSH | OP_MEMORY_END_PUSH_REC | OP_MEMORY_END
                | OP_MEMORY_END_REC => {
                    p += SIZE_MEMNUM as usize;
                    // A memory (capture) in look-around is found.
                    if lookaround_nesting != 0 {
                        impossible!();
                    }
                }
                OP_PUSH => {
                    p += SIZE_RELADDR as usize;
                    num_cache_opcodes += 1;
                }
                OP_PUSH_OR_JUMP_EXACT1 | OP_PUSH_IF_PEEK_NEXT => {
                    p += SIZE_RELADDR as usize + 1;
                    num_cache_opcodes += 1;
                }
                OP_REPEAT | OP_REPEAT_NG => {
                    if current_repeat_mem != -1 {
                        // A nested OP_REPEAT is not yet supported.
                        impossible!();
                    }
                    let repeat_mem = self.memnum_at(p);
                    p += SIZE_MEMNUM as usize + SIZE_RELADDR as usize;
                    let range = self.reg.repeat_range[repeat_mem as usize];
                    if range.lower == 0 && range.upper == 0 {
                        let mut dummy = 0i64;
                        let result = self.count_num_cache_opcodes_inner(repeat_mem, lookaround_nesting, &mut p, &mut dummy);
                        if result < 0 || dummy < 0 {
                            fail!(result);
                        }
                    } else {
                        if range.lower == 0 {
                            num_cache_opcodes += 1;
                        }
                        let result = self.count_num_cache_opcodes_inner(
                            repeat_mem,
                            lookaround_nesting,
                            &mut p,
                            &mut num_cache_opcodes,
                        );
                        if result < 0 || num_cache_opcodes < 0 {
                            fail!(result);
                        }
                        if range.lower < range.upper {
                            num_cache_opcodes += 1;
                        }
                    }
                }
                OP_REPEAT_INC | OP_REPEAT_INC_NG => {
                    let repeat_mem = self.memnum_at(p);
                    p += SIZE_MEMNUM as usize;
                    if repeat_mem != current_repeat_mem {
                        // A lone or invalid OP_REPEAT_INC is found.
                        impossible!();
                    }
                    break;
                }
                OP_REPEAT_INC_SG | OP_REPEAT_INC_NG_SG => impossible!(),
                OP_PUSH_POS | OP_PUSH_POS_NOT | OP_PUSH_LOOK_BEHIND_NOT => {
                    if lookaround_nesting < 0 {
                        // A look-around nested in a atomic grouping is found.
                        impossible!();
                    }
                    if op == OP_PUSH_POS_NOT {
                        p += SIZE_RELADDR as usize;
                    } else if op == OP_PUSH_LOOK_BEHIND_NOT {
                        p += SIZE_RELADDR as usize + SIZE_LENGTH as usize;
                    }
                    let result = self.count_num_cache_opcodes_inner(
                        current_repeat_mem,
                        lookaround_nesting + 1,
                        &mut p,
                        &mut num_cache_opcodes,
                    );
                    if result < 0 || num_cache_opcodes < 0 {
                        fail!(result);
                    }
                }
                OP_PUSH_STOP_BT => {
                    if lookaround_nesting != 0 {
                        // A nested atomic grouping is found.
                        impossible!();
                    }
                    let result = self.count_num_cache_opcodes_inner(current_repeat_mem, -1, &mut p, &mut num_cache_opcodes);
                    if result < 0 || num_cache_opcodes < 0 {
                        fail!(result);
                    }
                }
                OP_POP_POS | OP_FAIL_POS | OP_FAIL_LOOK_BEHIND_NOT | OP_POP_STOP_BT => break,
                OP_PUSH_ABSENT_POS | OP_ABSENT_END | OP_ABSENT => impossible!(),
                OP_CALL | OP_RETURN => impossible!(),
                OP_CONDITION => impossible!(),
                OP_STATE_CHECK_PUSH
                | OP_STATE_CHECK_PUSH_OR_JUMP
                | OP_STATE_CHECK
                | OP_STATE_CHECK_ANYCHAR_STAR
                | OP_STATE_CHECK_ANYCHAR_ML_STAR => impossible!(),
                _ => {
                    if self.skip_simple(op, &mut p).is_none() {
                        return ONIGERR_UNDEFINED_BYTECODE as isize;
                    }
                }
            }
        }

        *pp = p;
        *num_cache_opcodes_ptr = num_cache_opcodes;
        0
    }

    /// count the total number of cache opcodes for allocating a match cache buffer.
    fn count_num_cache_opcodes(&self, num: &mut i64) -> isize {
        let mut p = 0usize;
        *num = 0;
        let result = self.count_num_cache_opcodes_inner(-1, 0, &mut p, num);
        if result == 0 && *num >= 0 && p != self.p.len() {
            return ONIGERR_UNDEFINED_BYTECODE as isize;
        }
        result
    }

    /// `init_cache_opcodes_inner`. `cache_opcodes` is `None` for the dry run
    /// over a `{0}` repeat, where C passes a NULL array.
    fn init_cache_opcodes_inner(
        &self,
        current_repeat_mem: i32,
        lookaround_nesting: i32,
        cache_opcodes: &mut Option<&mut Vec<CacheOpcode>>,
        pp: &mut usize,
        num_cache_points_ptr: &mut i64,
    ) -> isize {
        let mut p = *pp;
        let pend = self.p.len();
        let mut cache_point = *num_cache_points_ptr;
        let step = if lookaround_nesting != 0 { 2 } else { 1 };

        macro_rules! inc_cache_opcodes {
            ($pbegin:expr) => {
                if let Some(v) = cache_opcodes.as_deref_mut() {
                    v.push(CacheOpcode {
                        addr: $pbegin,
                        cache_point,
                        outer_repeat_mem: current_repeat_mem,
                        num_cache_points_at_outer_repeat: 0,
                        num_cache_points_in_outer_repeat: 0,
                        lookaround_nesting,
                        match_addr: None,
                    });
                    cache_point += step;
                }
            };
        }
        macro_rules! unexpected {
            () => {
                return ONIGERR_UNEXPECTED_BYTECODE as isize
            };
        }

        while p < pend {
            let pbegin = p;
            let op = self.p[p];
            p += 1;
            match op {
                OP_ANYCHAR_STAR | OP_ANYCHAR_ML_STAR => inc_cache_opcodes!(pbegin),
                OP_ANYCHAR_STAR_PEEK_NEXT | OP_ANYCHAR_ML_STAR_PEEK_NEXT => {
                    p += 1;
                    inc_cache_opcodes!(pbegin);
                }
                OP_BACKREF1 | OP_BACKREF2 | OP_BACKREFN | OP_BACKREFN_IC | OP_BACKREF_MULTI | OP_BACKREF_MULTI_IC
                | OP_BACKREF_WITH_LEVEL => unexpected!(),
                OP_MEMORY_START | OP_MEMORY_START_PUSH | OP_MEMORY_END_PUSH | OP_MEMORY_END_PUSH_REC | OP_MEMORY_END
                | OP_MEMORY_END_REC => {
                    p += SIZE_MEMNUM as usize;
                    if lookaround_nesting != 0 {
                        unexpected!();
                    }
                }
                OP_PUSH => {
                    p += SIZE_RELADDR as usize;
                    inc_cache_opcodes!(pbegin);
                }
                OP_PUSH_OR_JUMP_EXACT1 | OP_PUSH_IF_PEEK_NEXT => {
                    p += SIZE_RELADDR as usize + 1;
                    inc_cache_opcodes!(pbegin);
                }
                OP_REPEAT | OP_REPEAT_NG => {
                    let repeat_mem = self.memnum_at(p);
                    p += SIZE_MEMNUM as usize + SIZE_RELADDR as usize;
                    let range = self.reg.repeat_range[repeat_mem as usize];
                    if range.lower == 0 && range.upper == 0 {
                        let mut dummy_points = 0i64;
                        let mut none = None;
                        let result =
                            self.init_cache_opcodes_inner(repeat_mem, lookaround_nesting, &mut none, &mut p, &mut dummy_points);
                        if result != 0 {
                            return result;
                        }
                    } else {
                        if range.lower == 0 {
                            inc_cache_opcodes!(pbegin);
                        }
                        let mut num_cache_points_in_repeat = 0i64;
                        let num_cache_points_at_repeat = cache_point;
                        let first_in_repeat = cache_opcodes.as_deref().map(|v| v.len()).unwrap_or(0);
                        let result = self.init_cache_opcodes_inner(
                            repeat_mem,
                            lookaround_nesting,
                            cache_opcodes,
                            &mut p,
                            &mut num_cache_points_in_repeat,
                        );
                        if result != 0 {
                            return result;
                        }
                        if range.lower < range.upper {
                            inc_cache_opcodes!(pbegin);
                            if cache_opcodes.is_some() {
                                cache_point -= step;
                            }
                        }
                        let repeat_bounds: i64 =
                            if range.upper == 0x7fffffff { 1 } else { (range.upper - range.lower) as i64 };
                        cache_point += num_cache_points_in_repeat * range.lower as i64
                            + (num_cache_points_in_repeat + step) * repeat_bounds;
                        if let Some(v) = cache_opcodes.as_deref_mut() {
                            for co in &mut v[first_in_repeat..] {
                                co.num_cache_points_at_outer_repeat = num_cache_points_at_repeat;
                                co.num_cache_points_in_outer_repeat = num_cache_points_in_repeat;
                            }
                        }
                    }
                }
                OP_REPEAT_INC | OP_REPEAT_INC_NG => {
                    p += SIZE_MEMNUM as usize;
                    break;
                }
                OP_REPEAT_INC_SG | OP_REPEAT_INC_NG_SG => unexpected!(),
                OP_PUSH_POS | OP_PUSH_POS_NOT | OP_PUSH_LOOK_BEHIND_NOT | OP_PUSH_STOP_BT => {
                    if op == OP_PUSH_POS_NOT {
                        p += SIZE_RELADDR as usize;
                    } else if op == OP_PUSH_LOOK_BEHIND_NOT {
                        p += SIZE_RELADDR as usize + SIZE_LENGTH as usize;
                    }
                    let nesting = if op == OP_PUSH_STOP_BT { -1 } else { lookaround_nesting + 1 };
                    let first = cache_opcodes.as_deref().map(|v| v.len()).unwrap_or(0);
                    let result = self.init_cache_opcodes_inner(
                        current_repeat_mem,
                        nesting,
                        cache_opcodes,
                        &mut p,
                        &mut cache_point,
                    );
                    if result != 0 {
                        return result;
                    }
                    let match_addr = p - 1;
                    if let Some(v) = cache_opcodes.as_deref_mut() {
                        for co in &mut v[first..] {
                            if co.match_addr.is_none() {
                                co.match_addr = Some(match_addr);
                            }
                        }
                    }
                }
                OP_POP_POS | OP_FAIL_POS | OP_FAIL_LOOK_BEHIND_NOT | OP_POP_STOP_BT => break,
                OP_ABSENT_END | OP_ABSENT => unexpected!(),
                OP_CALL | OP_RETURN => unexpected!(),
                OP_CONDITION => unexpected!(),
                OP_STATE_CHECK_PUSH
                | OP_STATE_CHECK_PUSH_OR_JUMP
                | OP_STATE_CHECK
                | OP_STATE_CHECK_ANYCHAR_STAR
                | OP_STATE_CHECK_ANYCHAR_ML_STAR => unexpected!(),
                _ => {
                    if self.skip_simple(op, &mut p).is_none() {
                        return ONIGERR_UNDEFINED_BYTECODE as isize;
                    }
                }
            }
        }

        *pp = p;
        *num_cache_points_ptr = cache_point;
        0
    }

    /// collect cache opcodes from the given regex program, and compute the total number of cache points.
    fn init_cache_opcodes(&self, cache_opcodes: &mut Vec<CacheOpcode>, num_cache_points: &mut i64) -> isize {
        let mut p = 0usize;
        *num_cache_points = 0;
        let mut some = Some(cache_opcodes);
        let result = self.init_cache_opcodes_inner(-1, 0, &mut some, &mut p, num_cache_points);
        if result == 0 && p != self.p.len() {
            return ONIGERR_UNDEFINED_BYTECODE as isize;
        }
        result
    }
}

/// `onig_check_linear_time`
pub fn check_linear_time(reg: &Regex) -> bool {
    let prog = Prog { reg, p: &reg.program };
    let mut n = 0i64;
    prog.count_num_cache_opcodes(&mut n);
    n != NUM_CACHE_OPCODES_IMPOSSIBLE
}

fn check_extended_match_cache_point(buf: &[u8], index: usize, mask: u8) -> bool {
    if mask & 0x80 != 0 { (buf[index + 1] & 0x01) > 0 } else { (buf[index] & (mask << 1)) > 0 }
}

fn memoize_extended_match_cache_point(buf: &mut [u8], index: usize, mask: u8) {
    buf[index] |= mask;
    if mask & 0x80 != 0 {
        buf[index + 1] |= 0x01;
    } else {
        buf[index] |= mask << 1;
    }
}

enum CacheCheck {
    Miss,
    Fail,
    StopBtFail,
    Jump(usize),
}

// ---- the VM ----

#[derive(PartialEq, Eq)]
enum Flow {
    /// The instruction consumed a character: `sprev` becomes its start.
    Next,
    /// `sprev` stays as the instruction left it.
    Jump,
    Fail,
    Finish,
}

struct Vm<'a, 'm, 'c> {
    reg: &'a Regex,
    prog: Reader<'a>,
    plen: usize,
    sb: &'a [u8],
    end: Pos,
    enc: Enc,
    option: u32,
    case_fold_flag: u32,
    num_mem: i32,
    pop_level: i32,
    msa: &'m mut MatchArg<'c>,
    stk: Vec<Stk>,
    repeat_stk: Vec<isize>,
    mem_start_stk: Vec<isize>,
    mem_end_stk: Vec<isize>,
}

type VmResult<T> = Result<T, isize>;

impl Vm<'_, '_, '_> {
    #[inline]
    fn ch(&self, s: Pos) -> u8 {
        self.sb[s as usize]
    }

    #[inline]
    fn u(&self, s: Pos) -> usize {
        debug_assert!(s >= 0);
        s as usize
    }

    #[inline]
    fn enclen(&self, s: Pos) -> isize {
        self.enc.enclen(self.sb, self.u(s), self.end as usize) as isize
    }

    #[inline]
    fn enclen_approx(&self, s: Pos) -> isize {
        self.enc.enclen_approx(self.sb, self.u(s), self.end as usize) as isize
    }

    #[inline]
    fn mbc_to_code(&self, s: Pos, end: Pos) -> CodePoint {
        self.enc.mbc_to_code(self.sb, self.u(s), end.min(self.end) as usize)
    }

    #[inline]
    fn is_mbc_word(&self, s: Pos) -> bool {
        self.enc.is_mbc_word(self.sb, self.u(s), self.end as usize)
    }

    #[inline]
    fn is_mbc_ascii_word(&self, s: Pos) -> bool {
        if s >= self.end {
            return false;
        }
        self.enc.is_mbc_ascii_word(self.sb, self.u(s), self.end as usize)
    }

    #[inline]
    fn is_mbc_newline(&self, s: Pos) -> bool {
        s >= 0 && self.enc.is_mbc_newline(self.sb, self.u(s), self.end as usize)
    }

    /// `ONIGENC_IS_MBC_CRNL`
    fn is_mbc_crnl(&self, p: Pos) -> bool {
        self.mbc_to_code(p, self.end) == 13 && self.mbc_to_code(p + self.enclen(p), self.end) == 10
    }

    /// `is_mbc_newline_ex`
    fn is_mbc_newline_ex(&self, p: Pos, check_prev: bool) -> bool {
        is_mbc_newline_ex(self.enc, self.sb, p, 0, self.end, self.option, check_prev)
    }

    fn prev_char_head(&self, start: Pos, s: Pos) -> Pos {
        prev_char_head(self.enc, self.sb, start, s, self.end)
    }

    // -- program operands --

    #[inline]
    fn get_length(&self, p: &mut usize) -> i32 {
        let v = self.prog.i32(*p);
        *p += 4;
        v
    }
    #[inline]
    fn get_memnum(&self, p: &mut usize) -> i32 {
        let v = self.prog.i16(*p) as i32;
        *p += 2;
        v
    }
    #[inline]
    fn get_reladdr(&self, p: &mut usize) -> i32 {
        let v = self.prog.i32(*p);
        *p += 4;
        v
    }
    #[inline]
    fn get_option(&self, p: &mut usize) -> u32 {
        let v = self.prog.u32(*p);
        *p += 4;
        v
    }
    #[inline]
    fn jump_target(p: usize, addr: i32) -> usize {
        usize::try_from(p as i64 + addr as i64).expect("jump out of program")
    }

    #[inline]
    fn bitset_at(&self, p: usize, c: u8) -> bool {
        let w = self.prog.u32(p + (c as usize / 32) * 4);
        w & (1u32 << (c as u32 % 32)) != 0
    }

    /// `onig_is_in_code_range` over the `BBuf` layout in the program.
    fn in_code_range(&self, p: usize, code: CodePoint) -> bool {
        let n = self.prog.u32(p) as usize;
        let data = p + 4;
        let (mut low, mut high) = (0usize, n);
        while low < high {
            let x = (low + high) >> 1;
            if code > self.prog.u32(data + (x * 2 + 1) * 4) {
                low = x + 1;
            } else {
                high = x;
            }
        }
        low < n && code >= self.prog.u32(data + low * 2 * 4)
    }

    // -- stack --

    fn grow(&mut self) -> VmResult<()> {
        let n = self.stk.capacity().max(INIT_MATCH_STACK_SIZE);
        let mut want = n * 2;
        let limit = get_match_stack_limit_size() as usize;
        if limit != 0 && want > limit {
            if self.stk.len() >= limit {
                return Err(ONIGERR_MATCH_STACK_LIMIT_OVER as isize);
            }
            want = limit;
        }
        let extra = want.saturating_sub(self.stk.len());
        self.stk.try_reserve_exact(extra).map_err(|_| ONIGERR_MEMORY as isize)
    }

    #[inline]
    fn push(&mut self, typ: u32, num: i32, w: [isize; 4]) -> VmResult<()> {
        if self.stk.len() == self.stk.capacity() {
            self.grow()?;
        }
        let null_check = match self.stk.last() {
            Some(l) => l.null_check,
            None => 0,
        };
        self.stk.push(Stk { typ, num, null_check, w });
        Ok(())
    }

    #[inline]
    fn push_state(&mut self, typ: u32, pcode: isize, s: Pos, sprev: Pos, keep: Pos) -> VmResult<()> {
        self.push(typ, 0, [pcode, s, sprev, keep])
    }

    #[inline]
    fn top(&self) -> isize {
        self.stk.len() as isize
    }

    fn push_mem_start(&mut self, mnum: i32, s: Pos) -> VmResult<()> {
        let m = mnum as usize;
        let (start, end) = (self.mem_start_stk[m], self.mem_end_stk[m]);
        let idx = self.top();
        self.push(STK_MEM_START, mnum, [0, s, start, end])?;
        self.mem_start_stk[m] = idx;
        self.mem_end_stk[m] = INVALID_STACK_INDEX;
        Ok(())
    }

    fn push_mem_end(&mut self, mnum: i32, s: Pos) -> VmResult<()> {
        let m = mnum as usize;
        let (start, end) = (self.mem_start_stk[m], self.mem_end_stk[m]);
        let idx = self.top();
        self.push(STK_MEM_END, mnum, [0, s, start, end])?;
        self.mem_end_stk[m] = idx;
        Ok(())
    }

    fn push_null_check(&mut self, typ: u32, cnum: i32, s: Pos) -> VmResult<()> {
        if self.stk.len() == self.stk.capacity() {
            self.grow()?;
        }
        let idx = self.top();
        self.stk.push(Stk { typ, num: cnum, null_check: idx, w: [0, s, 0, 0] });
        Ok(())
    }

    #[inline]
    fn restore_mem(&mut self, e: &Stk) {
        let m = e.num as usize;
        self.mem_start_stk[m] = e.w[2];
        self.mem_end_stk[m] = e.w[3];
    }

    #[inline]
    fn memoize_match_cache_point(&mut self, e: &Stk) {
        if e.typ == STK_MATCH_CACHE_POINT {
            self.msa.match_cache_buf[e.w[0] as usize] |= e.num as u8;
        } else if e.typ == STK_ATOMIC_MATCH_CACHE_POINT {
            memoize_extended_match_cache_point(&mut self.msa.match_cache_buf, e.w[0] as usize, e.num as u8);
        }
    }

    #[inline]
    fn pop_entry(&mut self) -> Stk {
        self.stk.pop().expect("match stack underflow")
    }

    /// `STACK_POP`: returns the entry that stopped the pop.
    fn stack_pop(&mut self) -> Stk {
        loop {
            let e = self.pop_entry();
            if e.typ & STK_MASK_POP_USED != 0 {
                return e;
            }
            match self.pop_level {
                STACK_POP_LEVEL_FREE => {}
                STACK_POP_LEVEL_MEM_START => {
                    if e.typ == STK_MEM_START {
                        self.restore_mem(&e);
                    }
                }
                _ => {
                    if e.typ == STK_MEM_START || e.typ == STK_MEM_END {
                        self.restore_mem(&e);
                    } else if e.typ == STK_REPEAT_INC {
                        self.stk[e.w[0] as usize].w[1] -= 1;
                    }
                }
            }
            self.memoize_match_cache_point(&e);
        }
    }

    fn memoize_lookaround(&mut self, e: &mut Stk) {
        if e.typ == STK_MATCH_CACHE_POINT {
            e.typ = STK_VOID;
            memoize_extended_match_cache_point(&mut self.msa.match_cache_buf, e.w[0] as usize, e.num as u8);
        }
    }

    /// `STACK_POP_TIL_POS_NOT`, `STACK_POP_TIL_LOOK_BEHIND_NOT`,
    /// `STACK_POP_TIL_ABSENT`.
    fn pop_til(&mut self, til: u32) {
        loop {
            let mut e = self.pop_entry();
            if e.typ == til {
                break;
            } else if e.typ == STK_MEM_START || e.typ == STK_MEM_END {
                self.restore_mem(&e);
            } else if e.typ == STK_REPEAT_INC {
                self.stk[e.w[0] as usize].w[1] -= 1;
            } else if til == STK_POS_NOT && e.typ & STK_MASK_TO_VOID_TARGET != 0 {
                self.msa.num_fails += 1;
            }
            if til == STK_POS_NOT {
                self.memoize_lookaround(&mut e);
            }
        }
    }

    /// `STACK_POS_END`: returns the index of the STK_POS entry.
    fn stack_pos_end(&mut self) -> usize {
        let mut k = self.stk.len();
        loop {
            k -= 1;
            let typ = self.stk[k].typ;
            if typ & STK_MASK_TO_VOID_TARGET != 0 {
                self.msa.num_fails += 1;
                self.stk[k].typ = STK_VOID;
            } else if typ == STK_POS {
                self.stk[k].typ = STK_VOID;
                return k;
            }
            let mut e = self.stk[k];
            self.memoize_lookaround(&mut e);
            self.stk[k].typ = e.typ;
        }
    }

    fn stack_stop_bt_end(&mut self) {
        let mut k = self.stk.len();
        loop {
            k -= 1;
            let typ = self.stk[k].typ;
            if typ & STK_MASK_TO_VOID_TARGET != 0 {
                self.msa.num_fails += 1;
                self.stk[k].typ = STK_VOID;
            } else if typ == STK_STOP_BT {
                self.stk[k].typ = STK_VOID;
                return;
            } else if typ == STK_MATCH_CACHE_POINT {
                self.stk[k].typ = STK_ATOMIC_MATCH_CACHE_POINT;
            }
        }
    }

    fn stack_stop_bt_fail(&mut self) {
        loop {
            let e = self.pop_entry();
            if e.typ == STK_STOP_BT {
                // C sets the type to VOID on the entry it leaves behind the top.
                return;
            }
            if e.typ == STK_MATCH_CACHE_POINT {
                memoize_extended_match_cache_point(&mut self.msa.match_cache_buf, e.w[0] as usize, e.num as u8);
            }
        }
    }

    /// `STACK_GET_MEM_START`: index of the matching MEM_START (or 0).
    fn stack_get_mem_start(&self, mnum: i32) -> usize {
        let mut level = 0;
        let mut k = self.stk.len();
        while k > 0 {
            k -= 1;
            let e = &self.stk[k];
            if e.typ & STK_MASK_MEM_END_OR_MARK != 0 && e.num == mnum {
                level += 1;
            } else if e.typ == STK_MEM_START && e.num == mnum {
                if level == 0 {
                    break;
                }
                level -= 1;
            }
        }
        k
    }

    /// `STACK_NULL_CHECK*`. Returns 0, 1 or -1 (empty, but position changed).
    fn stack_null_check(&self, id: i32, s: Pos, rec: bool, memst: bool) -> i32 {
        let top_nc = self.stk.last().expect("match stack underflow").null_check;
        let mut k = top_nc as usize + 1;
        let mut level = 0;
        loop {
            k -= 1;
            let e = &self.stk[k];
            if e.typ == STK_NULL_CHECK_START {
                if e.num == id {
                    if !rec || level == 0 {
                        if !memst {
                            return (e.w[1] == s) as i32;
                        }
                        if e.w[1] != s {
                            return 0;
                        }
                        let mut isnull = 1;
                        let mut j = k;
                        while j < self.stk.len() {
                            let m = &self.stk[j];
                            if m.typ == STK_MEM_START {
                                if m.w[3] == INVALID_STACK_INDEX {
                                    return 0;
                                }
                                let endp = if bit_status_at(self.reg.bt_mem_end, m.num) {
                                    self.stk[m.w[3] as usize].w[1]
                                } else {
                                    m.w[3]
                                };
                                if self.stk[m.w[2] as usize].w[1] != endp {
                                    return 0;
                                } else if endp != s {
                                    isnull = -1; /* empty, but position changed */
                                }
                            }
                            j += 1;
                        }
                        return isnull;
                    } else {
                        level -= 1;
                    }
                }
            } else if rec && e.typ == STK_NULL_CHECK_END && (!memst || e.num == id) {
                level += 1;
            }
        }
    }

    /// `STACK_GET_REPEAT`
    fn stack_get_repeat(&self, id: i32) -> usize {
        let mut level = 0;
        let mut k = self.stk.len();
        loop {
            k -= 1;
            let e = &self.stk[k];
            if e.typ == STK_REPEAT {
                if level == 0 && e.num == id {
                    return k;
                }
            } else if e.typ == STK_CALL_FRAME {
                level -= 1;
            } else if e.typ == STK_RETURN {
                level += 1;
            }
        }
    }

    /// `STACK_RETURN`
    fn stack_return(&self) -> usize {
        let mut level = 0;
        let mut k = self.stk.len();
        loop {
            k -= 1;
            let e = &self.stk[k];
            if e.typ == STK_CALL_FRAME {
                if level == 0 {
                    return e.w[0] as usize;
                }
                level -= 1;
            } else if e.typ == STK_RETURN {
                level += 1;
            }
        }
    }

    // -- captures --

    fn mem_pos(&self, mem: i32, start: bool) -> Pos {
        let m = mem as usize;
        if start {
            if bit_status_at(self.reg.bt_mem_start, mem) {
                self.stk[self.mem_start_stk[m] as usize].w[1]
            } else {
                self.mem_start_stk[m]
            }
        } else if bit_status_at(self.reg.bt_mem_end, mem) {
            self.stk[self.mem_end_stk[m] as usize].w[1]
        } else {
            self.mem_end_stk[m]
        }
    }

    /// `string_cmp_ic`.
    fn string_cmp_ic(&self, s1: Pos, ps2: &mut Pos, mblen: isize, text_end: Pos) -> bool {
        let mut buf1 = [0u8; ONIGENC_MBC_CASE_FOLD_MAXLEN];
        let mut buf2 = [0u8; ONIGENC_MBC_CASE_FOLD_MAXLEN];
        let te = text_end as usize;
        let mut s1 = self.u(s1);
        let mut s2 = self.u(*ps2);
        let end1 = s1 + mblen.max(0) as usize;
        while s1 < end1 {
            if s1 >= te || s2 >= te {
                return false;
            }
            let len1 = self.enc.mbc_case_fold(self.case_fold_flag, self.sb, &mut s1, te, &mut buf1);
            let len2 = self.enc.mbc_case_fold(self.case_fold_flag, self.sb, &mut s2, te, &mut buf2);
            if len1 != len2 || buf1[..len1] != buf2[..len2] {
                return false;
            }
        }
        *ps2 = s2 as Pos;
        true
    }

    fn mem_is_in_memp(&self, mem: i32, num: i32, memp: usize) -> bool {
        (0..num as usize).any(|i| self.prog.i16(memp + i * 2) as i32 == mem)
    }

    fn backref_match_at_nested_level(
        &self,
        ignore_case: bool,
        nest: i32,
        mem_num: i32,
        memp: usize,
        s: &mut Pos,
        send: Pos,
    ) -> bool {
        let mut pend = NULL_POS;
        let mut level = 0;
        let mut k = self.stk.len();
        while k > 0 {
            k -= 1;
            let e = &self.stk[k];
            if e.typ == STK_CALL_FRAME {
                level -= 1;
            } else if e.typ == STK_RETURN {
                level += 1;
            } else if level == nest {
                if e.typ == STK_MEM_START {
                    if self.mem_is_in_memp(e.num, mem_num, memp) {
                        let pstart = e.w[1];
                        if pend != NULL_POS {
                            if pend - pstart > send - *s {
                                return false; /* or goto next_mem; */
                            }
                            let mut ss = *s;
                            if ignore_case {
                                if !self.string_cmp_ic(pstart, &mut ss, pend - pstart, send) {
                                    return false; /* or goto next_mem; */
                                }
                            } else {
                                let n = (pend - pstart).max(0) as usize;
                                let a = &self.sb[self.u(pstart)..self.u(pstart) + n];
                                let b = &self.sb[self.u(ss)..self.u(ss) + n];
                                if a != b {
                                    return false; /* or goto next_mem; */
                                }
                                ss += n as isize;
                            }
                            *s = ss;
                            return true;
                        }
                    }
                } else if e.typ == STK_MEM_END && self.mem_is_in_memp(e.num, mem_num, memp) {
                    pend = e.w[1];
                }
            }
        }
        false
    }

    // -- match cache --

    fn find_cache_point(&self, p: usize) -> Option<(i64, usize)> {
        let ops = &self.msa.cache_opcodes;
        let num = ops.len() as i64;
        let op = self.prog.u8(p);
        let is_inc = op == OP_REPEAT_INC || op == OP_REPEAT_INC_NG;

        // bsearch_cache_opcodes
        let (mut l, mut r, mut m) = (0i64, num - 1, 0i64);
        while l <= r {
            m = (l + r) / 2;
            let a = ops[m as usize].addr;
            if a == p {
                break;
            }
            if a < p {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        if !(0 <= m && m < num && ops[m as usize].addr == p) {
            return None;
        }
        let co = &ops[m as usize];
        let cache_point = co.cache_point;
        if co.outer_repeat_mem == -1 {
            return Some((cache_point, m as usize));
        }
        let at = co.num_cache_points_at_outer_repeat;
        let inr = co.num_cache_points_in_outer_repeat;
        let range = self.reg.repeat_range[co.outer_repeat_mem as usize];
        let stkp = &self.stk[self.repeat_stk[co.outer_repeat_mem as usize] as usize];
        let count = if is_inc { stkp.w[1] - 1 } else { stkp.w[1] } as i64;
        let (lower, upper) = (range.lower as i64, range.upper as i64);
        let v = if count < lower {
            at + inr * count + cache_point
        } else if upper == 0x7fffffff {
            at + inr * (lower - if is_inc { 1 } else { 0 }) + if is_inc { 0 } else { 1 } + cache_point
        } else {
            at + inr * (lower - 1) + (inr + 1) * (count - lower + 1) + cache_point
        };
        Some((v, m as usize))
    }

    fn check_match_cache(&mut self, pbegin: usize, s: Pos) -> VmResult<CacheCheck> {
        if self.msa.match_cache_status != MATCH_CACHE_STATUS_ENABLED {
            return Ok(CacheCheck::Miss);
        }
        if let Some((cache_point, idx)) = self.find_cache_point(pbegin) {
            let mcp = self.msa.num_cache_points * s as i64 + cache_point;
            let index = (mcp >> 3) as usize;
            let mask = 1u8 << (mcp & 7);
            if self.msa.match_cache_buf[index] & mask != 0 {
                let co = self.msa.cache_opcodes[idx];
                return Ok(if co.lookaround_nesting == 0 {
                    CacheCheck::Fail
                } else if co.lookaround_nesting < 0 {
                    if check_extended_match_cache_point(&self.msa.match_cache_buf, index, mask) {
                        CacheCheck::StopBtFail
                    } else {
                        CacheCheck::Fail
                    }
                } else if check_extended_match_cache_point(&self.msa.match_cache_buf, index, mask) {
                    CacheCheck::Jump(co.match_addr.expect("match cache point without match address"))
                } else {
                    CacheCheck::Fail
                });
            }
            self.push(STK_MATCH_CACHE_POINT, mask as i32, [index as isize, 0, 0, 0])?;
        }
        Ok(CacheCheck::Miss)
    }

    /// The match cache bookkeeping done on every failure.
    fn on_fail_match_cache(&mut self) -> VmResult<()> {
        let msa = &mut *self.msa;
        let len = self.end as i64;
        if msa.match_cache_status == MATCH_CACHE_STATUS_DISABLED {
            return Ok(());
        }
        msa.num_fails += 1;
        if msa.num_fails < len * msa.num_cache_opcodes {
            return Ok(());
        }
        let prog = Prog { reg: self.reg, p: &self.reg.program };
        if msa.match_cache_status == MATCH_CACHE_STATUS_UNINIT {
            msa.match_cache_status = MATCH_CACHE_STATUS_INIT;
            let r = prog.count_num_cache_opcodes(&mut msa.num_cache_opcodes);
            if r < 0 {
                return Err(ONIGERR_UNDEFINED_BYTECODE as isize);
            }
        }
        if msa.num_cache_opcodes == NUM_CACHE_OPCODES_IMPOSSIBLE || msa.num_cache_opcodes == 0 {
            msa.match_cache_status = MATCH_CACHE_STATUS_DISABLED;
            return Ok(());
        }
        if msa.num_fails < len * msa.num_cache_opcodes {
            return Ok(());
        }
        if msa.cache_opcodes.is_empty() {
            msa.match_cache_status = MATCH_CACHE_STATUS_ENABLED;
            let mut ops = Vec::new();
            ops.try_reserve(msa.num_cache_opcodes as usize).map_err(|_| ONIGERR_MEMORY as isize)?;
            let r = prog.init_cache_opcodes(&mut ops, &mut msa.num_cache_points);
            if r < 0 {
                return Err(if r == ONIGERR_UNEXPECTED_BYTECODE as isize { r } else { ONIGERR_UNDEFINED_BYTECODE as isize });
            }
            msa.cache_opcodes = ops;
        }
        if msa.match_cache_buf.is_empty() {
            let length = (self.end as u64) + 1;
            let points = (msa.num_cache_points as u64).checked_mul(length).ok_or(ONIGERR_MEMORY as isize)?;
            if points >= i64::MAX as u64 {
                return Err(ONIGERR_MEMORY as isize);
            }
            let buf_len = (points >> 3) + if points & 7 != 0 { 1 } else { 0 } + 1;
            let mut buf = Vec::new();
            buf.try_reserve_exact(buf_len as usize).map_err(|_| ONIGERR_MEMORY as isize)?;
            buf.resize(buf_len as usize, 0);
            msa.match_cache_buf = buf;
        }
        Ok(())
    }

    fn check_interrupt(&mut self) -> VmResult<()> {
        self.msa.counter += 1;
        if self.msa.counter >= 128 {
            self.msa.counter = 0;
            let r = (self.msa.check)();
            if r != 0 {
                return Err(r as isize);
            }
        }
        Ok(())
    }

    /// `match_at`: match data(str - end) from position (sstart).
    /// if sstart == str then set sprev to NULL.
    fn run(&mut self, sstart: Pos, sprev_in: Pos, region: &mut Option<Region<'_>>) -> VmResult<isize> {
        let option = self.option;
        let mut p: usize = 0;
        let mut s: Pos = sstart;
        let mut sprev: Pos = sprev_in;
        let mut pkeep: Pos = sstart;
        let mut best_len: isize = ONIG_MISMATCH as isize;

        self.stk.clear();
        if self.stk.capacity() < INIT_MATCH_STACK_SIZE {
            self.stk.try_reserve(INIT_MATCH_STACK_SIZE).map_err(|_| ONIGERR_MEMORY as isize)?;
        }
        self.push_state(STK_ALT, FINISH_CODE, 0, 0, 0)?; /* bottom stack */

        macro_rules! data_ensure {
            ($l:lifetime, $n:expr) => {
                if s + ($n) > self.end {
                    break $l Flow::Fail;
                }
            };
        }
        macro_rules! cache_check {
            ($l:lifetime, $pbegin:expr) => {
                match self.check_match_cache($pbegin, s)? {
                    CacheCheck::Miss => {}
                    CacheCheck::Fail => break $l Flow::Fail,
                    CacheCheck::StopBtFail => {
                        self.stack_stop_bt_fail();
                        break $l Flow::Fail;
                    }
                    CacheCheck::Jump(a) => {
                        p = a;
                        break $l Flow::Jump;
                    }
                }
            };
        }

        loop {
            if p == FINISH_CODE as usize {
                break;
            }
            let pbegin = p;
            let sbegin = s;
            let op = self.prog.u8(p);
            p += 1;

            let flow = 'op: {
                match op {
                    OP_END => {
                        let n = s - sstart;
                        let mut record = n > best_len;
                        if record && is_find_longest(option) {
                            if n > self.msa.best_len {
                                self.msa.best_len = n;
                                self.msa.best_s = sstart;
                            } else {
                                record = false;
                            }
                        }
                        if record {
                            best_len = n;
                            if let Some(region) = region.as_mut() {
                                region.beg[0] = if pkeep > s { s } else { pkeep };
                                region.end[0] = s;
                                for i in 1..=self.num_mem {
                                    let iu = i as usize;
                                    if self.mem_end_stk[iu] != INVALID_STACK_INDEX {
                                        region.beg[iu] = self.mem_pos(i, true);
                                        region.end[iu] = self.mem_pos(i, false);
                                    } else {
                                        region.beg[iu] = ONIG_REGION_NOTPOS;
                                        region.end[iu] = ONIG_REGION_NOTPOS;
                                    }
                                }
                            }
                        }
                        if is_find_condition(option) {
                            if is_find_not_empty(option) && s == sstart {
                                best_len = ONIG_MISMATCH as isize;
                                break 'op Flow::Fail; /* for retry */
                            }
                            if is_find_longest(option) && s < self.end {
                                break 'op Flow::Fail; /* for retry */
                            }
                        }
                        /* default behavior: return first-matching result. */
                        Flow::Finish
                    }

                    OP_EXACT1 => {
                        data_ensure!('op, 1);
                        if self.prog.u8(p) != self.ch(s) {
                            break 'op Flow::Fail;
                        }
                        p += 1;
                        s += 1;
                        Flow::Next
                    }

                    OP_EXACT1_IC => {
                        let mut lowbuf = [0u8; ONIGENC_MBC_CASE_FOLD_MAXLEN];
                        data_ensure!('op, 1);
                        let mut su = self.u(s);
                        let len = self.enc.mbc_case_fold(self.case_fold_flag, self.sb, &mut su, self.end as usize, &mut lowbuf);
                        s = su as Pos;
                        data_ensure!('op, 0);
                        // C compares `len` bytes of the program from here even
                        // when they run into the next instruction.
                        match self.prog.p.get(p..p + len) {
                            Some(pat) if pat == &lowbuf[..len] => p += len,
                            _ => break 'op Flow::Fail,
                        }
                        Flow::Next
                    }

                    OP_EXACT2 | OP_EXACT3 | OP_EXACT4 | OP_EXACT5 => {
                        let n = (op - OP_EXACT1 + 1) as isize;
                        data_ensure!('op, n);
                        for _ in 0..n {
                            if self.prog.u8(p) != self.ch(s) {
                                break 'op Flow::Fail;
                            }
                            p += 1;
                            s += 1;
                        }
                        sprev = s - 1;
                        Flow::Jump
                    }

                    OP_EXACTN => {
                        let tlen = self.get_length(&mut p) as isize;
                        data_ensure!('op, tlen);
                        for _ in 0..tlen.max(0) {
                            if self.prog.u8(p) != self.ch(s) {
                                break 'op Flow::Fail;
                            }
                            p += 1;
                            s += 1;
                        }
                        sprev = s - 1;
                        Flow::Jump
                    }

                    OP_EXACTN_IC => {
                        let mut lowbuf = [0u8; ONIGENC_MBC_CASE_FOLD_MAXLEN];
                        let tlen = self.get_length(&mut p) as usize;
                        let endp = p + tlen;
                        while p < endp {
                            sprev = s;
                            data_ensure!('op, 1);
                            let mut su = self.u(s);
                            let len =
                                self.enc.mbc_case_fold(self.case_fold_flag, self.sb, &mut su, self.end as usize, &mut lowbuf);
                            s = su as Pos;
                            data_ensure!('op, 0);
                            match self.prog.p.get(p..p + len) {
                                Some(pat) if pat == &lowbuf[..len] => p += len,
                                _ => break 'op Flow::Fail,
                            }
                        }
                        Flow::Jump
                    }

                    OP_EXACTMB2N1 => {
                        data_ensure!('op, 2);
                        for _ in 0..2 {
                            if self.prog.u8(p) != self.ch(s) {
                                break 'op Flow::Fail;
                            }
                            p += 1;
                            s += 1;
                        }
                        Flow::Next
                    }

                    OP_EXACTMB2N2 | OP_EXACTMB2N3 => {
                        let chars = if op == OP_EXACTMB2N2 { 2 } else { 3 };
                        data_ensure!('op, chars * 2);
                        for c in 0..chars {
                            if c == chars - 1 {
                                sprev = s;
                            }
                            for _ in 0..2 {
                                if self.prog.u8(p) != self.ch(s) {
                                    break 'op Flow::Fail;
                                }
                                p += 1;
                                s += 1;
                            }
                        }
                        Flow::Jump
                    }

                    OP_EXACTMB2N | OP_EXACTMB3N => {
                        let w: isize = if op == OP_EXACTMB2N { 2 } else { 3 };
                        let tlen = self.get_length(&mut p) as isize;
                        data_ensure!('op, tlen * w);
                        for _ in 0..(tlen * w).max(0) {
                            if self.prog.u8(p) != self.ch(s) {
                                break 'op Flow::Fail;
                            }
                            p += 1;
                            s += 1;
                        }
                        sprev = s - w;
                        Flow::Jump
                    }

                    OP_EXACTMBN => {
                        let tlen = self.get_length(&mut p) as isize; /* mb-len */
                        let tlen2 = self.get_length(&mut p) as isize * tlen; /* string len */
                        data_ensure!('op, tlen2);
                        for _ in 0..tlen2.max(0) {
                            if self.prog.u8(p) != self.ch(s) {
                                break 'op Flow::Fail;
                            }
                            p += 1;
                            s += 1;
                        }
                        sprev = s - tlen;
                        Flow::Jump
                    }

                    OP_CCLASS | OP_CCLASS_NOT => {
                        data_ensure!('op, 1);
                        let hit = self.bitset_at(p, self.ch(s));
                        if hit != (op == OP_CCLASS) {
                            break 'op Flow::Fail;
                        }
                        p += SIZE_BITSET as usize;
                        s += self.enclen(s); /* OP_CCLASS can match mb-code. \D, \S */
                        Flow::Next
                    }

                    OP_CCLASS_MB | OP_CCLASS_MIX => {
                        if op == OP_CCLASS_MB {
                            if !self.enc.is_mbc_head(self.sb, self.u(s), self.end as usize) {
                                break 'op Flow::Fail;
                            }
                        } else {
                            data_ensure!('op, 1);
                            if !self.enc.is_mbc_head(self.sb, self.u(s), self.end as usize) {
                                if !self.bitset_at(p, self.ch(s)) {
                                    break 'op Flow::Fail;
                                }
                                p += SIZE_BITSET as usize;
                                let tlen = self.get_length(&mut p);
                                p = Self::jump_target(p, tlen);
                                s += 1;
                                break 'op Flow::Next;
                            }
                            p += SIZE_BITSET as usize;
                        }
                        // cclass_mb:
                        let tlen = self.get_length(&mut p);
                        data_ensure!('op, 1);
                        let mb_len = self.enclen_approx(s);
                        data_ensure!('op, mb_len);
                        let ss = s;
                        s += mb_len;
                        let code = self.mbc_to_code(ss, s);
                        if !self.in_code_range(p, code) {
                            break 'op Flow::Fail;
                        }
                        p = Self::jump_target(p, tlen);
                        Flow::Next
                    }

                    OP_CCLASS_MB_NOT | OP_CCLASS_MIX_NOT => {
                        data_ensure!('op, 1);
                        if op == OP_CCLASS_MB_NOT {
                            if !self.enc.is_mbc_head(self.sb, self.u(s), self.end as usize) {
                                s += 1;
                                let tlen = self.get_length(&mut p);
                                p = Self::jump_target(p, tlen);
                                break 'op Flow::Next; /* cc_mb_not_success */
                            }
                        } else if self.enc.is_mbc_head(self.sb, self.u(s), self.end as usize) {
                            p += SIZE_BITSET as usize;
                        } else {
                            if self.bitset_at(p, self.ch(s)) {
                                break 'op Flow::Fail;
                            }
                            p += SIZE_BITSET as usize;
                            let tlen = self.get_length(&mut p);
                            p = Self::jump_target(p, tlen);
                            s += 1;
                            break 'op Flow::Next;
                        }
                        // cclass_mb_not:
                        let tlen = self.get_length(&mut p);
                        let mb_len = self.enclen(s);
                        if s + mb_len > self.end {
                            data_ensure!('op, 1);
                            s = self.end;
                            p = Self::jump_target(p, tlen);
                            break 'op Flow::Next;
                        }
                        let ss = s;
                        s += mb_len;
                        let code = self.mbc_to_code(ss, s);
                        if self.in_code_range(p, code) {
                            break 'op Flow::Fail;
                        }
                        p = Self::jump_target(p, tlen);
                        Flow::Next
                    }

                    OP_ANYCHAR => {
                        data_ensure!('op, 1);
                        let n = self.enclen_approx(s);
                        data_ensure!('op, n);
                        if self.is_mbc_newline_ex(s, false) {
                            break 'op Flow::Fail;
                        }
                        s += n;
                        Flow::Next
                    }

                    OP_ANYCHAR_ML => {
                        data_ensure!('op, 1);
                        let n = self.enclen_approx(s);
                        data_ensure!('op, n);
                        s += n;
                        Flow::Next
                    }

                    OP_ANYCHAR_STAR | OP_ANYCHAR_ML_STAR => {
                        let ml = op == OP_ANYCHAR_ML_STAR;
                        while s < self.end {
                            cache_check!('op, pbegin);
                            self.push_state(STK_ALT, p as isize, s, sprev, pkeep)?;
                            let n = self.enclen_approx(s);
                            if ml {
                                if n > 1 {
                                    data_ensure!('op, n);
                                    sprev = s;
                                    s += n;
                                } else {
                                    sprev = s;
                                    s += 1;
                                }
                            } else {
                                data_ensure!('op, n);
                                if self.is_mbc_newline_ex(s, false) {
                                    break 'op Flow::Fail;
                                }
                                sprev = s;
                                s += n;
                            }
                        }
                        Flow::Jump
                    }

                    OP_ANYCHAR_STAR_PEEK_NEXT | OP_ANYCHAR_ML_STAR_PEEK_NEXT => {
                        let ml = op == OP_ANYCHAR_ML_STAR_PEEK_NEXT;
                        let c = self.prog.u8(p);
                        while s < self.end {
                            cache_check!('op, pbegin);
                            if c == self.ch(s) {
                                self.push_state(STK_ALT, p as isize + 1, s, sprev, pkeep)?;
                            } else {
                                // num_fails counts this case too, for invoking
                                // the cache optimization correctly.
                                self.msa.num_fails += 1;
                            }
                            let n = self.enclen_approx(s);
                            if ml {
                                if n > 1 {
                                    data_ensure!('op, n);
                                    sprev = s;
                                    s += n;
                                } else {
                                    sprev = s;
                                    s += 1;
                                }
                            } else {
                                data_ensure!('op, n);
                                if self.is_mbc_newline_ex(s, false) {
                                    break 'op Flow::Fail;
                                }
                                sprev = s;
                                s += n;
                            }
                        }
                        p += 1;
                        Flow::Next
                    }

                    OP_WORD | OP_NOT_WORD => {
                        data_ensure!('op, 1);
                        if self.is_mbc_word(s) != (op == OP_WORD) {
                            break 'op Flow::Fail;
                        }
                        s += self.enclen(s);
                        Flow::Next
                    }

                    OP_ASCII_WORD | OP_NOT_ASCII_WORD => {
                        data_ensure!('op, 1);
                        if self.is_mbc_ascii_word(s) != (op == OP_ASCII_WORD) {
                            break 'op Flow::Fail;
                        }
                        s += self.enclen(s);
                        Flow::Next
                    }

                    OP_WORD_BOUND | OP_ASCII_WORD_BOUND | OP_NOT_WORD_BOUND | OP_NOT_ASCII_WORD_BOUND => {
                        let ascii = op == OP_ASCII_WORD_BOUND || op == OP_NOT_ASCII_WORD_BOUND;
                        let not = op == OP_NOT_WORD_BOUND || op == OP_NOT_ASCII_WORD_BOUND;
                        let word = |vm: &Self, x: Pos| if ascii { vm.is_mbc_ascii_word(x) } else { vm.is_mbc_word(x) };
                        if s == 0 {
                            if !not {
                                data_ensure!('op, 1);
                                if !word(self, s) {
                                    break 'op Flow::Fail;
                                }
                            } else if s < self.end && word(self, s) {
                                break 'op Flow::Fail;
                            }
                        } else if s == self.end {
                            if word(self, sprev) == not {
                                break 'op Flow::Fail;
                            }
                        } else if (word(self, s) == word(self, sprev)) != not {
                            break 'op Flow::Fail;
                        }
                        Flow::Jump
                    }

                    OP_WORD_BEGIN | OP_ASCII_WORD_BEGIN => {
                        let ascii = op == OP_ASCII_WORD_BEGIN;
                        let word = |vm: &Self, x: Pos| if ascii { vm.is_mbc_ascii_word(x) } else { vm.is_mbc_word(x) };
                        if s < self.end && word(self, s) && (s == 0 || !word(self, sprev)) {
                            break 'op Flow::Jump;
                        }
                        Flow::Fail
                    }

                    OP_WORD_END | OP_ASCII_WORD_END => {
                        let ascii = op == OP_ASCII_WORD_END;
                        let word = |vm: &Self, x: Pos| if ascii { vm.is_mbc_ascii_word(x) } else { vm.is_mbc_word(x) };
                        if s != 0 && word(self, sprev) && (s == self.end || !word(self, s)) {
                            break 'op Flow::Jump;
                        }
                        Flow::Fail
                    }

                    OP_BEGIN_BUF => {
                        if s != 0 || is_notbos(self.msa.options) {
                            break 'op Flow::Fail;
                        }
                        Flow::Jump
                    }

                    OP_END_BUF => {
                        if s != self.end || is_noteos(self.msa.options) {
                            break 'op Flow::Fail;
                        }
                        Flow::Jump
                    }

                    OP_BEGIN_LINE => {
                        if s == 0 {
                            if is_notbol(self.msa.options) {
                                break 'op Flow::Fail;
                            }
                            break 'op Flow::Jump;
                        } else if self.is_mbc_newline(sprev)
                            && !(is_newline_crlf(option) && self.is_mbc_crnl(sprev))
                            && s != self.end
                        {
                            break 'op Flow::Jump;
                        }
                        Flow::Fail
                    }

                    OP_END_LINE => {
                        if s == self.end {
                            if is_noteol(self.msa.options) {
                                break 'op Flow::Fail;
                            }
                            break 'op Flow::Jump;
                        } else if self.is_mbc_newline_ex(s, true) {
                            break 'op Flow::Jump;
                        }
                        Flow::Fail
                    }

                    OP_SEMI_END_BUF => {
                        if s == self.end {
                            if is_noteol(self.msa.options) {
                                break 'op Flow::Fail;
                            }
                            break 'op Flow::Jump;
                        } else if self.is_mbc_newline_ex(s, true) {
                            let mut ss = s + self.enclen(s);
                            if ss == self.end {
                                break 'op Flow::Jump;
                            } else if is_newline_crlf(option) && self.is_mbc_crnl(s) {
                                ss += self.enclen(ss);
                                if ss == self.end {
                                    break 'op Flow::Jump;
                                }
                            }
                        }
                        Flow::Fail
                    }

                    OP_BEGIN_POSITION => {
                        if s != self.msa.gpos {
                            break 'op Flow::Fail;
                        }
                        Flow::Jump
                    }

                    OP_MEMORY_START_PUSH => {
                        let mem = self.get_memnum(&mut p);
                        self.push_mem_start(mem, s)?;
                        Flow::Jump
                    }

                    OP_MEMORY_START => {
                        let mem = self.get_memnum(&mut p) as usize;
                        self.mem_start_stk[mem] = s;
                        self.mem_end_stk[mem] = INVALID_STACK_INDEX;
                        Flow::Jump
                    }

                    OP_MEMORY_END_PUSH => {
                        let mem = self.get_memnum(&mut p);
                        self.push_mem_end(mem, s)?;
                        Flow::Jump
                    }

                    OP_MEMORY_END => {
                        let mem = self.get_memnum(&mut p) as usize;
                        self.mem_end_stk[mem] = s;
                        Flow::Jump
                    }

                    OP_KEEP => {
                        pkeep = s;
                        Flow::Jump
                    }

                    OP_MEMORY_END_PUSH_REC => {
                        let mem = self.get_memnum(&mut p);
                        let k = self.stack_get_mem_start(mem); /* should be before push mem-self.end. */
                        self.mem_start_stk[mem as usize] = k as isize;
                        self.push_mem_end(mem, s)?;
                        Flow::Jump
                    }

                    OP_MEMORY_END_REC => {
                        let mem = self.get_memnum(&mut p);
                        self.mem_end_stk[mem as usize] = s;
                        let k = self.stack_get_mem_start(mem);
                        self.mem_start_stk[mem as usize] =
                            if bit_status_at(self.reg.bt_mem_start, mem) { k as isize } else { self.stk[k].w[1] };
                        self.push(STK_MEM_END_MARK, mem, [0; 4])?;
                        Flow::Jump
                    }

                    OP_BACKREF1 | OP_BACKREF2 | OP_BACKREFN | OP_BACKREFN_IC => {
                        let mem = match op {
                            OP_BACKREF1 => 1,
                            OP_BACKREF2 => 2,
                            _ => self.get_memnum(&mut p),
                        };
                        /* if you want to remove following line,
                           you should check in parse and compile time. */
                        if mem > self.num_mem
                            || mem < 0
                            || self.mem_end_stk[mem as usize] == INVALID_STACK_INDEX
                            || self.mem_start_stk[mem as usize] == INVALID_STACK_INDEX
                        {
                            break 'op Flow::Fail;
                        }
                        let pstart = self.mem_pos(mem, true);
                        let pend = self.mem_pos(mem, false);
                        let n = pend - pstart;
                        data_ensure!('op, n);
                        sprev = s;
                        if op == OP_BACKREFN_IC {
                            if !self.string_cmp_ic(pstart, &mut s, n, self.end) {
                                break 'op Flow::Fail;
                            }
                        } else {
                            let n = n.max(0) as usize;
                            if self.sb[self.u(pstart)..self.u(pstart) + n] != self.sb[self.u(s)..self.u(s) + n] {
                                break 'op Flow::Fail;
                            }
                            s += n as isize;
                        }
                        loop {
                            let len = self.enclen_approx(sprev);
                            if sprev + len < s {
                                sprev += len;
                            } else {
                                break;
                            }
                        }
                        Flow::Jump
                    }

                    OP_BACKREF_MULTI | OP_BACKREF_MULTI_IC => {
                        let ic = op == OP_BACKREF_MULTI_IC;
                        let tlen = self.get_length(&mut p);
                        let mut matched = false;
                        let mut i = 0;
                        while i < tlen {
                            let mem = self.get_memnum(&mut p);
                            i += 1;
                            let mu = mem as usize;
                            if self.mem_end_stk[mu] == INVALID_STACK_INDEX || self.mem_start_stk[mu] == INVALID_STACK_INDEX {
                                continue;
                            }
                            let pstart = self.mem_pos(mem, true);
                            let pend = self.mem_pos(mem, false);
                            let n = pend - pstart;
                            if s + n > self.end {
                                continue;
                            }
                            sprev = s;
                            let mut swork = s;
                            let ok = if ic {
                                self.string_cmp_ic(pstart, &mut swork, n, self.end)
                            } else {
                                let n = n.max(0) as usize;
                                let ok = self.sb[self.u(pstart)..self.u(pstart) + n] == self.sb[self.u(s)..self.u(s) + n];
                                swork += n as isize;
                                ok
                            };
                            if !ok {
                                continue;
                            }
                            s = swork;
                            loop {
                                let len = if ic { self.enclen(sprev) } else { self.enclen_approx(sprev) };
                                if sprev + len < s {
                                    sprev += len;
                                } else {
                                    break;
                                }
                            }
                            p += SIZE_MEMNUM as usize * (tlen - i) as usize;
                            matched = true;
                            break; /* success */
                        }
                        if !matched {
                            break 'op Flow::Fail;
                        }
                        Flow::Jump
                    }

                    OP_BACKREF_WITH_LEVEL => {
                        let ic = self.get_option(&mut p);
                        let level = self.get_length(&mut p);
                        let tlen = self.get_length(&mut p);
                        sprev = s;
                        if self.backref_match_at_nested_level(ic != 0, level, tlen, p, &mut s, self.end) {
                            loop {
                                let len = self.enclen(sprev);
                                if sprev + len < s {
                                    sprev += len;
                                } else {
                                    break;
                                }
                            }
                            p += SIZE_MEMNUM as usize * tlen as usize;
                        } else {
                            break 'op Flow::Fail;
                        }
                        Flow::Jump
                    }

                    OP_NULL_CHECK_START => {
                        let mem = self.get_memnum(&mut p); /* mem: null check id */
                        self.push_null_check(STK_NULL_CHECK_START, mem, s)?;
                        Flow::Jump
                    }

                    OP_NULL_CHECK_END | OP_NULL_CHECK_END_MEMST | OP_NULL_CHECK_END_MEMST_PUSH => {
                        let mem = self.get_memnum(&mut p); /* mem: null check id */
                        let isnull = match op {
                            OP_NULL_CHECK_END => self.stack_null_check(mem, s, false, false),
                            OP_NULL_CHECK_END_MEMST => self.stack_null_check(mem, s, false, true),
                            _ => self.stack_null_check(mem, s, true, true),
                        };
                        if isnull != 0 {
                            if isnull == -1 {
                                break 'op Flow::Fail;
                            }
                            // null_check_found: empty loop founded, skip next instruction
                            let next = self.prog.u8(p);
                            p += 1;
                            match next {
                                OP_JUMP | OP_PUSH => p += SIZE_RELADDR as usize,
                                OP_REPEAT_INC | OP_REPEAT_INC_NG | OP_REPEAT_INC_SG | OP_REPEAT_INC_NG_SG => {
                                    p += SIZE_MEMNUM as usize
                                }
                                _ => return Err(ONIGERR_UNEXPECTED_BYTECODE as isize),
                            }
                        } else if op == OP_NULL_CHECK_END_MEMST_PUSH {
                            self.push_null_check(STK_NULL_CHECK_END, mem, s)?;
                        }
                        Flow::Jump
                    }

                    OP_JUMP => {
                        let addr = self.get_reladdr(&mut p);
                        p = Self::jump_target(p, addr);
                        self.check_interrupt()?;
                        Flow::Jump
                    }

                    OP_PUSH => {
                        let addr = self.get_reladdr(&mut p);
                        cache_check!('op, pbegin);
                        self.push_state(STK_ALT, Self::jump_target(p, addr) as isize, s, sprev, pkeep)?;
                        Flow::Jump
                    }

                    OP_POP => {
                        self.pop_entry();
                        // Onigmo makes a loop that is pairwise disjoint to what
                        // follows atomic; count it as a failure for the cache.
                        self.msa.num_fails += 1;
                        Flow::Jump
                    }

                    OP_PUSH_IF_PEEK_NEXT => {
                        let addr = self.get_reladdr(&mut p);
                        cache_check!('op, pbegin);
                        // C reads the byte at `self.end` here; there is none.
                        if s < self.end && self.prog.u8(p) == self.ch(s) {
                            p += 1;
                            self.push_state(STK_ALT, Self::jump_target(p, addr) as isize, s, sprev, pkeep)?;
                            break 'op Flow::Jump;
                        }
                        p += 1;
                        self.msa.num_fails += 1;
                        Flow::Jump
                    }

                    OP_REPEAT | OP_REPEAT_NG => {
                        let mem = self.get_memnum(&mut p); /* mem: OP_REPEAT ID */
                        let addr = self.get_reladdr(&mut p);
                        if self.stk.len() == self.stk.capacity() {
                            self.grow()?;
                        }
                        self.repeat_stk[mem as usize] = self.top();
                        self.push(STK_REPEAT, mem, [p as isize, 0, 0, 0])?;
                        if self.reg.repeat_range[mem as usize].lower == 0 {
                            cache_check!('op, pbegin);
                            if op == OP_REPEAT {
                                self.push_state(STK_ALT, Self::jump_target(p, addr) as isize, s, sprev, pkeep)?;
                            } else {
                                self.push_state(STK_ALT, p as isize, s, sprev, pkeep)?;
                                p = Self::jump_target(p, addr);
                            }
                        }
                        Flow::Jump
                    }

                    OP_REPEAT_INC | OP_REPEAT_INC_SG => {
                        let mem = self.get_memnum(&mut p); /* mem: OP_REPEAT ID */
                        let si = if op == OP_REPEAT_INC {
                            self.repeat_stk[mem as usize] as usize
                        } else {
                            self.stack_get_repeat(mem)
                        };
                        // repeat_inc:
                        self.stk[si].w[1] += 1;
                        let count = self.stk[si].w[1];
                        let range = self.reg.repeat_range[mem as usize];
                        if count >= range.upper as isize {
                            /* self.end of repeat. Nothing to do. */
                        } else if count >= range.lower as isize {
                            if op == OP_REPEAT_INC {
                                let r = self.check_match_cache(pbegin, s)?;
                                if !matches!(r, CacheCheck::Miss) {
                                    self.stk[si].w[1] -= 1;
                                }
                                match r {
                                    CacheCheck::Miss => {}
                                    CacheCheck::Fail => break 'op Flow::Fail,
                                    CacheCheck::StopBtFail => {
                                        self.stack_stop_bt_fail();
                                        break 'op Flow::Fail;
                                    }
                                    CacheCheck::Jump(a) => {
                                        p = a;
                                        break 'op Flow::Jump;
                                    }
                                }
                            }
                            self.push_state(STK_ALT, p as isize, s, sprev, pkeep)?;
                            p = self.stk[si].w[0] as usize; /* Don't use stkp after PUSH. */
                        } else {
                            p = self.stk[si].w[0] as usize;
                        }
                        self.push(STK_REPEAT_INC, 0, [si as isize, 0, 0, 0])?;
                        self.check_interrupt()?;
                        Flow::Jump
                    }

                    OP_REPEAT_INC_NG | OP_REPEAT_INC_NG_SG => {
                        let mem = self.get_memnum(&mut p); /* mem: OP_REPEAT ID */
                        let si = if op == OP_REPEAT_INC_NG {
                            self.repeat_stk[mem as usize] as usize
                        } else {
                            self.stack_get_repeat(mem)
                        };
                        // repeat_inc_ng:
                        self.stk[si].w[1] += 1;
                        let count = self.stk[si].w[1];
                        let range = self.reg.repeat_range[mem as usize];
                        if count < range.upper as isize {
                            if count >= range.lower as isize {
                                let pcode = self.stk[si].w[0];
                                self.push(STK_REPEAT_INC, 0, [si as isize, 0, 0, 0])?;
                                if op == OP_REPEAT_INC_NG {
                                    cache_check!('op, pbegin);
                                }
                                self.push_state(STK_ALT, pcode, s, sprev, pkeep)?;
                            } else {
                                p = self.stk[si].w[0] as usize;
                                self.push(STK_REPEAT_INC, 0, [si as isize, 0, 0, 0])?;
                            }
                        } else if count == range.upper as isize {
                            self.push(STK_REPEAT_INC, 0, [si as isize, 0, 0, 0])?;
                        }
                        self.check_interrupt()?;
                        Flow::Jump
                    }

                    OP_PUSH_POS => {
                        self.push_state(STK_POS, 0, s, sprev, pkeep)?;
                        Flow::Jump
                    }

                    OP_POP_POS => {
                        let k = self.stack_pos_end();
                        s = self.stk[k].w[1];
                        sprev = self.stk[k].w[2];
                        Flow::Jump
                    }

                    OP_PUSH_POS_NOT => {
                        let addr = self.get_reladdr(&mut p);
                        self.push_state(STK_POS_NOT, Self::jump_target(p, addr) as isize, s, sprev, pkeep)?;
                        Flow::Jump
                    }

                    OP_FAIL_POS => {
                        self.pop_til(STK_POS_NOT);
                        Flow::Fail
                    }

                    OP_PUSH_STOP_BT => {
                        self.push(STK_STOP_BT, 0, [0; 4])?;
                        Flow::Jump
                    }

                    OP_POP_STOP_BT => {
                        self.stack_stop_bt_end();
                        Flow::Jump
                    }

                    OP_LOOK_BEHIND => {
                        let tlen = self.get_length(&mut p);
                        match self.enc.step_back(self.sb, 0, self.u(s), self.end as usize, tlen.max(0) as usize) {
                            None => break 'op Flow::Fail,
                            Some(q) => s = q as Pos,
                        }
                        sprev = self.prev_char_head(0, s);
                        Flow::Jump
                    }

                    OP_PUSH_LOOK_BEHIND_NOT => {
                        let addr = self.get_reladdr(&mut p);
                        let tlen = self.get_length(&mut p);
                        match self.enc.step_back(self.sb, 0, self.u(s), self.end as usize, tlen.max(0) as usize) {
                            None => {
                                /* too short case -> success. ex. /(?<!XXX)a/.match("a")
                                   If you want to change to fail, replace following line. */
                                p = Self::jump_target(p, addr);
                            }
                            Some(q) => {
                                self.push_state(
                                    STK_LOOK_BEHIND_NOT,
                                    Self::jump_target(p, addr) as isize,
                                    s,
                                    sprev,
                                    pkeep,
                                )?;
                                s = q as Pos;
                                sprev = self.prev_char_head(0, s);
                            }
                        }
                        Flow::Jump
                    }

                    OP_FAIL_LOOK_BEHIND_NOT => {
                        self.pop_til(STK_LOOK_BEHIND_NOT);
                        Flow::Fail
                    }

                    OP_PUSH_ABSENT_POS => {
                        /* Save the absent-start-pos and the original self.end-pos. */
                        self.push(STK_ABSENT_POS, 0, [s, self.end, 0, 0])?;
                        Flow::Jump
                    }

                    OP_ABSENT => {
                        let aend = self.end;
                        let selfp = p - 1;
                        let e = self.pop_entry(); /* STACK_POP_ABSENT_POS: restore self.end-pos. */
                        let absent = e.w[0];
                        self.end = e.w[1];
                        let addr = self.get_reladdr(&mut p);
                        if absent > aend && s > absent {
                            /* An empty match occurred in (?~...) at the start point.
                             * Never match. */
                            self.stack_pop();
                            break 'op Flow::Fail;
                        } else if s >= aend && s > absent {
                            if s > aend {
                                /* Only one (or less) character matched in the last iteration.
                                 * This is not a possible point. */
                                break 'op Flow::Fail;
                            }
                            /* All possible points were found. Try matching after (?~...). */
                            if s > self.end {
                                break 'op Flow::Fail;
                            }
                            p = Self::jump_target(p, addr);
                        } else if s == self.end {
                            /* At the self.end of the string, just match with it */
                            p = Self::jump_target(p, addr);
                        } else {
                            self.push_state(STK_ALT, Self::jump_target(p, addr) as isize, s, sprev, pkeep)?; /* Push possible point. */
                            let n = self.enclen(s);
                            self.push(STK_ABSENT_POS, 0, [absent, self.end, 0, 0])?; /* Save the original pos. */
                            self.push_state(STK_ALT, selfp as isize, s + n, s, pkeep)?; /* Next iteration. */
                            self.push(STK_ABSENT, 0, [0; 4])?;
                            self.end = aend;
                        }
                        Flow::Jump
                    }

                    OP_ABSENT_END => {
                        /* The pattern inside (?~...) was matched.
                         * Set the self.end-pos temporary and go to next iteration. */
                        if sprev < self.end {
                            self.end = sprev;
                        }
                        self.pop_til(STK_ABSENT);
                        Flow::Fail
                    }

                    OP_CALL => {
                        let addr = self.prog.i32(p);
                        p += 4;
                        self.push(STK_CALL_FRAME, 0, [p as isize, 0, 0, 0])?;
                        p = usize::try_from(addr).expect("bad call address");
                        Flow::Jump
                    }

                    OP_RETURN => {
                        p = self.stack_return();
                        self.push(STK_RETURN, 0, [0; 4])?;
                        Flow::Jump
                    }

                    OP_CONDITION => {
                        let mem = self.get_memnum(&mut p);
                        let addr = self.get_reladdr(&mut p);
                        if mem > self.num_mem
                            || mem < 0
                            || self.mem_end_stk[mem as usize] == INVALID_STACK_INDEX
                            || self.mem_start_stk[mem as usize] == INVALID_STACK_INDEX
                        {
                            p = Self::jump_target(p, addr);
                        }
                        Flow::Jump
                    }

                    OP_FINISH => Flow::Finish,

                    OP_FAIL => Flow::Fail,

                    _ => return Err(ONIGERR_UNDEFINED_BYTECODE as isize),
                }
            };

            match flow {
                Flow::Next => sprev = sbegin,
                Flow::Jump => {}
                Flow::Finish => break,
                Flow::Fail => {
                    let e = self.stack_pop();
                    p = if e.w[0] == FINISH_CODE { FINISH_CODE as usize } else { e.w[0] as usize };
                    s = e.w[1];
                    sprev = e.w[2];
                    pkeep = e.w[3];
                    self.on_fail_match_cache()?;
                    self.check_interrupt()?;
                }
            }
        }
        Ok(best_len)
    }
}

/// `is_mbc_newline_ex` (USE_CRNL_AS_LINE_TERMINATOR)
fn is_mbc_newline_ex(enc: Enc, sb: &[u8], p: Pos, start: Pos, end: Pos, option: u32, check_prev: bool) -> bool {
    if p < 0 || p >= end {
        return false;
    }
    let (pu, eu) = (p as usize, end as usize);
    if is_newline_crlf(option) {
        if enc.mbc_to_code(sb, pu, eu) == 0x0a {
            if check_prev {
                let prev = prev_char_head(enc, sb, start, p, end);
                !(prev != NULL_POS && enc.mbc_to_code(sb, prev as usize, eu) == 0x0d)
            } else {
                true
            }
        } else {
            let pnext = pu + enc.enclen(sb, pu, eu);
            if pnext < eu && enc.mbc_to_code(sb, pu, eu) == 0x0d && enc.mbc_to_code(sb, pnext, eu) == 0x0a {
                return true;
            }
            enc.is_mbc_newline(sb, pu, eu)
        }
    } else {
        enc.is_mbc_newline(sb, pu, eu)
    }
}

/// `onigenc_get_prev_char_head`, NULL_POS when `s` is at `start`.
fn prev_char_head(enc: Enc, sb: &[u8], start: Pos, s: Pos, end: Pos) -> Pos {
    match enc.get_prev_char_head(sb, start as usize, s as usize, end as usize) {
        Some(q) => q as Pos,
        None => NULL_POS,
    }
}

/// Runs `match_at` for one start position. `str` is position 0 and `end`
/// is `sb.len()`.
fn match_at(reg: &Regex, sb: &[u8], sstart: Pos, sprev: Pos, msa: &mut MatchArg, region: &mut Option<Region<'_>>) -> isize {
    let n_mem = reg.num_mem as usize + 1; /* ADD_NUMMEM: #0 is the whole pattern for \g<0> */
    let mut vm = Vm {
        reg,
        prog: Reader { p: &reg.program },
        plen: reg.program.len(),
        sb,
        end: sb.len() as Pos,
        enc: reg.enc,
        option: reg.options,
        case_fold_flag: reg.case_fold_flag,
        num_mem: reg.num_mem,
        pop_level: reg.stack_pop_level,
        stk: std::mem::take(&mut msa.stack),
        repeat_stk: vec![INVALID_STACK_INDEX; reg.num_repeat as usize],
        mem_start_stk: vec![INVALID_STACK_INDEX; n_mem],
        mem_end_stk: vec![INVALID_STACK_INDEX; n_mem],
        msa,
    };
    let _ = vm.plen;
    let r = match vm.run(sstart, sprev, region) {
        Ok(v) => v,
        Err(e) => e,
    };
    let stk = std::mem::take(&mut vm.stk);
    vm.msa.stack = stk; /* STACK_SAVE */
    r
}

// ---- string search used by the optimizer ----

fn slow_search(enc: Enc, target: &[u8], sb: &[u8], text: Pos, text_end: Pos, text_range: Pos) -> Option<Pos> {
    let tlen = target.len() as isize;
    let mut end = text_end - (tlen - 1);
    if end > text_range {
        end = text_range;
    }
    let mut s = text;
    let fixed = enc.min_len() == enc.max_len();
    while s < end {
        if sb[s as usize] == target[0] && sb[s as usize + 1..(s + tlen) as usize] == target[1..] {
            return Some(s);
        }
        s += if fixed { enc.max_len() as isize } else { enc.enclen(sb, s as usize, text_end as usize) as isize };
    }
    None
}

/// `str_lower_case_match`: `t` against the folded text from `p`.
fn str_lower_case_match(enc: Enc, case_fold_flag: u32, t: &[u8], sb: &[u8], p: Pos, end: Pos) -> bool {
    let mut lowbuf = [0u8; ONIGENC_MBC_CASE_FOLD_MAXLEN];
    let mut ti = 0;
    let mut pu = p as usize;
    let eu = end as usize;
    while ti < t.len() {
        if pu >= eu {
            return false;
        }
        let lowlen = enc.mbc_case_fold(case_fold_flag, sb, &mut pu, eu, &mut lowbuf);
        for &c in &lowbuf[..lowlen] {
            if ti >= t.len() || t[ti] != c {
                return false;
            }
            ti += 1;
        }
    }
    true
}

fn slow_search_ic(
    enc: Enc,
    case_fold_flag: u32,
    target: &[u8],
    sb: &[u8],
    text: Pos,
    text_end: Pos,
    text_range: Pos,
) -> Option<Pos> {
    let mut end = text_end - (target.len() as isize - 1);
    if end > text_range {
        end = text_range;
    }
    let mut s = text;
    while s < end {
        if str_lower_case_match(enc, case_fold_flag, target, sb, s, text_end) {
            return Some(s);
        }
        s += enc.enclen(sb, s as usize, text_end as usize) as isize;
    }
    None
}

fn slow_search_backward(
    enc: Enc,
    target: &[u8],
    sb: &[u8],
    text: Pos,
    adjust_text: Pos,
    text_end: Pos,
    text_start: Pos,
) -> Option<Pos> {
    let mut s = text_end - target.len() as isize;
    if s > text_start {
        s = text_start;
    } else {
        s = enc.left_adjust_char_head(sb, adjust_text as usize, s as usize, text_end as usize) as Pos;
    }
    while s >= text {
        let su = s as usize;
        if sb[su] == target[0] && sb.get(su + 1..su + target.len()) == Some(&target[1..]) {
            return Some(s);
        }
        s = prev_char_head(enc, sb, adjust_text, s, text_end);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn slow_search_backward_ic(
    enc: Enc,
    case_fold_flag: u32,
    target: &[u8],
    sb: &[u8],
    text: Pos,
    adjust_text: Pos,
    text_end: Pos,
    text_start: Pos,
) -> Option<Pos> {
    let mut s = text_end - target.len() as isize;
    if s > text_start {
        s = text_start;
    } else {
        s = enc.left_adjust_char_head(sb, adjust_text as usize, s as usize, text_end as usize) as Pos;
    }
    while s >= text {
        if str_lower_case_match(enc, case_fold_flag, target, sb, s, text_end) {
            return Some(s);
        }
        s = prev_char_head(enc, sb, adjust_text, s, text_end);
    }
    None
}

/// Sunday's quick search applied to a multibyte string
fn bm_search_notrev(reg: &Regex, target: &[u8], sb: &[u8], text: Pos, text_end: Pos, text_range: Pos) -> Option<Pos> {
    let enc = reg.enc;
    let tlen1 = target.len() as isize - 1;
    let mut end = text_range;
    if end + tlen1 > text_end {
        end = text_end - tlen1;
    }
    let mut s = text;
    while s < end {
        let se = s + tlen1;
        let mut p = se;
        let mut t = tlen1;
        while sb[p as usize] == target[t as usize] {
            if t == 0 {
                return Some(s);
            }
            p -= 1;
            t -= 1;
        }
        if s + 1 >= end {
            break;
        }
        let skip = reg.map[sb[(se + 1) as usize] as usize] as isize;
        let t0 = s;
        loop {
            s += enc.enclen(sb, s as usize, end as usize) as isize;
            if !((s - t0) < skip && s < end) {
                break;
            }
        }
    }
    None
}

/// Sunday's quick search
fn bm_search(reg: &Regex, target: &[u8], sb: &[u8], text: Pos, text_end: Pos, text_range: Pos) -> Option<Pos> {
    let tlen1 = target.len() as isize - 1;
    let mut end = text_range + tlen1;
    if end > text_end {
        end = text_end;
    }
    let mut s = text + tlen1;
    while s < end {
        let mut p = s;
        let mut t = tlen1;
        while sb[p as usize] == target[t as usize] {
            if t == 0 {
                return Some(p);
            }
            p -= 1;
            t -= 1;
        }
        if s + 1 >= end {
            break;
        }
        s += reg.map[sb[(s + 1) as usize] as usize] as isize;
    }
    None
}

/// Sunday's quick search applied to a multibyte string (ignore case)
fn bm_search_notrev_ic(reg: &Regex, target: &[u8], sb: &[u8], text: Pos, text_end: Pos, text_range: Pos) -> Option<Pos> {
    let enc = reg.enc;
    let tlen1 = target.len() as isize - 1;
    let mut end = text_range;
    if end + tlen1 > text_end {
        end = text_end - tlen1;
    }
    let mut s = text;
    while s < end {
        let se = s + tlen1;
        if str_lower_case_match(enc, reg.case_fold_flag, target, sb, s, se + 1) {
            return Some(s);
        }
        if s + 1 >= end {
            break;
        }
        let skip = reg.map[sb[(se + 1) as usize] as usize] as isize;
        let t0 = s;
        loop {
            s += enc.enclen(sb, s as usize, end as usize) as isize;
            if !((s - t0) < skip && s < end) {
                break;
            }
        }
    }
    None
}

/// Sunday's quick search (ignore case)
fn bm_search_ic(reg: &Regex, target: &[u8], sb: &[u8], text: Pos, text_end: Pos, text_range: Pos) -> Option<Pos> {
    let tlen1 = target.len() as isize - 1;
    let mut end = text_range + tlen1;
    if end > text_end {
        end = text_end;
    }
    let mut s = text + tlen1;
    while s < end {
        let p = s - tlen1;
        if str_lower_case_match(reg.enc, reg.case_fold_flag, target, sb, p, s + 1) {
            return Some(p);
        }
        if s + 1 >= end {
            break;
        }
        s += reg.map[sb[(s + 1) as usize] as usize] as isize;
    }
    None
}

fn map_search(enc: Enc, map: &[u8; ONIG_CHAR_TABLE_SIZE], sb: &[u8], text: Pos, text_range: Pos, text_end: Pos) -> Option<Pos> {
    let mut s = text;
    while s < text_range {
        if map[sb[s as usize] as usize] != 0 {
            return Some(s);
        }
        s += enc.enclen(sb, s as usize, text_end as usize) as isize;
    }
    None
}

fn map_search_backward(
    enc: Enc,
    map: &[u8; ONIG_CHAR_TABLE_SIZE],
    sb: &[u8],
    text: Pos,
    adjust_text: Pos,
    text_start: Pos,
    text_end: Pos,
) -> Option<Pos> {
    let mut s = text_start;
    while s >= text {
        if s < text_end && map[sb[s as usize] as usize] != 0 {
            return Some(s);
        }
        s = prev_char_head(enc, sb, adjust_text, s, text_end);
    }
    None
}

struct SearchRange {
    low: Pos,
    high: Pos,
    low_prev: Pos,
}

fn forward_search_range(reg: &Regex, sb: &[u8], s: Pos, range: Pos, want_low_prev: bool) -> Option<SearchRange> {
    let enc = reg.enc;
    let end = sb.len() as Pos;
    let input_len = end as usize;
    let mut pprev = NULL_POS;
    let mut low_prev = NULL_POS;

    if reg.dmin > input_len {
        return None;
    }
    let mut p = s;
    if reg.dmin != 0 {
        if ((end - p) as usize) <= reg.dmin {
            return None; /* fail */
        }
        if enc.max_len() == 1 {
            p += reg.dmin as isize;
        } else {
            let q = p + reg.dmin as isize;
            while p < q {
                p += enc.enclen(sb, p as usize, end as usize) as isize;
            }
        }
    }

    loop {
        // retry:
        let found = match reg.optimize {
            ONIG_OPTIMIZE_EXACT => slow_search(enc, &reg.exact, sb, p, end, range),
            ONIG_OPTIMIZE_EXACT_IC => slow_search_ic(enc, reg.case_fold_flag, &reg.exact, sb, p, end, range),
            ONIG_OPTIMIZE_EXACT_BM => bm_search(reg, &reg.exact, sb, p, end, range),
            ONIG_OPTIMIZE_EXACT_BM_NOT_REV => bm_search_notrev(reg, &reg.exact, sb, p, end, range),
            ONIG_OPTIMIZE_EXACT_BM_IC => bm_search_ic(reg, &reg.exact, sb, p, end, range),
            ONIG_OPTIMIZE_EXACT_BM_NOT_REV_IC => bm_search_notrev_ic(reg, &reg.exact, sb, p, end, range),
            ONIG_OPTIMIZE_MAP => map_search(enc, &reg.map, sb, p, range, end),
            _ => None,
        };
        let Some(fp) = found else { return None };
        p = fp;
        if p >= range {
            return None;
        }

        let mut retry = (p - s) < reg.dmin as isize;
        if !retry && reg.sub_anchor != 0 {
            match reg.sub_anchor {
                ANCHOR_BEGIN_LINE => {
                    if p != 0 {
                        let prev = prev_char_head(enc, sb, if pprev != NULL_POS { pprev } else { 0 }, p, end);
                        if !is_mbc_newline_ex(enc, sb, prev, 0, end, reg.options, false) {
                            retry = true;
                        }
                    }
                }
                ANCHOR_END_LINE => {
                    if p == end {
                        /* USE_NEWLINE_AT_END_OF_STRING_HAS_EMPTY_LINE */
                    } else if !is_mbc_newline_ex(enc, sb, p, 0, end, reg.options, true) {
                        retry = true;
                    }
                }
                _ => {}
            }
        }
        if retry {
            // retry_gate:
            pprev = p;
            p += enc.enclen(sb, p as usize, end as usize) as isize;
            continue;
        }

        let (low, high);
        if reg.dmax == 0 {
            low = p;
            if want_low_prev {
                low_prev = if low > s {
                    prev_char_head(enc, sb, s, p, end)
                } else {
                    prev_char_head(enc, sb, if pprev != NULL_POS { pprev } else { 0 }, p, end)
                };
            }
            high = p;
        } else {
            if reg.dmax != INF {
                if ((p) as usize) < reg.dmax {
                    low = 0;
                    if want_low_prev {
                        low_prev = prev_char_head(enc, sb, 0, low, end);
                    }
                } else {
                    let mut l = p - reg.dmax as isize;
                    if l > s {
                        let (q, prev) = enc.get_right_adjust_char_head_with_prev(sb, s as usize, l as usize, end as usize);
                        l = q as Pos;
                        if want_low_prev {
                            low_prev = match prev {
                                Some(v) => v as Pos,
                                None => prev_char_head(enc, sb, if pprev != NULL_POS { pprev } else { s }, l, end),
                            };
                        }
                    } else if want_low_prev {
                        low_prev = prev_char_head(enc, sb, if pprev != NULL_POS { pprev } else { 0 }, l, end);
                    }
                    low = l;
                }
            } else {
                low = 0; /* unused: the caller only checks the range */
            }
            /* no needs to adjust *high, *high is used as range check only */
            high = if (p as usize) < reg.dmin { 0 } else { p - reg.dmin as isize };
        }
        return Some(SearchRange { low, high, low_prev });
    }
}

fn backward_search_range(reg: &Regex, sb: &[u8], s: Pos, range: Pos, adjrange: Pos) -> Option<(Pos, Pos)> {
    let enc = reg.enc;
    let end = sb.len() as Pos;
    if reg.dmin > end as usize {
        return None;
    }
    let mut p = s;
    loop {
        // retry:
        let found = match reg.optimize {
            ONIG_OPTIMIZE_EXACT | ONIG_OPTIMIZE_EXACT_BM | ONIG_OPTIMIZE_EXACT_BM_NOT_REV => {
                slow_search_backward(enc, &reg.exact, sb, range, adjrange, end, p)
            }
            ONIG_OPTIMIZE_EXACT_IC | ONIG_OPTIMIZE_EXACT_BM_IC | ONIG_OPTIMIZE_EXACT_BM_NOT_REV_IC => {
                slow_search_backward_ic(enc, reg.case_fold_flag, &reg.exact, sb, range, adjrange, end, p)
            }
            ONIG_OPTIMIZE_MAP => map_search_backward(enc, &reg.map, sb, range, adjrange, p, end),
            _ => None,
        };
        let fp = found?;
        p = fp;

        if reg.sub_anchor != 0 {
            match reg.sub_anchor {
                ANCHOR_BEGIN_LINE => {
                    if p != 0 {
                        let prev = prev_char_head(enc, sb, 0, p, end);
                        if !is_mbc_newline_ex(enc, sb, prev, 0, end, reg.options, false) {
                            p = prev;
                            continue;
                        }
                    }
                }
                ANCHOR_END_LINE => {
                    if p == end {
                        /* USE_NEWLINE_AT_END_OF_STRING_HAS_EMPTY_LINE */
                    } else if !is_mbc_newline_ex(enc, sb, p, 0, end, reg.options, true) {
                        p = prev_char_head(enc, sb, adjrange, p, end);
                        if p == NULL_POS {
                            return None;
                        }
                        continue;
                    }
                }
                _ => {}
            }
        }

        let (mut low, mut high) = (0, 0);
        if reg.dmax != INF {
            low = if (p as usize) < reg.dmax { 0 } else { p - reg.dmax as isize };
            if reg.dmin != 0 {
                high = if (p as usize) < reg.dmin { 0 } else { p - reg.dmin as isize };
            } else {
                high = p;
            }
            high = enc.get_right_adjust_char_head(sb, adjrange as usize, high as usize, end as usize) as Pos;
        }
        return Some((low, high));
    }
}

/// The outcome of a search: a position, ONIG_MISMATCH or an error code
/// (`OnigPosition` of C).
pub type SearchResult = isize;

/// `onig_search_gpos`: positions are offsets into `sb`.
pub fn search(
    reg: &Regex,
    sb: &[u8],
    global_pos: Pos,
    start: Pos,
    range: Pos,
    mut region: Option<Region<'_>>,
    option: u32,
    check: &mut dyn FnMut() -> i32,
) -> SearchResult {
    let enc = reg.enc;
    let str_: Pos = 0;
    let end: Pos = sb.len() as Pos;
    let mut start = start;
    let mut range = range;
    let orig_range = range;
    let _ = orig_range;

    if let Some(r) = region.as_mut() {
        r.clear();
    }
    if start > end || start < str_ {
        return ONIG_MISMATCH as isize;
    }

    /* anchor optimize: resume search range */
    if reg.anchor != 0 && str_ < end {
        let mut min_semi_end = end;
        let mut max_semi_end = end;
        let mut do_end_buf = false;
        if reg.anchor & ANCHOR_BEGIN_POSITION != 0 {
            /* search start-position only */
            (start, range) = begin_position(start, range, global_pos);
        } else if reg.anchor & ANCHOR_BEGIN_BUF != 0 {
            /* search str-position only */
            if range > start {
                if start != str_ {
                    return ONIG_MISMATCH as isize;
                }
                range = str_ + 1;
            } else if range <= str_ {
                start = str_;
                range = str_;
            } else {
                return ONIG_MISMATCH as isize;
            }
        } else if reg.anchor & ANCHOR_END_BUF != 0 {
            do_end_buf = true;
        } else if reg.anchor & ANCHOR_SEMI_END_BUF != 0 {
            let pre_end = enc.step_back(sb, 0, end as usize, end as usize, 1).map(|v| v as Pos).unwrap_or(NULL_POS);
            max_semi_end = end;
            if pre_end != NULL_POS && enc.is_mbc_newline(sb, pre_end as usize, end as usize) {
                min_semi_end = pre_end;
                /* USE_CRNL_AS_LINE_TERMINATOR */
                let pre_end2 =
                    enc.step_back(sb, 0, pre_end as usize, end as usize, 1).map(|v| v as Pos).unwrap_or(NULL_POS);
                if pre_end2 != NULL_POS && is_newline_crlf(reg.options) && {
                    let a = enc.mbc_to_code(sb, pre_end2 as usize, end as usize);
                    let nx = pre_end2 as usize + enc.enclen(sb, pre_end2 as usize, end as usize);
                    a == 13 && enc.mbc_to_code(sb, nx, end as usize) == 10
                } {
                    min_semi_end = pre_end2;
                }
                if min_semi_end > str_ && start <= min_semi_end {
                    do_end_buf = true;
                }
            } else {
                min_semi_end = end;
                do_end_buf = true;
            }
        } else if reg.anchor & ANCHOR_ANYCHAR_STAR_ML != 0 {
            (start, range) = begin_position(start, range, global_pos);
        }

        if do_end_buf {
            // end_buf:
            if ((max_semi_end - str_) as usize) < reg.anchor_dmin {
                return ONIG_MISMATCH as isize;
            }
            if range > start {
                if ((min_semi_end - start) as usize) > reg.anchor_dmax {
                    start = min_semi_end - reg.anchor_dmax as isize;
                    if start < end {
                        start = enc.get_right_adjust_char_head(sb, 0, start as usize, end as usize) as Pos;
                    }
                }
                if ((max_semi_end - (range - 1)) as usize) < reg.anchor_dmin {
                    if ((max_semi_end - str_ + 1) as usize) < reg.anchor_dmin {
                        return ONIG_MISMATCH as isize;
                    } else {
                        range = max_semi_end - reg.anchor_dmin as isize + 1;
                    }
                }
                if start > range {
                    return ONIG_MISMATCH as isize;
                }
                /* If start == range, match with empty at end.
                   Backward search is used. */
            } else {
                if ((min_semi_end - range) as usize) > reg.anchor_dmax {
                    range = min_semi_end - reg.anchor_dmax as isize;
                }
                if ((max_semi_end - start) as usize) < reg.anchor_dmin {
                    if ((max_semi_end - str_) as usize) < reg.anchor_dmin {
                        return ONIG_MISMATCH as isize;
                    } else {
                        start = max_semi_end - reg.anchor_dmin as isize;
                        start = enc.left_adjust_char_head(sb, 0, start as usize, end as usize) as Pos;
                    }
                }
                if range > start {
                    return ONIG_MISMATCH as isize;
                }
            }
        }
    } else if str_ == end {
        /* empty string */
        if reg.threshold_len == 0 {
            let mut msa = MatchArg::new(option, 0, check);
            let r = match try_match_at(reg, sb, 0, NULL_POS, &mut msa, &mut region) {
                Matched::At(p) => p,
                Matched::Err(e) => e,
                Matched::No => mismatch(reg, &msa),
            };
            return finish_result(reg, r, &msa, &mut region);
        }
        return ONIG_MISMATCH as isize;
    }

    let mut msa = MatchArg::new(option, global_pos, check);
    let r = search_body(reg, sb, start, range, &mut msa, &mut region);
    finish_result(reg, r, &msa, &mut region)
}

fn begin_position(start: Pos, range: Pos, global_pos: Pos) -> (Pos, Pos) {
    let mut range = range;
    if range > start {
        if global_pos > start {
            if global_pos < range {
                range = global_pos + 1;
            }
        } else {
            range = start + 1;
        }
    } else {
        range = start;
    }
    (start, range)
}

enum Matched {
    At(Pos),
    No,
    Err(isize),
}

/// The common tail of `onig_search`: `r` is a match_at result, MISMATCH or
/// an error. A match position comes back as `Ok`-like non-negative value.
fn finish_result(reg: &Regex, r: isize, msa: &MatchArg, region: &mut Option<Region<'_>>) -> isize {
    let _ = msa;
    if r >= 0 {
        return r;
    }
    if r != ONIGERR_TIMEOUT as isize && is_find_not_empty(reg.options) {
        if let Some(rg) = region.as_mut() {
            rg.clear();
        }
    }
    r
}

/// The search loops of `onig_search_gpos`. Returns the start of the match,
/// ONIG_MISMATCH or an error.
fn search_body(reg: &Regex, sb: &[u8], start: Pos, range: Pos, msa: &mut MatchArg, region: &mut Option<Region<'_>>) -> isize {
    let enc = reg.enc;
    let end = sb.len() as Pos;
    let mut s = start;
    let mut prev: Pos;

    // MATCH_AND_RETURN_CHECK
    macro_rules! try_match {
        () => {
            match try_match_at(reg, sb, s, prev, msa, region) {
                Matched::No => {}
                Matched::At(p) => return p,
                Matched::Err(e) => return e,
            }
        };
    }

    if range > start {
        /* forward search */
        prev = if s > 0 { prev_char_head(enc, sb, 0, s, end) } else { NULL_POS };

        if reg.optimize != ONIG_OPTIMIZE_NONE {
            let sch_range = if reg.dmax != 0 {
                if reg.dmax == INF {
                    end
                } else if ((end - range) as usize) < reg.dmax {
                    end
                } else {
                    range + reg.dmax as isize
                }
            } else {
                range
            };

            if (end - start) < reg.threshold_len as isize {
                return mismatch(reg, msa);
            }

            if reg.dmax != INF {
                loop {
                    let Some(sr) = forward_search_range(reg, sb, s, sch_range, true) else {
                        return mismatch(reg, msa);
                    };
                    if s < sr.low {
                        s = sr.low;
                        prev = sr.low_prev;
                    }
                    while s <= sr.high {
                        try_match!();
                        prev = s;
                        s += enc.enclen(sb, s as usize, end as usize) as isize;
                    }
                    if s >= range {
                        break;
                    }
                }
                return mismatch(reg, msa);
            } else {
                /* check only. */
                if forward_search_range(reg, sb, s, sch_range, false).is_none() {
                    return mismatch(reg, msa);
                }
                if reg.anchor & ANCHOR_ANYCHAR_STAR != 0 {
                    loop {
                        try_match!();
                        prev = s;
                        s += enc.enclen(sb, s as usize, end as usize) as isize;
                        if reg.anchor & (ANCHOR_LOOK_BEHIND | ANCHOR_PREC_READ_NOT) == 0 {
                            while !is_mbc_newline_ex(enc, sb, prev, 0, end, reg.options, false) && s < range {
                                prev = s;
                                s += enc.enclen(sb, s as usize, end as usize) as isize;
                            }
                        }
                        if s >= range {
                            break;
                        }
                    }
                    return mismatch(reg, msa);
                }
            }
        }

        loop {
            try_match!();
            prev = s;
            s += enc.enclen(sb, s as usize, end as usize) as isize;
            if s >= range {
                break;
            }
        }
        if s == range {
            /* because empty match with /$/. */
            try_match!();
        }
    } else {
        /* backward search */
        if reg.optimize != ONIG_OPTIMIZE_NONE {
            let adjrange =
                if range < end { enc.left_adjust_char_head(sb, 0, range as usize, end as usize) as Pos } else { end };
            let min_range = if ((end - range) as usize) > reg.dmin { range + reg.dmin as isize } else { end };

            if reg.dmax != INF && end - range >= reg.threshold_len as isize {
                loop {
                    let sch_start = if ((end - s) as usize) > reg.dmax { s + reg.dmax as isize } else { end };
                    let Some((low, high)) = backward_search_range(reg, sb, sch_start, min_range, adjrange) else {
                        return mismatch(reg, msa);
                    };
                    if s > high {
                        s = high;
                    }
                    while s >= low {
                        prev = prev_char_head(enc, sb, 0, s, end);
                        try_match!();
                        s = prev;
                    }
                    if s < range {
                        break;
                    }
                }
                return mismatch(reg, msa);
            } else {
                /* check only. */
                if end - range < reg.threshold_len as isize {
                    return mismatch(reg, msa);
                }
                let sch_start = if reg.dmax != 0 {
                    if reg.dmax == INF {
                        end
                    } else if ((end - s) as usize) > reg.dmax {
                        let v = s + reg.dmax as isize;
                        enc.left_adjust_char_head(sb, start as usize, v as usize, end as usize) as Pos
                    } else {
                        end
                    }
                } else {
                    s
                };
                if backward_search_range(reg, sb, sch_start, min_range, adjrange).is_none() {
                    return mismatch(reg, msa);
                }
            }
        }

        loop {
            prev = prev_char_head(enc, sb, 0, s, end);
            try_match!();
            s = prev;
            if s < range {
                break;
            }
        }
    }
    mismatch(reg, msa)
}

fn mismatch(reg: &Regex, msa: &MatchArg) -> isize {
    /* USE_FIND_LONGEST_SEARCH_ALL_OF_RANGE */
    if is_find_longest(reg.options) && msa.best_len >= 0 {
        return msa.best_s;
    }
    ONIG_MISMATCH as isize
}

fn try_match_at(reg: &Regex, sb: &[u8], s: Pos, prev: Pos, msa: &mut MatchArg, region: &mut Option<Region<'_>>) -> Matched {
    let r = match_at(reg, sb, s, prev, msa, region);
    if r == ONIG_MISMATCH as isize {
        Matched::No
    } else if r >= 0 {
        if is_find_longest(reg.options) { Matched::No } else { Matched::At(s) }
    } else {
        Matched::Err(r)
    }
}

/// `onig_match`: match at `at` only. Returns the match length.
pub fn match_at_pos(
    reg: &Regex,
    sb: &[u8],
    at: Pos,
    mut region: Option<Region<'_>>,
    option: u32,
    check: &mut dyn FnMut() -> i32,
) -> isize {
    if let Some(r) = region.as_mut() {
        r.clear();
    }
    let mut msa = MatchArg::new(option, at, check);
    let prev = prev_char_head(reg.enc, sb, 0, at, sb.len() as Pos);
    match_at(reg, sb, at, prev, &mut msa, &mut region)
}
