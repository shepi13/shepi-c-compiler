use Register::*;
use std::{cmp::Ordering, collections::HashMap, sync::atomic::AtomicUsize};

use crate::{
    parse::parse_tree,
    tac_generation::tac_ast,
    validate::ctype::{CType, Initializer, Symbols},
};

pub type BackendSymbols = HashMap<String, AsmSymbol>;
#[derive(Debug, Clone)]
pub struct Program {
    pub program: Vec<TopLevelDecl>,
    pub backend_symbols: HashMap<String, AsmSymbol>,
}
#[derive(Debug, Clone)]
pub enum TopLevelDecl {
    FunctionDecl(Function),
    Var(StaticVar),
    Constant(StaticConstant),
}
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AsmSymbol {
    ObjectEntry(AssemblyType, bool),
    FunctionEntry(bool),
}
#[derive(Debug, Clone)]
pub struct StaticVar {
    pub name: String,
    pub global: bool,
    pub alignment: u64,
    pub init: Vec<Initializer>,
}
#[derive(Debug, Clone)]
pub struct StaticConstant {
    pub name: String,
    pub alignment: u64,
    pub init: Initializer,
    pub neg_zero: bool,
}
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
    pub global: bool,
}
// Instructions
#[derive(Debug, Clone)]
pub enum Instruction {
    Mov(Operand, Operand, AssemblyType),
    Lea(Operand, Operand),
    MovSignExtend(Operand, Operand, AssemblyType, AssemblyType),
    MovZeroExtend(Operand, Operand, AssemblyType, AssemblyType),
    Cvttsd2si(Operand, Operand, AssemblyType),
    Cvtsi2sd(Operand, Operand, AssemblyType),
    Unary(UnaryOperator, Operand, AssemblyType),
    Binary(BinaryOperator, Operand, Operand, AssemblyType),
    Compare(Operand, Operand, AssemblyType),
    IDiv(Operand, AssemblyType),
    Div(Operand, AssemblyType),
    Cdq(AssemblyType),
    Jmp(String),
    JmpCond(Condition, String),
    SetCond(Condition, Operand),
    Label(String),
    Push(Operand),
    Call(String),
    Ret,
    Nop,
}
#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    Not,
    Neg,
    Shr,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Mult,
    Sub,
    DoubleDiv,
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
    LeftShiftUnsigned,
    RightShiftUnsigned,
}
// Operands
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Imm(i128),
    Memory(Register, isize),
    Register(Register),
    Data(String),
    Indexed(Register, Register, u64),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyType {
    Byte,
    Longword,
    Quadword,
    Double,
    ByteArray(u64, usize),
}
impl AssemblyType {
    pub fn get_alignment(&self) -> usize {
        match self {
            Self::Byte => 1,
            Self::Longword => 4,
            Self::Double | Self::Quadword => 8,
            Self::ByteArray(_, alignment) => *alignment,
        }
    }
}
impl From<CType> for AssemblyType {
    fn from(ctype: CType) -> Self {
        match &ctype {
            CType::Char | CType::SignedChar | CType::UnsignedChar => AssemblyType::Byte,
            CType::Int | CType::UnsignedInt => AssemblyType::Longword,
            CType::Long | CType::UnsignedLong => AssemblyType::Quadword,
            CType::Double => AssemblyType::Double,
            CType::Pointer(_) => AssemblyType::Quadword,
            CType::Array(elem_t, _) => {
                let size = ctype.size();
                let alignment = if size < 16 { elem_t.size().next_power_of_two() / 2 } else { 16 };
                let alignment = alignment.max(1);
                assert!([1, 2, 4, 8, 16].contains(&alignment), "Alignment error: {:#?}", alignment);
                AssemblyType::ByteArray(size, alignment as usize)
            }
            CType::Union(_) | CType::Structure(_) | CType::Function(_, _) | CType::VarArgs | CType::Void => {
                panic!("Not a variable!")
            }
        }
    }
}

pub trait IntoAsmType {
    fn get_ctype(&self, symbols: &Symbols) -> CType;
    fn get_asm_type(&self, symbols: &Symbols) -> AssemblyType;
}
impl IntoAsmType for tac_ast::Value {
    fn get_ctype(&self, symbols: &Symbols) -> CType {
        match self {
            tac_ast::Value::Variable(name) => symbols[name].ctype.clone(),
            tac_ast::Value::ConstValue(constexpr) => match constexpr {
                parse_tree::Constant::Int(_) => CType::Int,
                parse_tree::Constant::UInt(_) => CType::UnsignedInt,
                parse_tree::Constant::Long(_) => CType::Long,
                parse_tree::Constant::ULong(_) => CType::UnsignedLong,
                parse_tree::Constant::Double(_) => CType::Double,
                parse_tree::Constant::Char(_) => CType::Char,
                parse_tree::Constant::UChar(_) => CType::UnsignedChar,
            },
        }
    }
    fn get_asm_type(&self, symbols: &Symbols) -> AssemblyType {
        AssemblyType::from(self.get_ctype(symbols))
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Register {
    AX, CX, DX,             // x registers
    DI, SI,                 // i registers
    R8, R9, R10, R11,       // # registers (used for scratch operations)
    CL,                     // Short c reg, used for bitshifts
    SP, BP,                 // Stack and Base Pointers
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,     // FP registers
    XMM14, XMM15,           // FP scratch registers
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    UnsignedGreaterThan,
    UnsignedGreaterEqual,
    UnsignedLessThan,
    UnsignedLessEqual,
    Parity,
}

#[derive(Default)]
pub struct StackGen {
    pub static_constants: Vec<(f64, StaticConstant)>,
    pub stack_variables: HashMap<String, usize>,
    pub stack_offset: usize,
}
impl StackGen {
    pub fn reset_stack(&mut self) {
        self.stack_variables = HashMap::new();
        self.stack_offset = 0;
    }
    pub fn static_constant(&mut self, value: f64, neg_zero: bool) -> String {
        let search_result = self.static_constants.binary_search_by(|probe| {
            if probe.0 == value && probe.1.neg_zero == neg_zero {
                Ordering::Equal
            } else if probe.0 > value || probe.0 == value && probe.1.neg_zero {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        match search_result {
            Ok(location) => self.static_constants[location].1.name.clone(),
            Err(location) => {
                let name = StackGen::gen_static_constant_name();
                let alignment = if neg_zero { 16 } else { 8 };
                let static_constant = StaticConstant {
                    name: name.clone(),
                    alignment,
                    init: Initializer::Double(value),
                    neg_zero,
                };
                self.static_constants.insert(location, (value, static_constant));
                name
            }
        }
    }
    pub fn gen_static_constant_name() -> String {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        format!(".Lconst_double.{}", COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

pub const INT_REGISTERS: [Register; 6] = [DI, SI, DX, CX, R8, R9];
pub const FLOAT_REGISTERS: [Register; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];
