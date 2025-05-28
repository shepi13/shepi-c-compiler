use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::zip;
use std::sync::atomic::AtomicUsize;

use super::assembly_rewrite::check_overflow;
use crate::parse::parse_tree;
use crate::parse::parse_tree::CType;
use crate::tac_generation::generator::{self, Value, gen_label};
use crate::validate::type_check::{Initializer, SymbolAttr, Symbols};

use Operand::Imm;
use Operand::Memory;
use Operand::Register as Reg;
use Register::*;

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
    pub alignment: i32,
    pub init: Initializer,
}
#[derive(Debug, Clone)]
pub struct StaticConstant {
    pub name: String,
    pub alignment: i32,
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
    MovSignExtend(Operand, Operand),
    MovZeroExtend(Operand, Operand),
    Lea(Operand, Operand),
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
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyType {
    Longword,
    Quadword,
    Double,
}
impl AssemblyType {
    pub fn from(ctype: &CType) -> Self {
        match ctype {
            CType::Int | CType::UnsignedInt => AssemblyType::Longword,
            CType::Long | CType::UnsignedLong => AssemblyType::Quadword,
            CType::Double => AssemblyType::Double,
            CType::Pointer(_) => AssemblyType::Quadword,
            _ => panic!("Expected a variable!"),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Register {
    AX,
    CX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
    CL,
    SP,
    BP,
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM14,
    XMM15,
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

struct StackGen {
    static_constants: Vec<(f64, StaticConstant)>,
    stack_variables: HashMap<String, usize>,
    stack_offset: usize,
}

impl StackGen {
    pub fn new() -> StackGen {
        StackGen {
            static_constants: Vec::new(),
            stack_variables: HashMap::new(),
            stack_offset: 0,
        }
    }
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
    fn gen_static_constant_name() -> String {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        format!(".Lconst_double.{}", COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

const INT_REGISTERS: [Register; 6] = [DI, SI, DX, CX, R8, R9];
const FLOAT_REGISTERS: [Register; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];

pub fn gen_assembly_tree(ast: generator::Program, symbols: Symbols) -> Program {
    let mut program = Vec::new();
    let mut stack = StackGen::new();
    for decl in ast.into_iter() {
        match decl {
            generator::TopLevelDecl::Function(function) => {
                if function.instructions.is_empty() {
                    continue;
                }
                // Setup initial state (including nop that will be replaced by stack allocation later)
                let mut instructions: Vec<Instruction> = Vec::new();
                instructions.push(Instruction::Nop);
                stack.reset_stack();
                // Pull arguments from registers/stack
                let mut param_setup = set_up_parameters(function.params, &mut stack, &symbols);
                instructions.append(&mut param_setup);
                // Generate instructions for function body
                let mut body = gen_instructions(function.instructions, &mut stack, &symbols);
                instructions.append(&mut body);
                // Replace Nop with stack allocation, and add instructions to program
                instructions[0] = Instruction::Binary(
                    BinaryOperator::Sub,
                    Imm((stack.stack_offset + 16 - stack.stack_offset % 16) as i128),
                    Reg(SP),
                    AssemblyType::Quadword,
                );
                program.push(TopLevelDecl::FunctionDecl(Function {
                    name: function.name,
                    global: function.global,
                    instructions,
                }));
            }
            generator::TopLevelDecl::StaticDecl(static_data) => {
                let alignment = static_data.ctype.size() as i32;
                program.push(TopLevelDecl::Var(StaticVar {
                    name: static_data.identifier,
                    global: static_data.global,
                    alignment,
                    init: static_data.initializer,
                }));
            }
        }
    }
    program.append(
        &mut stack.static_constants.into_iter().map(|var| TopLevelDecl::Constant(var.1)).collect(),
    );
    Program {
        program,
        backend_symbols: gen_backend_symbols(symbols),
    }
}

fn set_up_parameters(
    params: Vec<generator::Value>,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    let param_groups = classify_parameters(params, stack, symbols);
    //Copy params from registers
    let mut copy_params = |param_group, registers: &[Register]| {
        for ((asm_type, param), reg) in zip(param_group, registers) {
            instructions.push(Instruction::Mov(Reg(*reg), param, asm_type));
        }
    };
    copy_params(param_groups.int_args, &INT_REGISTERS);
    copy_params(param_groups.float_args, &FLOAT_REGISTERS);
    // Copy remaining params from stack
    let mut offset = 16;
    for (asm_type, param) in param_groups.stack_args {
        instructions.push(Instruction::Mov(Operand::Memory(BP, offset), param, asm_type));
        offset += 8;
    }
    instructions
}

fn gen_backend_symbols(symbols: Symbols) -> HashMap<String, AsmSymbol> {
    let mut backend_symbols = HashMap::new();
    for (name, symbol) in symbols {
        match &symbol.attrs {
            SymbolAttr::Function(func_attrs) => {
                backend_symbols.insert(name, AsmSymbol::FunctionEntry(func_attrs.defined));
            }
            _ => {
                let asm_type = AssemblyType::from(&symbol.ctype);
                let is_static = symbol.attrs != SymbolAttr::Local;
                backend_symbols.insert(name, AsmSymbol::ObjectEntry(asm_type, is_static));
            }
        }
    }
    backend_symbols
}

fn gen_instructions(
    instructions: Vec<generator::Instruction>,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> Vec<Instruction> {
    let mut asm_instructions: Vec<Instruction> = Vec::new();
    for instruction in instructions {
        use Instruction::*;
        match instruction {
            generator::Instruction::Return(val) => {
                let val_type = AssemblyType::from(&get_type(&val, symbols));
                let val = gen_operand(val, stack, symbols);
                let reg = if val_type == AssemblyType::Double { Reg(XMM0) } else { Reg(AX) };
                asm_instructions.push(Mov(val, reg, val_type));
                asm_instructions.push(Ret);
            }
            generator::Instruction::UnaryOp(val) => {
                use BinaryOperator::BitXor;
                let src_type = AssemblyType::from(&get_type(&val.src, symbols));
                let dst_type = AssemblyType::from(&get_type(&val.dst, symbols));
                let src = gen_operand(val.src, stack, symbols);
                let dst = gen_operand(val.dst, stack, symbols);
                let operator = match val.operator {
                    generator::UnaryOperator::Complement => UnaryOperator::Not,
                    generator::UnaryOperator::Negate => {
                        if src_type == AssemblyType::Double {
                            // Doubles use xor with -0.0 for negation
                            let neg_zero = Operand::Data(stack.static_constant(-0.0, true));
                            asm_instructions.push(Mov(src, dst.clone(), AssemblyType::Double));
                            asm_instructions.push(Binary(BitXor, neg_zero, dst, src_type));
                            continue;
                        } else {
                            UnaryOperator::Neg
                        }
                    }
                    generator::UnaryOperator::LogicalNot => {
                        // Logical not uses cmp instead of unary operators
                        if src_type == AssemblyType::Double {
                            let nan_label = gen_label("nan");
                            let end_label = gen_label("end");
                            asm_instructions.push(Binary(BitXor, Reg(XMM0), Reg(XMM0), src_type));
                            asm_instructions.push(Compare(src, Reg(XMM0), src_type));
                            asm_instructions.push(JmpCond(Condition::Parity, nan_label.clone()));
                            asm_instructions.push(Mov(Imm(0), dst.clone(), dst_type.clone()));
                            asm_instructions.push(SetCond(Condition::Equal, dst.clone()));
                            asm_instructions.push(Jmp(end_label.clone()));
                            asm_instructions.push(Label(nan_label));
                            asm_instructions.push(Mov(Imm(0), dst, dst_type));
                            asm_instructions.push(Label(end_label));
                        } else {
                            asm_instructions.push(Compare(Imm(0), src, src_type));
                            asm_instructions.push(Mov(Imm(0), dst.clone(), dst_type));
                            asm_instructions.push(SetCond(Condition::Equal, dst));
                        }
                        continue;
                    }
                };
                asm_instructions.push(Mov(src, dst.clone(), src_type));
                asm_instructions.push(Unary(operator, dst, src_type));
            }
            generator::Instruction::BinaryOp(binop) => {
                gen_binary_op(&mut asm_instructions, binop, stack, symbols);
            }
            generator::Instruction::Jump(target) => {
                asm_instructions.push(Jmp(target));
            }
            generator::Instruction::JumpCond(jump) => {
                use BinaryOperator::BitXor;
                let condition = match jump.jump_type {
                    generator::JumpType::JumpIfZero => Condition::Equal,
                    generator::JumpType::JumpIfNotZero => Condition::NotEqual,
                };
                let cmp_type = AssemblyType::from(&get_type(&jump.condition, symbols));
                let dst = gen_operand(jump.condition, stack, symbols);
                if cmp_type == AssemblyType::Double {
                    let end_label = gen_label("jumpcond_end");
                    asm_instructions.push(Binary(BitXor, Reg(XMM0), Reg(XMM0), cmp_type));
                    asm_instructions.push(Compare(dst, Reg(XMM0), cmp_type));
                    match jump.jump_type {
                        generator::JumpType::JumpIfNotZero => {
                            asm_instructions.push(JmpCond(Condition::Parity, jump.target.clone()))
                        }
                        generator::JumpType::JumpIfZero => {
                            asm_instructions.push(JmpCond(Condition::Parity, end_label.clone()))
                        }
                    }
                    asm_instructions.push(JmpCond(condition, jump.target));
                    asm_instructions.push(Label(end_label));
                } else {
                    asm_instructions.push(Compare(Imm(0), dst, cmp_type));
                    asm_instructions.push(JmpCond(condition, jump.target));
                }
            }
            generator::Instruction::Copy(copy) => {
                let src_type = AssemblyType::from(&get_type(&copy.src, symbols));
                let src = gen_operand(copy.src, stack, symbols);
                let dst = gen_operand(copy.dst, stack, symbols);
                asm_instructions.push(Mov(src, dst, src_type));
            }
            generator::Instruction::Label(target) => {
                asm_instructions.push(Label(target));
            }
            generator::Instruction::Function(name, args, dst) => {
                gen_func_call(&mut asm_instructions, stack, name, args, dst, symbols);
            }
            generator::Instruction::SignExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(MovSignExtend(src, dst));
            }
            generator::Instruction::Truncate(src, dst) => {
                let mut src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                if check_overflow(&src, i32::MAX as i128) {
                    let Imm(val) = src else { panic!("Overflow must be IMM") };
                    src = Imm(val & 0xFFFFFFFF);
                }
                asm_instructions.push(Mov(src, dst, AssemblyType::Longword));
            }
            generator::Instruction::ZeroExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(MovZeroExtend(src, dst));
            }
            generator::Instruction::DoubleToInt(src, dst) => {
                let dst_type = AssemblyType::from(&get_type(&dst, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Cvttsd2si(src, dst, dst_type));
            }
            generator::Instruction::IntToDouble(src, dst) => {
                let src_type = AssemblyType::from(&get_type(&src, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Cvtsi2sd(src, dst, src_type));
            }
            generator::Instruction::DoubleToUInt(src, dst) => match get_type(&dst, symbols) {
                CType::UnsignedInt => {
                    let src = gen_operand(src, stack, symbols);
                    let dst = gen_operand(dst, stack, symbols);
                    asm_instructions.push(Cvttsd2si(src, Reg(AX), AssemblyType::Quadword));
                    asm_instructions.push(Mov(Reg(AX), dst, AssemblyType::Longword));
                }
                CType::UnsignedLong => {
                    gen_double2ull(&mut asm_instructions, src, dst, stack, symbols)
                }
                _ => panic!("Expected an unsigned dst"),
            },
            generator::Instruction::UIntToDouble(src, dst) => match get_type(&src, symbols) {
                CType::UnsignedInt => {
                    let src = gen_operand(src, stack, symbols);
                    let dst = gen_operand(dst, stack, symbols);
                    asm_instructions.push(MovZeroExtend(src, Reg(AX)));
                    asm_instructions.push(Cvtsi2sd(Reg(AX), dst, AssemblyType::Quadword));
                }
                CType::UnsignedLong => {
                    gen_ull2double(&mut asm_instructions, src, dst, stack, symbols)
                }
                _ => panic!("Expected an unsigned src!"),
            },
            generator::Instruction::GetAddress(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Lea(src, dst));
            }
            generator::Instruction::Load(ptr, dst) => {
                let dst_type = AssemblyType::from(&get_type(&dst, symbols));
                let ptr = gen_operand(ptr, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Mov(ptr, Reg(AX), AssemblyType::Quadword));
                asm_instructions.push(Mov(Memory(AX, 0), dst, dst_type));
            }
            generator::Instruction::Store(src, ptr) => {
                let src_type = AssemblyType::from(&get_type(&src, symbols));
                let src = gen_operand(src, stack, symbols);
                let ptr = gen_operand(ptr, stack, symbols);
                asm_instructions.push(Mov(ptr, Reg(AX), AssemblyType::Quadword));
                asm_instructions.push(Mov(src, Memory(AX, 0), src_type));
            }
        }
    }
    asm_instructions
}

fn gen_double2ull(
    instructions: &mut Vec<Instruction>,
    src: Value,
    dst: Value,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    use BinaryOperator::{Add, Sub};
    use Instruction::*;
    use Operand::Data;
    let src = gen_operand(src, stack, symbols);
    let dst = gen_operand(dst, stack, symbols);
    // Long max + 1
    let upper_bound = stack.static_constant(9223372036854775808.0, false);

    let out_of_range_lbl = gen_label("out_of_range");
    let end_lbl = gen_label("end");
    // If it fits in signed long, conversion is trivial
    instructions.push(Compare(Data(upper_bound.clone()), src.clone(), AssemblyType::Double));
    instructions.push(JmpCond(Condition::UnsignedGreaterEqual, out_of_range_lbl.clone()));
    instructions.push(Cvttsd2si(src.clone(), Reg(AX), AssemblyType::Quadword));
    instructions.push(Jmp(end_lbl.clone()));

    // Otherwise subtract long_max+1, convert, and add it again
    instructions.push(Label(out_of_range_lbl));
    instructions.push(Mov(src, Reg(XMM1), AssemblyType::Double));
    instructions.push(Binary(Sub, Data(upper_bound), Reg(XMM1), AssemblyType::Double));
    instructions.push(Cvttsd2si(Reg(XMM1), Reg(AX), AssemblyType::Quadword));
    instructions.push(Mov(Imm(9223372036854775808), Reg(DX), AssemblyType::Quadword));
    instructions.push(Binary(Add, Reg(DX), Reg(AX), AssemblyType::Quadword));

    // Result is in AX
    instructions.push(Label(end_lbl));
    instructions.push(Mov(Reg(AX), dst, AssemblyType::Quadword));
}

fn gen_ull2double(
    instructions: &mut Vec<Instruction>,
    src: Value,
    dst: Value,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    use Instruction::*;
    let src = gen_operand(src, stack, symbols);
    let dst = gen_operand(dst, stack, symbols);

    let out_of_range_lbl = gen_label("out_of_range");
    let end_lbl = gen_label("end");
    // Check if it fits in a signed long, if so conversion is trivial with Cvtsi2sd
    instructions.push(Compare(Imm(0), src.clone(), AssemblyType::Quadword));
    instructions.push(JmpCond(Condition::LessThan, out_of_range_lbl.clone()));
    instructions.push(Cvtsi2sd(src.clone(), Reg(XMM0), AssemblyType::Quadword));
    instructions.push(Jmp(end_lbl.clone()));

    // Otherwise Divide by 2 and round to odd by using bitwise and/or
    instructions.push(Label(out_of_range_lbl));
    instructions.push(Mov(src, Reg(AX), AssemblyType::Quadword));
    instructions.push(Mov(Reg(AX), Reg(DX), AssemblyType::Quadword));
    instructions.push(Unary(UnaryOperator::Shr, Reg(DX), AssemblyType::Quadword));
    instructions.push(Binary(BinaryOperator::BitAnd, Imm(1), Reg(AX), AssemblyType::Quadword));
    instructions.push(Binary(BinaryOperator::BitOr, Reg(AX), Reg(DX), AssemblyType::Quadword));
    // Convert divided value and multiply by 2
    instructions.push(Cvtsi2sd(Reg(DX), Reg(XMM0), AssemblyType::Quadword));
    instructions.push(Binary(BinaryOperator::Add, Reg(XMM0), Reg(XMM0), AssemblyType::Double));

    // Result is in XMM0
    instructions.push(Label(end_lbl));
    instructions.push(Mov(Reg(XMM0), dst, AssemblyType::Double));
}

struct ParamGroups {
    int_args: Vec<(AssemblyType, Operand)>,
    float_args: Vec<(AssemblyType, Operand)>,
    stack_args: Vec<(AssemblyType, Operand)>,
}
fn classify_parameters(
    parameters: Vec<Value>,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> ParamGroups {
    let mut int_args = Vec::new();
    let mut float_args = Vec::new();
    let mut stack_args = Vec::new();
    for param in parameters {
        let param_type = AssemblyType::from(&get_type(&param, symbols));
        let operand = gen_operand(param, stack, symbols);
        if param_type == AssemblyType::Double && float_args.len() < 8 {
            float_args.push((param_type, operand));
        } else if param_type != AssemblyType::Double && int_args.len() < 6 {
            int_args.push((param_type, operand));
        } else {
            stack_args.push((param_type, operand))
        }
    }
    ParamGroups { int_args, float_args, stack_args }
}

fn gen_func_call(
    instructions: &mut Vec<Instruction>,
    stack: &mut StackGen,
    name: String,
    args: Vec<generator::Value>,
    dst: generator::Value,
    symbols: &Symbols,
) {
    use AssemblyType::*;
    use Instruction::*;
    let param_groups = classify_parameters(args, stack, symbols);
    // Calculate and adjust stack alignment
    let padding = if param_groups.stack_args.len() % 2 == 1 { 8 } else { 0 };
    let bytes_to_remove = 8 * param_groups.stack_args.len() as i128 + padding;
    if padding != 0 {
        instructions.push(Binary(BinaryOperator::Sub, Imm(padding), Reg(SP), Quadword));
    }
    // Pass arguments in registers
    let mut pass_args = |param_group, registers: &[Register]| {
        for ((arg_type, arg), reg) in zip(param_group, registers) {
            instructions.push(Mov(arg, Reg(*reg), arg_type));
        }
    };
    pass_args(param_groups.int_args, &INT_REGISTERS);
    pass_args(param_groups.float_args, &FLOAT_REGISTERS);
    // Pass arguments on stack
    for (arg_type, arg) in param_groups.stack_args.into_iter().rev() {
        if matches!(arg, Reg(_) | Imm(_)) || matches!(arg_type, Double | Quadword) {
            instructions.push(Push(arg));
        } else {
            instructions.push(Mov(arg, Reg(AX), arg_type));
            instructions.push(Push(Reg(AX)));
        }
    }
    // Call function
    instructions.push(Call(name));
    // Clean up stack
    if bytes_to_remove != 0 {
        instructions.push(Binary(BinaryOperator::Add, Imm(bytes_to_remove), Reg(SP), Quadword));
    }
    // Get return value from eax or xmm0
    let dst_type = AssemblyType::from(&get_type(&dst, symbols));
    let dst = gen_operand(dst, stack, symbols);
    let result_reg = if dst_type == AssemblyType::Double { Reg(XMM0) } else { Reg(AX) };
    instructions.push(Instruction::Mov(result_reg, dst, dst_type));
}

fn gen_binary_op(
    instructions: &mut Vec<Instruction>,
    binary_instruction: generator::InstructionBinary,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    use parse_tree::BinaryOperator::*;
    let src_ctype = get_type(&binary_instruction.src1, symbols);
    let asm_type = AssemblyType::from(&src_ctype);
    let src1 = gen_operand(binary_instruction.src1, stack, symbols);
    let src2 = gen_operand(binary_instruction.src2, stack, symbols);
    let dst = gen_operand(binary_instruction.dst, stack, symbols);
    // Instructions for generic binary ops (addition, subtraction, mult, bitwise)
    let mut gen_arithmetic = |operator| {
        instructions.push(Instruction::Mov(src1.clone(), dst.clone(), asm_type));
        instructions.push(Instruction::Binary(operator, src2.clone(), dst.clone(), asm_type));
    };
    match &binary_instruction.operator {
        // Handle arithmetic binary operators
        Add => gen_arithmetic(BinaryOperator::Add),
        Multiply => gen_arithmetic(BinaryOperator::Mult),
        Subtract => gen_arithmetic(BinaryOperator::Sub),
        BitAnd => gen_arithmetic(BinaryOperator::BitAnd),
        BitOr => gen_arithmetic(BinaryOperator::BitOr),
        BitXor => gen_arithmetic(BinaryOperator::BitXor),
        LeftShift => {
            let operator = match src_ctype.is_signed() {
                true => BinaryOperator::LeftShift,
                false => BinaryOperator::LeftShiftUnsigned,
            };
            gen_shift(instructions, operator, src1, src2, dst, asm_type);
        }
        RightShift => {
            let operator = match src_ctype.is_signed() {
                true => BinaryOperator::RightShift,
                false => BinaryOperator::RightShiftUnsigned,
            };
            gen_shift(instructions, operator, src1, src2, dst, asm_type);
        }
        // Division is handled separately
        Divide => {
            if asm_type == AssemblyType::Double {
                gen_arithmetic(BinaryOperator::DoubleDiv);
            } else {
                let sign = src_ctype.is_signed();
                gen_division(instructions, src1, src2, dst, Register::AX, asm_type, sign);
            }
        }
        Remainder => {
            let sign = src_ctype.is_signed();
            gen_division(instructions, src1, src2, dst, Register::DX, asm_type, sign);
        }
        GreaterThan | GreaterThanEqual | IsEqual | LessThan | LessThanEqual | LogicalAnd
        | LogicalOr | NotEqual => {
            use AssemblyType::Longword;
            let signed =
                if asm_type == AssemblyType::Double { false } else { src_ctype.is_arithmetic() && src_ctype.is_signed() };
            // NAN returns true for !=, false otherwise
            let nan_result = (binary_instruction.operator == NotEqual) as i128;
            let condition = get_condition(binary_instruction.operator, signed);
            let nan_label = gen_label("is_nan");
            let end_label = gen_label("end");
            instructions.push(Instruction::Compare(src2, src1, asm_type));
            // Double skips the set and defaults to false if either value is NaN
            if asm_type == AssemblyType::Double {
                instructions.push(Instruction::JmpCond(Condition::Parity, nan_label.clone()));
            }
            instructions.push(Instruction::Mov(Imm(0), dst.clone(), Longword));
            instructions.push(Instruction::SetCond(condition, dst.clone()));
            if asm_type == AssemblyType::Double {
                instructions.push(Instruction::Jmp(end_label.clone()));
                instructions.push(Instruction::Label(nan_label));
                instructions.push(Instruction::Mov(Imm(nan_result), dst, Longword));
                instructions.push(Instruction::Label(end_label));
            }
        }
    };
}

fn get_condition(op: parse_tree::BinaryOperator, signed: bool) -> Condition {
    match signed {
        true => match op {
            parse_tree::BinaryOperator::GreaterThan => Condition::GreaterThan,
            parse_tree::BinaryOperator::GreaterThanEqual => Condition::GreaterThanEqual,
            parse_tree::BinaryOperator::LessThan => Condition::LessThan,
            parse_tree::BinaryOperator::LessThanEqual => Condition::LessThanEqual,
            parse_tree::BinaryOperator::NotEqual => Condition::NotEqual,
            parse_tree::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected relational operator!"),
        },
        false => match op {
            parse_tree::BinaryOperator::GreaterThan => Condition::UnsignedGreaterThan,
            parse_tree::BinaryOperator::GreaterThanEqual => Condition::UnsignedGreaterEqual,
            parse_tree::BinaryOperator::LessThan => Condition::UnsignedLessThan,
            parse_tree::BinaryOperator::LessThanEqual => Condition::UnsignedLessEqual,
            parse_tree::BinaryOperator::NotEqual => Condition::NotEqual,
            parse_tree::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected releational operator"),
        },
    }
}

fn gen_shift(
    instructions: &mut Vec<Instruction>,
    operator: BinaryOperator,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    shift_type: AssemblyType,
) {
    instructions.push(Instruction::Mov(src1, Reg(AX), shift_type));
    instructions.push(Instruction::Mov(src2, Reg(CX), shift_type));
    instructions.push(Instruction::Binary(operator, Reg(CL), Reg(AX), shift_type));
    instructions.push(Instruction::Mov(Reg(AX), dst, shift_type));
}

fn gen_division(
    instructions: &mut Vec<Instruction>,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    result_reg: Register,
    div_type: AssemblyType,
    is_signed: bool,
) {
    // Move dividend into AX
    instructions.push(Instruction::Mov(src1, Reg(AX), div_type));

    // USE CDQ or 0 extend to setup registers for division
    // Use IDiv for signed and Div for unsigned
    if is_signed {
        instructions.push(Instruction::Cdq(div_type));
        instructions.push(Instruction::IDiv(src2, div_type));
    } else {
        instructions.push(Instruction::Mov(Imm(0), Reg(DX), div_type));
        instructions.push(Instruction::Div(src2, div_type));
    }
    // Division result is in AX, Remainder in DX
    instructions.push(Instruction::Mov(Reg(result_reg), dst, div_type));
}

fn get_type(value: &generator::Value, symbols: &Symbols) -> CType {
    match &value {
        generator::Value::Variable(name) => symbols[name].ctype.clone(),
        generator::Value::ConstValue(constexpr) => match constexpr {
            parse_tree::Constant::Int(_) => CType::Int,
            parse_tree::Constant::UnsignedInt(_) => CType::UnsignedInt,
            parse_tree::Constant::Long(_) => CType::Long,
            parse_tree::Constant::UnsignedLong(_) => CType::UnsignedLong,
            parse_tree::Constant::Double(_) => CType::Double,
        },
    }
}

fn gen_operand(value: generator::Value, stack: &mut StackGen, symbols: &Symbols) -> Operand {
    match value {
        generator::Value::ConstValue(constexpr) => match constexpr {
            parse_tree::Constant::Int(val) | parse_tree::Constant::Long(val) => {
                Operand::Imm(val.into())
            }
            parse_tree::Constant::UnsignedInt(val) | parse_tree::Constant::UnsignedLong(val) => {
                Operand::Imm(val.into())
            }
            parse_tree::Constant::Double(val) => {
                let name = stack.static_constant(val, false);
                Operand::Data(name)
            }
        },
        generator::Value::Variable(name) => {
            let symbol = &symbols[&name];
            let var_type = AssemblyType::from(&symbol.ctype);
            if matches!(symbol.attrs, SymbolAttr::Static(_)) {
                Operand::Data(name)
            } else if let Some(location) = stack.stack_variables.get(&name) {
                Operand::Memory(BP, -(*location as isize))
            } else {
                match var_type {
                    AssemblyType::Longword => stack.stack_offset += 4,
                    AssemblyType::Quadword | AssemblyType::Double => {
                        stack.stack_offset += 8 + (8 - stack.stack_offset % 8);
                    }
                }
                stack.stack_variables.insert(name.to_string(), stack.stack_offset);
                Operand::Memory(BP, -(stack.stack_offset as isize))
            }
        }
    }
}
