use std::collections::HashMap;

use crate::assembly;
use crate::assembly_rewrite::check_overflow;
use crate::generator;
use crate::parser;
use crate::parser::CType;
use crate::type_check::Initializer;
use crate::type_check::{SymbolAttr, Symbols};

use Operand::IMM;
use Operand::Register as Reg;
use assembly::Register::*;

pub type BackendSymbols = HashMap<String, AsmSymbol>;
#[derive(Debug)]
pub struct Program {
    pub program: Vec<TopLevelDecl>,
    pub backend_symbols: HashMap<String, AsmSymbol>,
}
#[derive(Debug)]
pub enum TopLevelDecl {
    FUNCTION(Function),
    STATICVAR(StaticVar),
}
#[derive(Debug)]
pub enum AsmSymbol {
    ObjectEntry(AssemblyType, bool),
    FunctionEntry(bool),
}
#[derive(Debug)]
pub struct StaticVar {
    pub name: String,
    pub global: bool,
    pub alignment: i32,
    pub init: Initializer,
}
#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
    pub global: bool,
}
// Instructions
#[derive(Debug)]
pub enum Instruction {
    Mov(Operand, Operand, AssemblyType),
    MovSignExtend(Operand, Operand),
    MovZeroExtend(Operand, Operand),
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
    NOP,
}
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Not,
    Neg,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Mult,
    Sub,
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
    LeftShiftUnsigned,
    RightShiftUnsigned,
}
// Operands
#[derive(Debug, Clone)]
pub enum Operand {
    IMM(i128),
    Stack(isize),
    Register(Register),
    Data(String),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblyType {
    Longword,
    Quadword,
}
impl AssemblyType {
    pub fn from(ctype: &CType) -> Self {
        match ctype {
            CType::Int | CType::UnsignedInt => AssemblyType::Longword,
            CType::Long | CType::UnsignedLong => AssemblyType::Quadword,
            _ => panic!("Expected a variable!"),
        }
    }
}
#[derive(Debug, Clone)]
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
}
#[derive(Debug, Clone)]
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
}

struct StackGen {
    variables: HashMap<String, usize>,
    stack_offset: usize,
}

impl StackGen {
    pub fn new() -> StackGen {
        StackGen {
            variables: HashMap::new(),
            stack_offset: 0,
        }
    }
}

const ARG_REGISTERS: [Register; 6] = [DI, SI, DX, CX, R8, R9];

pub fn gen_assembly_tree(ast: generator::Program, symbols: Symbols) -> Program {
    let mut program = Vec::new();
    for decl in ast.into_iter() {
        match decl {
            generator::TopLevelDecl::Function(function) => {
                if function.instructions.is_empty() {
                    continue;
                }
                // Setup initial state (including nop that will be replaced by stack allocation later)
                let stack = &mut StackGen::new();
                let mut instructions: Vec<Instruction> = Vec::new();
                instructions.push(Instruction::NOP);
                // Pull arguments from registers/stack
                for (i, param) in function.params.iter().enumerate() {
                    let param_type = AssemblyType::from(&symbols[param].ctype);
                    let param_val = generator::Value::Variable(param.to_string());
                    let param = gen_operand(param_val, stack, &symbols);
                    let register = ARG_REGISTERS.get(i).map(|reg| Reg(reg.clone()));
                    match register {
                        Some(reg) => instructions.push(Instruction::Mov(reg, param, param_type)),
                        None => instructions.push(Instruction::Mov(
                            Operand::Stack((4 - i as isize) * 8),
                            param,
                            param_type,
                        )),
                    }
                }
                // Generate instructions for function body
                let mut body = gen_instructions(function.instructions, stack, &symbols);
                instructions.append(&mut body);
                // Replace Nop with stack allocation, and add instructions to program
                instructions[0] = Instruction::Binary(
                    BinaryOperator::Sub,
                    IMM((stack.stack_offset + 16 - stack.stack_offset % 16) as i128),
                    Reg(SP),
                    AssemblyType::Quadword,
                );
                program.push(TopLevelDecl::FUNCTION(Function {
                    name: function.name,
                    global: function.global,
                    instructions,
                }));
            }
            generator::TopLevelDecl::StaticDecl(static_data) => {
                let alignment = static_data.ctype.size() as i32;
                program.push(TopLevelDecl::STATICVAR(StaticVar {
                    name: static_data.identifier,
                    global: static_data.global,
                    alignment,
                    init: static_data.initializer,
                }));
            }
        }
    }
    Program {
        program,
        backend_symbols: gen_backend_symbols(symbols),
    }
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
        match instruction {
            generator::Instruction::Return(val) => {
                let val_type = AssemblyType::from(&get_type(&val, symbols));
                let val = gen_operand(val, stack, symbols);
                asm_instructions.push(Instruction::Mov(val, Reg(AX), val_type));
                asm_instructions.push(Instruction::Ret);
            }
            generator::Instruction::UnaryOp(val) => {
                let src_type = AssemblyType::from(&get_type(&val.src, symbols));
                let src = gen_operand(val.src, stack, symbols);
                let dst = gen_operand(val.dst, stack, symbols);
                let operator = match val.operator {
                    generator::UnaryOperator::Complement => UnaryOperator::Not,
                    generator::UnaryOperator::Negate => UnaryOperator::Neg,
                    generator::UnaryOperator::LogicalNot => {
                        // Logical not uses cmp instead of unary operators
                        asm_instructions.push(Instruction::Compare(IMM(0), src, src_type.clone()));
                        asm_instructions.push(Instruction::Mov(IMM(0), dst.clone(), src_type));
                        asm_instructions.push(Instruction::SetCond(Condition::Equal, dst));
                        continue;
                    }
                };
                asm_instructions.push(Instruction::Mov(src, dst.clone(), src_type.clone()));
                asm_instructions.push(Instruction::Unary(operator, dst, src_type));
            }
            generator::Instruction::BinaryOp(binop) => {
                gen_binary_op(&mut asm_instructions, binop, stack, symbols);
            }
            generator::Instruction::Jump(target) => {
                asm_instructions.push(Instruction::Jmp(target));
            }
            generator::Instruction::JumpCond(jump) => {
                let condition = match jump.jump_type {
                    generator::JumpType::JumpIfZero => Condition::Equal,
                    generator::JumpType::JumpIfNotZero => Condition::NotEqual,
                };
                let cmp_type = AssemblyType::from(&get_type(&jump.condition, symbols));
                let dst = gen_operand(jump.condition, stack, symbols);
                asm_instructions.push(Instruction::Compare(IMM(0), dst, cmp_type));
                asm_instructions.push(Instruction::JmpCond(condition, jump.target));
            }
            generator::Instruction::Copy(copy) => {
                let src_type = AssemblyType::from(&get_type(&copy.src, symbols));
                let src = gen_operand(copy.src, stack, symbols);
                let dst = gen_operand(copy.dst, stack, symbols);
                asm_instructions.push(Instruction::Mov(src, dst, src_type));
            }
            generator::Instruction::Label(target) => {
                asm_instructions.push(Instruction::Label(target));
            }
            generator::Instruction::Function(name, args, dst) => {
                gen_func_call(&mut asm_instructions, stack, name, args, dst, symbols);
            }
            generator::Instruction::SignExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Instruction::MovSignExtend(src, dst));
            }
            generator::Instruction::Truncate(src, dst) => {
                let mut src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                if check_overflow(&src, i32::MAX as i128) {
                    let IMM(val) = src else {
                        panic!("Overflow must be IMM")
                    };
                    src = IMM(val & 0xFFFFFFFF);
                }
                asm_instructions.push(Instruction::Mov(src, dst, AssemblyType::Longword));
            }
            generator::Instruction::ZeroExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Instruction::MovZeroExtend(src, dst));
            }
        }
    }
    asm_instructions
}

fn gen_func_call(
    instructions: &mut Vec<Instruction>,
    stack: &mut StackGen,
    name: String,
    args: Vec<generator::Value>,
    dst: generator::Value,
    symbols: &Symbols,
) {
    let mut stack_padding = 0;
    // Pad stack to multiple of 8 if necessary
    if args.len() > 6 && args.len() % 2 != 0 {
        stack_padding = 8;
        instructions.push(Instruction::Binary(
            BinaryOperator::Sub,
            IMM(stack_padding),
            Reg(SP),
            AssemblyType::Quadword,
        ));
    }
    // Put first 6 arguments in arg_registers
    for (arg, register) in args.get(0..6).unwrap_or(&args).iter().zip(ARG_REGISTERS) {
        let arg_type = AssemblyType::from(&get_type(&arg, symbols));
        let operand = gen_operand(arg.clone(), stack, symbols);
        instructions.push(Instruction::Mov(operand, Reg(register.clone()), arg_type));
    }
    // Remaining arguments go on the stack in reverse order
    for i in (6..args.len()).rev() {
        let op_type = AssemblyType::from(&get_type(&args[i as usize], symbols));
        let operand = gen_operand(args[i as usize].clone(), stack, symbols);
        if matches!(operand, IMM(_) | Reg(_)) || op_type == AssemblyType::Quadword {
            instructions.push(Instruction::Push(operand.clone()));
        } else {
            instructions.push(Instruction::Mov(operand, Reg(AX), AssemblyType::Longword));
            instructions.push(Instruction::Push(Reg(AX)));
        }
    }
    // Call function
    instructions.push(Instruction::Call(name));
    // Calculate extra bytes reserved for pushed arguments and deallocate
    let mut extra_bytes = stack_padding;
    if args.len() > 6 {
        extra_bytes += 8 * (args.len() as i128 - 6)
    }
    if extra_bytes != 0 {
        instructions.push(Instruction::Binary(
            BinaryOperator::Add,
            IMM(extra_bytes),
            Reg(SP),
            AssemblyType::Quadword,
        ));
    }
    // Get return value from eax
    let dst_type = AssemblyType::from(&get_type(&dst, symbols));
    let dst = gen_operand(dst, stack, symbols);
    instructions.push(Instruction::Mov(Reg(AX), dst, dst_type));
}

fn gen_binary_op(
    instructions: &mut Vec<Instruction>,
    binary_instruction: generator::InstructionBinary,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    use parser::BinaryOperator::*;
    let src_ctype = get_type(&binary_instruction.src1, symbols);
    let asm_type = AssemblyType::from(&src_ctype);
    let src1 = gen_operand(binary_instruction.src1, stack, symbols);
    let src2 = gen_operand(binary_instruction.src2, stack, symbols);
    let dst = gen_operand(binary_instruction.dst, stack, symbols);

    let mut gen_arithmetic = |operator| {
        instructions.push(Instruction::Mov(
            src1.clone(),
            dst.clone(),
            asm_type.clone(),
        ));
        instructions.push(Instruction::Binary(
            operator,
            src2.clone(),
            dst.clone(),
            asm_type.clone(),
        ));
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
            let sign = src_ctype.is_signed();
            gen_division(instructions, src1, src2, dst, Register::AX, asm_type, sign);
        }
        Remainder => {
            let sign = src_ctype.is_signed();
            gen_division(instructions, src1, src2, dst, Register::DX, asm_type, sign);
        }
        GreaterThan | GreaterThanEqual | IsEqual | LessThan | LessThanEqual | LogicalAnd
        | LogicalOr | NotEqual => {
            use AssemblyType::Longword;
            let condition = get_condition(binary_instruction.operator, src_ctype.is_signed());
            instructions.push(Instruction::Compare(src2, src1, asm_type));
            instructions.push(Instruction::Mov(IMM(0), dst.clone(), Longword));
            instructions.push(Instruction::SetCond(condition, dst));
        }
    };
}

fn get_condition(op: parser::BinaryOperator, signed: bool) -> Condition {
    match signed {
        true => match op {
            parser::BinaryOperator::GreaterThan => Condition::GreaterThan,
            parser::BinaryOperator::GreaterThanEqual => Condition::GreaterThanEqual,
            parser::BinaryOperator::LessThan => Condition::LessThan,
            parser::BinaryOperator::LessThanEqual => Condition::LessThanEqual,
            parser::BinaryOperator::NotEqual => Condition::NotEqual,
            parser::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected relational operator!"),
        },
        false => match op {
            parser::BinaryOperator::GreaterThan => Condition::UnsignedGreaterThan,
            parser::BinaryOperator::GreaterThanEqual => Condition::UnsignedGreaterEqual,
            parser::BinaryOperator::LessThan => Condition::UnsignedLessThan,
            parser::BinaryOperator::LessThanEqual => Condition::UnsignedLessEqual,
            parser::BinaryOperator::NotEqual => Condition::NotEqual,
            parser::BinaryOperator::IsEqual => Condition::Equal,
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
    instructions.push(Instruction::Mov(src1, Reg(AX), shift_type.clone()));
    instructions.push(Instruction::Mov(src2, Reg(CX), shift_type.clone()));
    instructions.push(Instruction::Binary(
        operator,
        Reg(CL),
        Reg(AX),
        shift_type.clone(),
    ));
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
    instructions.push(Instruction::Mov(src1, Reg(AX), div_type.clone()));

    // USE CDQ or 0 extend to setup registers for division
    // Use IDiv for signed and Div for unsigned
    if is_signed {
        instructions.push(Instruction::Cdq(div_type.clone()));
        instructions.push(Instruction::IDiv(src2, div_type.clone()));
    } else {
        instructions.push(Instruction::Mov(IMM(0), Reg(DX), div_type.clone()));
        instructions.push(Instruction::Div(src2, div_type.clone()));
    }
    // Division result is in AX, Remainder in DX
    instructions.push(Instruction::Mov(Reg(result_reg), dst, div_type));
}

fn get_type(value: &generator::Value, symbols: &Symbols) -> CType {
    match &value {
        generator::Value::Variable(name) => symbols[name].ctype.clone(),
        generator::Value::ConstValue(constexpr) => match constexpr {
            parser::Constant::Int(_) => CType::Int,
            parser::Constant::UnsignedInt(_) => CType::UnsignedInt,
            parser::Constant::Long(_) => CType::Long,
            parser::Constant::UnsignedLong(_) => CType::UnsignedLong,
            parser::Constant::Double(_) => CType::Double,
        },
    }
}

fn gen_operand(value: generator::Value, stack: &mut StackGen, symbols: &Symbols) -> Operand {
    match value {
        generator::Value::ConstValue(constexpr) => match constexpr {
            parser::Constant::Int(val) | parser::Constant::Long(val) => Operand::IMM(val.into()),
            parser::Constant::UnsignedInt(val) | parser::Constant::UnsignedLong(val) => {
                Operand::IMM(val.into())
            }
            parser::Constant::Double(_) => panic!("Not implemented!"),
        },
        generator::Value::Variable(name) => {
            let symbol = &symbols[&name];
            let var_type = AssemblyType::from(&symbol.ctype);
            if matches!(symbol.attrs, SymbolAttr::Static(_)) {
                Operand::Data(name)
            } else if let Some(location) = stack.variables.get(&name) {
                Operand::Stack(*location as isize)
            } else {
                match var_type {
                    AssemblyType::Longword => stack.stack_offset += 4,
                    AssemblyType::Quadword => {
                        stack.stack_offset += 8 + (8 - stack.stack_offset % 8);
                    }
                }
                stack.variables.insert(name.to_string(), stack.stack_offset);
                Operand::Stack(stack.stack_offset as isize)
            }
        }
    }
}
