use std::collections::HashMap;

use crate::assembly;
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

pub fn gen_assembly_tree(ast: generator::Program, symbols: Symbols) -> Program {
    let mut program = Vec::new();
    for decl in ast.into_iter() {
        match decl {
            generator::TopLevelDecl::Function(function) => {
                if function.instructions.is_empty() {
                    continue;
                }
                let stack = &mut StackGen::new();
                let mut instructions: Vec<Instruction> = Vec::new();
                instructions.push(Instruction::NOP);
                for (i, param) in function.params.iter().enumerate() {
                    let param_type = AssemblyType::from(&symbols[param].ctype);
                    let param = gen_operand(
                        generator::Value::Variable(param.to_string()),
                        stack,
                        &symbols,
                    );
                    let register = match i {
                        0 => Some(Operand::Register(Register::DI)),
                        1 => Some(Operand::Register(Register::SI)),
                        2 => Some(Operand::Register(Register::DX)),
                        3 => Some(Operand::Register(Register::CX)),
                        4 => Some(Operand::Register(Register::R8)),
                        5 => Some(Operand::Register(Register::R9)),
                        _ => None,
                    };
                    match register {
                        Some(reg) => instructions.push(Instruction::Mov(reg, param, param_type)),
                        None => instructions.push(Instruction::Mov(
                            Operand::Stack((4 - i as isize) * 8),
                            param,
                            param_type,
                        )),
                    }
                }
                instructions.append(&mut gen_instructions(
                    function.instructions,
                    stack,
                    &symbols,
                ));
                instructions[0] = Instruction::Binary(
                    BinaryOperator::Sub,
                    Operand::IMM((stack.stack_offset + 16 - stack.stack_offset % 16) as i128),
                    Operand::Register(Register::SP),
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
    let mut assembly_instructions: Vec<Instruction> = Vec::new();
    for instruction in instructions {
        match instruction {
            generator::Instruction::Return(val) => {
                let val_type = AssemblyType::from(&get_type(&val, symbols));
                let val = gen_operand(val, stack, symbols);
                assembly_instructions.push(Instruction::Mov(
                    val,
                    Operand::Register(Register::AX),
                    val_type,
                ));
                assembly_instructions.push(Instruction::Ret);
            }
            generator::Instruction::UnaryOp(val) => {
                let operator = match val.operator {
                    generator::UnaryOperator::Complement => UnaryOperator::Not,
                    generator::UnaryOperator::Negate => UnaryOperator::Neg,
                    generator::UnaryOperator::LogicalNot => {
                        let src_type = AssemblyType::from(&get_type(&val.src, symbols));
                        let src = gen_operand(val.src, stack, symbols);
                        let dst = gen_operand(val.dst, stack, symbols);
                        gen_compare(
                            &mut assembly_instructions,
                            &Operand::IMM(0),
                            &src,
                            src_type.clone(),
                        );
                        assembly_instructions.push(Instruction::Mov(
                            Operand::IMM(0),
                            dst.clone(),
                            src_type,
                        ));
                        assembly_instructions.push(Instruction::SetCond(Condition::Equal, dst));
                        continue;
                    }
                };
                let src_type = AssemblyType::from(&get_type(&val.src, symbols));
                let src = gen_operand(val.src, stack, symbols);
                let dst = gen_operand(val.dst, stack, symbols);
                assembly_instructions.push(Instruction::Mov(src, dst.clone(), src_type.clone()));
                assembly_instructions.push(Instruction::Unary(operator, dst, src_type));
            }
            generator::Instruction::BinaryOp(binop) => {
                gen_binary_op(&mut assembly_instructions, binop, stack, symbols);
            }
            generator::Instruction::Jump(target) => {
                assembly_instructions.push(Instruction::Jmp(target));
            }
            generator::Instruction::JumpCond(jump) => {
                let condition = match jump.jump_type {
                    generator::JumpType::JumpIfZero => Condition::Equal,
                    generator::JumpType::JumpIfNotZero => Condition::NotEqual,
                };
                let cmp_type = AssemblyType::from(&get_type(&jump.condition, symbols));
                let dst = gen_operand(jump.condition, stack, symbols);
                gen_compare(&mut assembly_instructions, &Operand::IMM(0), &dst, cmp_type);
                assembly_instructions.push(Instruction::JmpCond(condition, jump.target));
            }
            generator::Instruction::Copy(copy) => {
                let src_type = AssemblyType::from(&get_type(&copy.src, symbols));
                let src = gen_operand(copy.src, stack, symbols);
                let dst = gen_operand(copy.dst, stack, symbols);
                assembly_instructions.push(Instruction::Mov(src, dst, src_type));
            }
            generator::Instruction::Label(target) => {
                assembly_instructions.push(Instruction::Label(target));
            }
            generator::Instruction::Function(name, args, dst) => {
                gen_func_call(&mut assembly_instructions, stack, name, args, dst, symbols);
            }
            generator::Instruction::SignExtend(src, dst) => {
                let src_type = AssemblyType::from(&get_type(&src, symbols));
                let dst_type = AssemblyType::from(&get_type(&dst, symbols));
                let mut src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                if matches!(src, Operand::IMM(_)) {
                    assembly_instructions.push(Instruction::Mov(
                        src,
                        Operand::Register(Register::R10),
                        src_type,
                    ));
                    src = Operand::Register(Register::R10)
                }

                if matches!(dst, Operand::Data(_) | Operand::Stack(_)) {
                    assembly_instructions.push(Instruction::MovSignExtend(
                        src,
                        Operand::Register(Register::R11),
                    ));
                    assembly_instructions.push(Instruction::Mov(
                        Operand::Register(Register::R11),
                        dst,
                        dst_type,
                    ));
                } else {
                    assembly_instructions.push(Instruction::MovSignExtend(src, dst));
                }
            }
            generator::Instruction::Truncate(src, dst) => {
                let mut src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                if let Operand::IMM(val) = src {
                    if val > i32::MAX as i128 {
                        src = Operand::IMM(val & (0xFFFFFFFF));
                    }
                }
                assembly_instructions.push(Instruction::Mov(src, dst, AssemblyType::Longword));
            }
            generator::Instruction::ZeroExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                match (&src, &dst) {
                    (_, Operand::Register(_)) => {
                        assembly_instructions.push(Instruction::Mov(
                            src,
                            dst,
                            AssemblyType::Longword,
                        ));
                    }
                    _ => {
                        assembly_instructions.push(Instruction::Mov(
                            src,
                            Operand::Register(Register::R11),
                            AssemblyType::Longword,
                        ));
                        assembly_instructions.push(Instruction::Mov(
                            Operand::Register(Register::R11),
                            dst,
                            AssemblyType::Quadword,
                        ));
                    }
                }
            }
        }
    }
    assembly_instructions
}

fn gen_func_call(
    instructions: &mut Vec<Instruction>,
    stack: &mut StackGen,
    name: String,
    args: Vec<generator::Value>,
    dst: generator::Value,
    symbols: &Symbols,
) {
    let arg_registers = [
        Register::DI,
        Register::SI,
        Register::DX,
        Register::CX,
        Register::R8,
        Register::R9,
    ];
    let stack_padding = if args.len() > 6 && args.len() % 2 != 0 {
        8
    } else {
        0
    };
    if stack_padding != 0 {
        instructions.push(Instruction::Binary(
            BinaryOperator::Sub,
            Operand::IMM(stack_padding),
            Operand::Register(Register::SP),
            AssemblyType::Quadword,
        ));
    }
    for (i, arg) in args.iter().enumerate() {
        if i >= 6 {
            break;
        }
        let arg_type = AssemblyType::from(&get_type(&arg, symbols));
        let arg_operand = gen_operand(arg.clone(), stack, symbols);
        instructions.push(Instruction::Mov(
            arg_operand,
            Operand::Register(arg_registers[i].clone()),
            arg_type,
        ));
    }
    for i in (6..args.len()).rev() {
        let op_type = AssemblyType::from(&get_type(&args[i as usize], symbols));
        let operand = gen_operand(args[i as usize].clone(), stack, symbols);
        if matches!(operand, Operand::IMM(_) | Operand::Register(_))
            || op_type == AssemblyType::Quadword
        {
            gen_push(instructions, &operand);
        } else {
            instructions.push(Instruction::Mov(
                operand,
                Operand::Register(Register::AX),
                AssemblyType::Longword,
            ));
            gen_push(instructions, &Operand::Register(Register::AX));
        }
    }
    instructions.push(Instruction::Call(name));
    let extra_bytes = if args.len() > 6 {
        8 * (args.len() as i128 - 6) + stack_padding
    } else {
        stack_padding as i128
    };
    if extra_bytes != 0 {
        instructions.push(Instruction::Binary(
            BinaryOperator::Add,
            Operand::IMM(extra_bytes),
            Operand::Register(Register::SP),
            AssemblyType::Quadword,
        ));
    }
    let dst_type = AssemblyType::from(&get_type(&dst, symbols));
    let dst = gen_operand(dst, stack, symbols);
    instructions.push(Instruction::Mov(
        Operand::Register(Register::AX),
        dst,
        dst_type,
    ));
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
            gen_shift(instructions, operator, src1, src2, dst, asm_type.clone());
        }
        RightShift => {
            let operator = match src_ctype.is_signed() {
                true => BinaryOperator::RightShift,
                false => BinaryOperator::RightShiftUnsigned,
            };
            gen_shift(instructions, operator, src1, src2, dst, asm_type.clone());
        }
        // Division is handled separately
        Divide => {
            gen_division(
                instructions,
                src1,
                src2,
                dst,
                Register::AX,
                asm_type.clone(),
                src_ctype.is_signed(),
            );
        }
        Remainder => {
            gen_division(
                instructions,
                src1,
                src2,
                dst,
                Register::DX,
                asm_type.clone(),
                src_ctype.is_signed(),
            );
        }
        GreaterThan | GreaterThanEqual | IsEqual | LessThan | LessThanEqual | LogicalAnd
        | LogicalOr | NotEqual => {
            gen_relational_op(
                instructions,
                binary_instruction.operator,
                src1,
                src2,
                dst,
                asm_type.clone(),
                src_ctype.is_signed(),
            );
        }
    };
}

fn gen_push(instructions: &mut Vec<Instruction>, operand: &Operand) {
    if let Operand::IMM(val) = operand {
        if *val > i32::MAX as i128 {
            instructions.push(Instruction::Mov(
                operand.clone(),
                Operand::Register(Register::R10),
                AssemblyType::Quadword,
            ));
            instructions.push(Instruction::Push(Operand::Register(Register::R10)));
            return;
        }
    }
    instructions.push(Instruction::Push(operand.clone()));
}

fn gen_compare(
    instructions: &mut Vec<Instruction>,
    mut src: &Operand,
    dst: &Operand,
    cmp_type: AssemblyType,
) {
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let operands_in_mem = matches!(
        (&src, &dst),
        (
            Operand::Stack(_) | Operand::Data(_),
            Operand::Stack(_) | Operand::Data(_)
        )
    );
    let int_overflow = if let Operand::IMM(val) = src {
        *val > i32::MAX as i128
    } else {
        false
    };
    if operands_in_mem || int_overflow {
        instructions.push(Instruction::Mov(
            src.clone(),
            Operand::Register(Register::R10),
            cmp_type.clone(),
        ));
        src = &Operand::Register(Register::R10);
    }

    // Rewrite dst if it's a constant
    if matches!(dst, Operand::IMM(_)) {
        instructions.push(Instruction::Mov(
            dst.clone(),
            Operand::Register(Register::R11),
            cmp_type.clone(),
        ));
        instructions.push(Instruction::Compare(
            src.clone(),
            Operand::Register(Register::R11),
            cmp_type,
        ));
    } else {
        instructions.push(Instruction::Compare(src.clone(), dst.clone(), cmp_type));
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

fn gen_relational_op(
    instructions: &mut Vec<Instruction>,
    operator: parser::BinaryOperator,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    op_type: AssemblyType,
    is_signed: bool,
) {
    let condition = match is_signed {
        true => match operator {
            parser::BinaryOperator::GreaterThan => Condition::GreaterThan,
            parser::BinaryOperator::GreaterThanEqual => Condition::GreaterThanEqual,
            parser::BinaryOperator::LessThan => Condition::LessThan,
            parser::BinaryOperator::LessThanEqual => Condition::LessThanEqual,
            parser::BinaryOperator::NotEqual => Condition::NotEqual,
            parser::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected relational operator!"),
        },
        false => match operator {
            parser::BinaryOperator::GreaterThan => Condition::UnsignedGreaterThan,
            parser::BinaryOperator::GreaterThanEqual => Condition::UnsignedGreaterEqual,
            parser::BinaryOperator::LessThan => Condition::UnsignedLessThan,
            parser::BinaryOperator::LessThanEqual => Condition::UnsignedLessEqual,
            parser::BinaryOperator::NotEqual => Condition::NotEqual,
            parser::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected releational operator"),
        },
    };
    gen_compare(instructions, &src2, &src1, op_type.clone());
    instructions.push(Instruction::Mov(
        Operand::IMM(0),
        dst.clone(),
        AssemblyType::Longword,
    ));
    instructions.push(Instruction::SetCond(condition, dst));
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
                println!("Var: {name} -> {}", stack.stack_offset);
                stack.variables.insert(name.to_string(), stack.stack_offset);
                Operand::Stack(stack.stack_offset as isize)
            }
        }
    }
}
