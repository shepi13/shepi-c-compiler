use std::collections::HashMap;

use crate::generator;
use crate::parser;
use crate::parser::CType;
use crate::type_check::StaticInitializer;
use crate::type_check::{SymbolAttr, Symbols};

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
    pub init: StaticInitializer,
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
    MovSX(Operand, Operand),
    Unary(UnaryOperator, Operand, AssemblyType),
    Binary(BinaryOperator, Operand, Operand, AssemblyType),
    Compare(Operand, Operand, AssemblyType),
    IDiv(Operand, AssemblyType),
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
}
// Operands
#[derive(Debug, Clone)]
pub enum Operand {
    IMM(i64),
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
            CType::Int => AssemblyType::Longword,
            CType::Long => AssemblyType::Quadword,
            _ => panic!("Expected a variable!")
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
                    let (param, param_type) = gen_operand(
                        generator::Value::Variable(param.to_string()),
                        stack,
                        &symbols,
                    );
                    let register= match i {
                        0 => Some(Operand::Register(Register::DI)),
                        1 => Some(Operand::Register(Register::SI)),
                        2 => Some(Operand::Register(Register::DX)),
                        3 => Some(Operand::Register(Register::CX)),
                        4 => Some(Operand::Register(Register::R8)),
                        5 => Some(Operand::Register(Register::R9)),
                        _ => None,
                    };
                    match register {
                        Some(reg) => gen_move(&mut instructions, &reg, &param, param_type),
                        None => gen_move(&mut instructions, &Operand::Stack((4 - i as isize) * 8), &param, param_type),
                    }
                    
                }
                instructions.append(&mut gen_instructions(function.instructions, stack, &symbols));
                instructions[0] = Instruction::Binary(
                    BinaryOperator::Sub,
                    Operand::IMM((stack.stack_offset + 16 - stack.stack_offset % 16) as i64),
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
                let alignment = match &static_data.ctype {
                    CType::Int => 4,
                    CType::Long => 8,
                    _ => panic!("Not a variable!"),
                };
                program.push(TopLevelDecl::STATICVAR(StaticVar {
                    name: static_data.identifier,
                    global: static_data.global,
                    alignment,
                    init: static_data.initializer,
                }));
            }
        }
    }
    Program { program, backend_symbols: gen_backend_symbols(symbols) }
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
                let (val, val_type) = gen_operand(val, stack, symbols);
                gen_move(
                    &mut assembly_instructions,
                    &val,
                    &Operand::Register(Register::AX),
                    val_type,
                );
                assembly_instructions.push(Instruction::Ret);
            }
            generator::Instruction::UnaryOp(val) => {
                let operator = match val.operator {
                    generator::UnaryOperator::Complement => UnaryOperator::Not,
                    generator::UnaryOperator::Negate => UnaryOperator::Neg,
                    generator::UnaryOperator::LogicalNot => {
                        let (src, src_type) = gen_operand(val.src, stack, symbols);
                        let (dst, _) = gen_operand(val.dst, stack, symbols);
                        gen_compare(
                            &mut assembly_instructions,
                            &Operand::IMM(0),
                            &src,
                            src_type.clone(),
                        );
                        gen_move(&mut assembly_instructions, &Operand::IMM(0), &dst, src_type);
                        assembly_instructions.push(Instruction::SetCond(Condition::Equal, dst));
                        continue;
                    }
                };
                let (src, src_type) = gen_operand(val.src, stack, symbols);
                let (dst, _) = gen_operand(val.dst, stack, symbols);
                gen_move(&mut assembly_instructions, &src, &dst, src_type.clone());
                assembly_instructions.push(Instruction::Unary(operator, dst, src_type));
            }
            generator::Instruction::BinaryOp(val) => {
                let (src1, src_type) = gen_operand(val.src1, stack, symbols);
                let (src2, _) = gen_operand(val.src2, stack, symbols);
                let (dst, _) = gen_operand(val.dst, stack, symbols);
                let operator = match val.operator {
                    // Handle simple binary operators
                    parser::BinaryOperator::Add => BinaryOperator::Add,
                    parser::BinaryOperator::Multiply => BinaryOperator::Mult,
                    parser::BinaryOperator::Subtract => BinaryOperator::Sub,
                    parser::BinaryOperator::BitAnd => BinaryOperator::BitAnd,
                    parser::BinaryOperator::BitXor => BinaryOperator::BitXor,
                    parser::BinaryOperator::BitOr => BinaryOperator::BitOr,
                    parser::BinaryOperator::LeftShift => {
                        gen_shift(
                            &mut assembly_instructions,
                            BinaryOperator::LeftShift,
                            src1,
                            src2,
                            dst,
                            src_type.clone(),
                        );
                        continue;
                    }
                    parser::BinaryOperator::RightShift => {
                        gen_shift(
                            &mut assembly_instructions,
                            BinaryOperator::RightShift,
                            src1,
                            src2,
                            dst,
                            src_type.clone(),
                        );
                        continue;
                    }
                    // Division is handled separately
                    parser::BinaryOperator::Divide => {
                        gen_division(
                            &mut assembly_instructions,
                            src1,
                            src2,
                            dst,
                            Register::AX,
                            src_type.clone(),
                        );
                        continue;
                    }
                    parser::BinaryOperator::Remainder => {
                        gen_division(
                            &mut assembly_instructions,
                            src1,
                            src2,
                            dst,
                            Register::DX,
                            src_type.clone(),
                        );
                        continue;
                    }
                    parser::BinaryOperator::GreaterThan
                    | parser::BinaryOperator::GreaterThanEqual
                    | parser::BinaryOperator::IsEqual
                    | parser::BinaryOperator::LessThan
                    | parser::BinaryOperator::LessThanEqual
                    | parser::BinaryOperator::LogicalAnd
                    | parser::BinaryOperator::LogicalOr
                    | parser::BinaryOperator::NotEqual => {
                        gen_relational_op(
                            &mut assembly_instructions,
                            val.operator,
                            src1,
                            src2,
                            dst,
                            src_type.clone(),
                        );
                        continue;
                    }
                };
                gen_move(&mut assembly_instructions, &src1, &dst, src_type.clone());
                gen_binary_op(&mut assembly_instructions, operator, src2, dst, src_type);
            }
            generator::Instruction::Jump(target) => {
                assembly_instructions.push(Instruction::Jmp(target));
            }
            generator::Instruction::JumpCond(jump) => {
                let condition = match jump.jump_type {
                    generator::JumpType::JumpIfZero => Condition::Equal,
                    generator::JumpType::JumpIfNotZero => Condition::NotEqual,
                };
                gen_compare(
                    &mut assembly_instructions,
                    &Operand::IMM(0),
                    &gen_operand(jump.condition, stack, symbols).0,
                    AssemblyType::Longword,
                );
                assembly_instructions.push(Instruction::JmpCond(condition, jump.target));
            }
            generator::Instruction::Copy(copy) => {
                let (src, src_type) = gen_operand(copy.src, stack, symbols);
                let (dst, _) = gen_operand(copy.dst, stack, symbols);
                gen_move(&mut assembly_instructions, &src, &dst, src_type);
            }
            generator::Instruction::Label(target) => {
                assembly_instructions.push(Instruction::Label(target));
            }
            generator::Instruction::Function(name, args, dst) => {
                gen_func_call(&mut assembly_instructions, stack, name, args, dst, symbols);
            }
            generator::Instruction::SignExtend(src, dst) => {
                let (mut src, src_type) = gen_operand(src, stack, symbols);
                let (dst, dst_type) = gen_operand(dst, stack, symbols);
                if matches!(src, Operand::IMM(_)) {
                    gen_move(&mut assembly_instructions, &src, &Operand::Register(Register::R10), src_type);
                    src = Operand::Register(Register::R10)
                }

                if matches!(dst, Operand::Data(_) | Operand::Stack(_)) {
                    assembly_instructions.push(Instruction::MovSX(src, Operand::Register(Register::R11)));
                    gen_move(&mut assembly_instructions, &Operand::Register(Register::R11), &dst, dst_type);
                } else {
                    assembly_instructions.push(Instruction::MovSX(src, dst));
                }
                
             } 
             generator::Instruction::Truncate(src, dst) => {
                let (mut src, _) = gen_operand(src, stack, symbols);
                let (dst, _) = gen_operand(dst, stack, symbols);
                if let Operand::IMM(val) = src {
                    if val > i32::MAX as i64 {
                        src = Operand::IMM(val & (0xFFFFFFFF));
                    }
                }
                gen_move(&mut assembly_instructions, &src, &dst, AssemblyType::Longword);
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
        let (arg_operand, arg_type) = gen_operand(arg.clone(), stack, symbols);
        gen_move(
            instructions,
            &arg_operand,
            &Operand::Register(arg_registers[i].clone()),
            arg_type,
        );
    }
    let mut i = args.len() as isize - 1;
    while i >= 6 {
        let (operand, op_type) = gen_operand(args[i as usize].clone(), stack, symbols);
        if matches!(operand, Operand::IMM(_) | Operand::Register(_))
            || op_type == AssemblyType::Quadword
        {
            instructions.push(Instruction::Push(operand));
        } else {
            gen_move(
                instructions,
                &operand,
                &Operand::Register(Register::AX),
                AssemblyType::Longword,
            );
            instructions.push(Instruction::Push(Operand::Register(Register::AX)));
        }
        i -= 1;
    }
    instructions.push(Instruction::Call(name));
    let extra_bytes = if args.len() > 6 {
        8 * (args.len() as i64 - 6) + stack_padding
    } else {
        stack_padding
    };
    if extra_bytes != 0 {
        instructions.push(Instruction::Binary(
            BinaryOperator::Add,
            Operand::IMM(extra_bytes),
            Operand::Register(Register::SP),
            AssemblyType::Quadword,
        ));
    }
    let (dst, dst_type) = gen_operand(dst, stack, symbols);
    gen_move(
        instructions,
        &Operand::Register(Register::AX),
        &dst,
        dst_type,
    )
}
fn gen_move(
    instructions: &mut Vec<Instruction>,
    src: &Operand,
    dst: &Operand,
    mov_type: AssemblyType,
) {
    match (src, dst) {
        (Operand::Stack(_) | Operand::Data(_), Operand::Stack(_) | Operand::Data(_)) => {
            instructions.push(Instruction::Mov(
                src.clone(),
                Operand::Register(Register::R10),
                mov_type.clone(),
            ));
            instructions.push(Instruction::Mov(
                Operand::Register(Register::R10),
                dst.clone(),
                mov_type,
            ));
        }
        (Operand::IMM(val), Operand::Stack(_) | Operand::Data(_)) => {
            if *val > i32::MAX as i64 {
                instructions.push(Instruction::Mov(src.clone(), Operand::Register(Register::R10), mov_type.clone()));
                instructions.push(Instruction::Mov(Operand::Register(Register::R10), dst.clone(), mov_type));
            }
        }
        _ => {
            instructions.push(Instruction::Mov(src.clone(), dst.clone(), mov_type));
        }
    }
}

fn gen_compare(
    instructions: &mut Vec<Instruction>,
    src: &Operand,
    dst: &Operand,
    cmp_type: AssemblyType,
) {
    match (src, dst) {
        (Operand::Stack(_) | Operand::Data(_), Operand::Stack(_) | Operand::Data(_)) => {
            gen_move(
                instructions,
                &src,
                &Operand::Register(Register::R10),
                cmp_type.clone(),
            );
            instructions.push(Instruction::Compare(
                Operand::Register(Register::R10),
                dst.clone(),
                cmp_type,
            ));
        }
        (_, Operand::IMM(_)) => {
            gen_move(
                instructions,
                dst,
                &Operand::Register(Register::R11),
                cmp_type.clone(),
            );
            instructions.push(Instruction::Compare(
                src.clone(),
                Operand::Register(Register::R11),
                cmp_type,
            ));
        }
        _ => {
            instructions.push(Instruction::Compare(src.clone(), dst.clone(), cmp_type));
        }
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
    gen_move(
        instructions,
        &src1,
        &Operand::Register(Register::AX),
        shift_type.clone(),
    );
    gen_move(
        instructions,
        &src2,
        &Operand::Register(Register::CX),
        shift_type.clone(),
    );
    instructions.push(Instruction::Binary(
        operator,
        Operand::Register(Register::CL),
        Operand::Register(Register::AX),
        shift_type.clone(),
    ));
    gen_move(
        instructions,
        &Operand::Register(Register::AX),
        &dst,
        shift_type,
    );
}

fn gen_binary_op(
    instructions: &mut Vec<Instruction>,
    operator: BinaryOperator,
    mut src1: Operand,
    src2: Operand,
    op_type: AssemblyType,
) {
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let operands_in_mem = matches!((&src1, &src2), (Operand::Stack(_) | Operand::Data(_), Operand::Stack(_) | Operand::Data(_)));
    let int_overflow = if let Operand::IMM(val) = src1 {
        val > i32::MAX as i64
    } else { false };
    if operands_in_mem || int_overflow {
        gen_move(instructions, &src1, &Operand::Register(Register::R10), op_type.clone());
        src1 = Operand::Register(Register::R10);
    }

    // Rewrite dst (currently only if mult tries to put result in memory)
    if operator == BinaryOperator::Mult && matches!(src2, Operand::Data(_) | Operand::Stack(_)) {
        gen_move(instructions, &src2, &Operand::Register(Register::R11), op_type.clone());
        instructions.push(Instruction::Binary(
            operator,
            src1,
            Operand::Register(Register::R11),
            op_type.clone(),
        ));
        gen_move(instructions,&Operand::Register(Register::R11), &src2, op_type);
    } else {
        instructions.push(Instruction::Binary(operator, src1, src2, op_type));
    }
}

fn gen_relational_op(
    instructions: &mut Vec<Instruction>,
    operator: parser::BinaryOperator,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    op_type: AssemblyType,
) {
    let condition = match operator {
        parser::BinaryOperator::GreaterThan => Condition::GreaterThan,
        parser::BinaryOperator::GreaterThanEqual => Condition::GreaterThanEqual,
        parser::BinaryOperator::LessThan => Condition::LessThan,
        parser::BinaryOperator::LessThanEqual => Condition::LessThanEqual,
        parser::BinaryOperator::NotEqual => Condition::NotEqual,
        parser::BinaryOperator::IsEqual => Condition::Equal,
        _ => panic!("Expected relational operator!"),
    };
    gen_compare(instructions, &src2, &src1, op_type.clone());
    instructions.push(Instruction::Mov(Operand::IMM(0), dst.clone(), op_type));
    instructions.push(Instruction::SetCond(condition, dst));
}

fn gen_division(
    instructions: &mut Vec<Instruction>,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    result_reg: Register,
    div_type: AssemblyType,
) {
    gen_move(
        instructions,
        &src1,
        &Operand::Register(Register::AX),
        div_type.clone(),
    );
    instructions.push(Instruction::Cdq(div_type.clone()));
    if let Operand::IMM(_) = src2 {
        gen_move(
            instructions,
            &src2,
            &Operand::Register(Register::R10),
            div_type.clone(),
        );
        instructions.push(Instruction::IDiv(
            Operand::Register(Register::R10),
            div_type.clone(),
        ));
    } else {
        instructions.push(Instruction::IDiv(src2.clone(), div_type.clone()));
    }
    gen_move(instructions, &Operand::Register(result_reg), &dst, div_type);
}

fn gen_operand(
    value: generator::Value,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> (Operand, AssemblyType) {
    match value {
        generator::Value::ConstValue(constexpr) => match constexpr {
            parser::Constant::Int(val) => (Operand::IMM(val), AssemblyType::Longword),
            parser::Constant::Long(val) => (Operand::IMM(val), AssemblyType::Quadword),
        },
        generator::Value::Variable(name) => {
            let symbol = &symbols[&name];
            let var_type = AssemblyType::from(&symbol.ctype);
            if matches!(symbol.attrs, SymbolAttr::Static(_)) {
                (Operand::Data(name), var_type)
            } else if let Some(location) = stack.variables.get(&name) {
                (Operand::Stack(*location as isize), var_type)
            } else {
                stack.stack_offset += 4;
                stack.variables.insert(name.to_string(), stack.stack_offset);
                (Operand::Stack(stack.stack_offset as isize), var_type)
            }
        }
    }
}
