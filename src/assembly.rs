use std::collections::HashMap;

use crate::generator;
use crate::generator::StaticVariable;
use crate::parser;
use crate::type_check::{Symbol, SymbolAttr, Symbols};

pub type Program = Vec<TopLevelDecl>;
#[derive(Debug)]
pub enum TopLevelDecl {
    FUNCTION(Function),
    STATICVAR(StaticVariable),
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
    Mov(Operand, Operand),
    Unary(UnaryOperator, Operand),
    Binary(BinaryOperator, Operand, Operand),
    Compare(Operand, Operand),
    IDiv(Operand),
    Jmp(String),
    JmpCond(Condition, String),
    SetCond(Condition, Operand),
    Label(String),
    StackAllocate(usize),
    StackDeallocate(usize),
    Push(Operand),
    Call(String),
    Cdq,
    Ret,
}
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Not,
    Neg,
}
#[derive(Debug, Clone)]
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

pub fn gen_assembly_tree(ast: generator::Program, symbols: &Symbols) -> Program {
    let mut program: Program = Vec::new();
    for decl in ast.into_iter() {
        match decl {
            generator::TopLevelDecl::Function(function) => {
                if function.instructions.is_empty() {
                    continue;
                }
                let stack = &mut StackGen::new();
                let mut instructions: Vec<Instruction> = Vec::new();
                instructions.push(Instruction::StackAllocate(0));
                for (i, param) in function.params.iter().enumerate() {
                    let param = gen_operand(
                        generator::Value::Variable(param.to_string()),
                        stack,
                        symbols,
                    );
                    match i {
                        0 => gen_move(&mut instructions, &Operand::Register(Register::DI), &param),
                        1 => gen_move(&mut instructions, &Operand::Register(Register::SI), &param),
                        2 => gen_move(&mut instructions, &Operand::Register(Register::DX), &param),
                        3 => gen_move(&mut instructions, &Operand::Register(Register::CX), &param),
                        4 => gen_move(&mut instructions, &Operand::Register(Register::R8), &param),
                        5 => gen_move(&mut instructions, &Operand::Register(Register::R9), &param),
                        _ => gen_move(
                            &mut instructions,
                            &Operand::Stack((4 - i as isize) * 8),
                            &param,
                        ),
                    }
                }
                instructions.append(&mut gen_instructions(function.instructions, stack, symbols));
                instructions[0] =
                    Instruction::StackAllocate(stack.stack_offset + 16 - stack.stack_offset % 16);
                program.push(TopLevelDecl::FUNCTION(Function {
                    name: function.name,
                    global: function.global,
                    instructions,
                }));
            }
            generator::TopLevelDecl::StaticDecl(static_data) => {
                program.push(TopLevelDecl::STATICVAR(static_data));
            }
        }
    }
    program
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
                let val = gen_operand(val, stack, symbols);
                gen_move(
                    &mut assembly_instructions,
                    &val,
                    &Operand::Register(Register::AX),
                );
                assembly_instructions.push(Instruction::Ret);
            }
            generator::Instruction::UnaryOp(val) => {
                let operator = match val.operator {
                    generator::UnaryOperator::Complement => UnaryOperator::Not,
                    generator::UnaryOperator::Negate => UnaryOperator::Neg,
                    generator::UnaryOperator::LogicalNot => {
                        let src = gen_operand(val.src, stack, symbols);
                        let dst = gen_operand(val.dst, stack, symbols);
                        gen_compare(&mut assembly_instructions, &Operand::IMM(0), &src);
                        gen_move(&mut assembly_instructions, &Operand::IMM(0), &dst);
                        assembly_instructions.push(Instruction::SetCond(Condition::Equal, dst));
                        continue;
                    }
                };
                let src = gen_operand(val.src, stack, symbols);
                let dst = gen_operand(val.dst, stack, symbols);
                gen_move(&mut assembly_instructions, &src, &dst);
                assembly_instructions.push(Instruction::Unary(operator, dst));
            }
            generator::Instruction::BinaryOp(val) => {
                let src1 = gen_operand(val.src1, stack, symbols);
                let src2 = gen_operand(val.src2, stack, symbols);
                let dst = gen_operand(val.dst, stack, symbols);
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
                        );
                        continue;
                    }
                    // Division is handled separately
                    parser::BinaryOperator::Divide => {
                        gen_division(&mut assembly_instructions, src1, src2, dst, Register::AX);
                        continue;
                    }
                    parser::BinaryOperator::Remainder => {
                        gen_division(&mut assembly_instructions, src1, src2, dst, Register::DX);
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
                        );
                        continue;
                    }
                };
                gen_move(&mut assembly_instructions, &src1, &dst);
                gen_binary_op(&mut assembly_instructions, operator, src2, dst);
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
                    &gen_operand(jump.condition, stack, symbols),
                );
                assembly_instructions.push(Instruction::JmpCond(condition, jump.target));
            }
            generator::Instruction::Copy(copy) => {
                let src = gen_operand(copy.src, stack, symbols);
                let dst = gen_operand(copy.dst, stack, symbols);
                gen_move(&mut assembly_instructions, &src, &dst);
            }
            generator::Instruction::Label(target) => {
                assembly_instructions.push(Instruction::Label(target));
            }
            generator::Instruction::Function(name, args, dst) => {
                gen_func_call(&mut assembly_instructions, stack, name, args, dst, symbols);
            }
            generator::Instruction::SignExtend(_, _) | generator::Instruction::Truncate(_, _) => panic!("Not implemented!")
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
        instructions.push(Instruction::StackAllocate(stack_padding));
    }
    for (i, arg) in args.iter().enumerate() {
        if i >= 6 {
            break;
        }
        gen_move(
            instructions,
            &gen_operand(arg.clone(), stack, symbols),
            &Operand::Register(arg_registers[i].clone()),
        );
    }
    let mut i = args.len() as isize - 1;
    while i >= 6 {
        let operand = gen_operand(args[i as usize].clone(), stack, symbols);
        match operand {
            Operand::IMM(_) | Operand::Register(_) => {
                instructions.push(Instruction::Push(operand));
            }
            _ => {
                gen_move(instructions, &operand, &Operand::Register(Register::AX));
                instructions.push(Instruction::Push(Operand::Register(Register::AX)));
            }
        }
        i -= 1;
    }
    instructions.push(Instruction::Call(name));
    let extra_bytes = if args.len() > 6 {
        8 * (args.len() - 6) + stack_padding
    } else {
        stack_padding
    };
    if extra_bytes != 0 {
        instructions.push(Instruction::StackDeallocate(extra_bytes));
    }
    gen_move(
        instructions,
        &Operand::Register(Register::AX),
        &gen_operand(dst, stack, symbols),
    )
}
fn gen_move(instructions: &mut Vec<Instruction>, src: &Operand, dst: &Operand) {
    match (src, dst) {
        (Operand::Stack(_) | Operand::Data(_), Operand::Stack(_) | Operand::Data(_)) => {
            instructions.push(Instruction::Mov(
                src.clone(),
                Operand::Register(Register::R10),
            ));
            instructions.push(Instruction::Mov(
                Operand::Register(Register::R10),
                dst.clone(),
            ));
        }
        _ => {
            instructions.push(Instruction::Mov(src.clone(), dst.clone()));
        }
    }
}

fn gen_compare(instructions: &mut Vec<Instruction>, src: &Operand, dst: &Operand) {
    match (src, dst) {
        (Operand::Stack(_) | Operand::Data(_), Operand::Stack(_) | Operand::Data(_)) => {
            gen_move(instructions, &src, &Operand::Register(Register::R10));
            instructions.push(Instruction::Compare(
                Operand::Register(Register::R10),
                dst.clone(),
            ));
        }
        (_, Operand::IMM(_)) => {
            gen_move(instructions, dst, &Operand::Register(Register::R11));
            instructions.push(Instruction::Compare(
                src.clone(),
                Operand::Register(Register::R11),
            ));
        }
        _ => {
            instructions.push(Instruction::Compare(src.clone(), dst.clone()));
        }
    }
}

fn gen_shift(
    instructions: &mut Vec<Instruction>,
    operator: BinaryOperator,
    src1: Operand,
    src2: Operand,
    dst: Operand,
) {
    gen_move(instructions, &src1, &Operand::Register(Register::AX));
    gen_move(instructions, &src2, &Operand::Register(Register::CX));
    instructions.push(Instruction::Binary(
        operator,
        Operand::Register(Register::CL),
        Operand::Register(Register::AX),
    ));
    gen_move(instructions, &Operand::Register(Register::AX), &dst);
}

fn gen_binary_op(
    instructions: &mut Vec<Instruction>,
    operator: BinaryOperator,
    src1: Operand,
    src2: Operand,
) {
    match (&operator, &src1, &src2) {
        (BinaryOperator::Mult, _, Operand::Stack(_) | Operand::Data(_)) => {
            gen_move(instructions, &src2, &Operand::Register(Register::R11));
            instructions.push(Instruction::Binary(
                operator,
                src1,
                Operand::Register(Register::R11),
            ));
            gen_move(instructions, &Operand::Register(Register::R11), &src2);
        }
        (_, Operand::Stack(_) | Operand::Data(_), Operand::Stack(_) | Operand::Data(_)) => {
            gen_move(instructions, &src1, &Operand::Register(Register::R10));
            instructions.push(Instruction::Binary(
                operator,
                Operand::Register(Register::R10),
                src2,
            ));
        }
        _ => {
            instructions.push(Instruction::Binary(operator, src1, src2));
        }
    };
}

fn gen_relational_op(
    instructions: &mut Vec<Instruction>,
    operator: parser::BinaryOperator,
    src1: Operand,
    src2: Operand,
    dst: Operand,
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
    gen_compare(instructions, &src2, &src1);
    instructions.push(Instruction::Mov(Operand::IMM(0), dst.clone()));
    instructions.push(Instruction::SetCond(condition, dst));
}

fn gen_division(
    instructions: &mut Vec<Instruction>,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    result_reg: Register,
) {
    gen_move(instructions, &src1, &Operand::Register(Register::AX));
    instructions.push(Instruction::Cdq);
    if let Operand::IMM(_) = src2 {
        gen_move(instructions, &src2, &Operand::Register(Register::R10));
        instructions.push(Instruction::IDiv(Operand::Register(Register::R10)));
    } else {
        instructions.push(Instruction::IDiv(src2.clone()));
    }
    gen_move(instructions, &Operand::Register(result_reg), &dst);
}

fn gen_operand(value: generator::Value, stack: &mut StackGen, symbols: &Symbols) -> Operand {
    match value {
        generator::Value::ConstValue(val) => Operand::IMM(val.value()),
        generator::Value::Variable(name) => {
            if let Some(Symbol {
                attrs: SymbolAttr::Static(_),
                ctype: _,
            }) = symbols.get(&name)
            {
                Operand::Data(name)
            } else if let Some(location) = stack.variables.get(&name) {
                Operand::Stack(*location as isize)
            } else {
                stack.stack_offset += 4;
                stack.variables.insert(name.to_string(), stack.stack_offset);
                Operand::Stack(stack.stack_offset as isize)
            }
        }
    }
}
