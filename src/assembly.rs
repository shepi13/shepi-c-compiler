use std::collections::HashMap;

use crate::generator;
use crate::parser;

#[derive(Debug)]
pub struct Program {
    pub main: Function,
}
#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
}

// Instructions
#[derive(Debug)]
pub enum Instruction {
    MOV(Operand, Operand),
    UNARY(UnaryOperator, Operand),
    BINARY(BinaryOperator, Operand, Operand),
    COMPARE(Operand, Operand),
    IDIV(Operand),
    JMP(String),
    JMPCond(Condition, String),
    SetCond(Condition, Operand),
    LABEL(String),
    STACKALLOCATE(usize),
    CDQ,
    RET,
}
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    NOT,
    NEG,
}
#[derive(Debug, Clone)]
pub enum BinaryOperator {
    ADD,
    MULT,
    SUB,
    BITAND,
    BITOR,
    BITXOR,
    LEFTSHIFT,
    RIGHTSHIFT,
}
// Operands
#[derive(Debug, Clone)]
pub enum Operand {
    IMM(u32),
    STACK(usize),
    REGISTER(Register),
}
#[derive(Debug, Clone)]
pub enum Register {
    AX,
    CX,
    DX,
    R10,
    R11,
    CL,
}
#[derive(Debug, Clone)]
pub enum Condition {
    EQUAL,
    NOTEQUAL,
    GREATERTHAN,
    GREATERTHANEQUAL,
    LESSTHAN,
    LESSTHANEQUAL,
}

pub struct StackGen {
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

pub fn gen_assembly_tree(ast: generator::Program, stack: &mut StackGen) -> Program {
    Program {
        main: gen_function(ast.main, stack),
    }
}
fn gen_function(function: generator::Function, stack: &mut StackGen) -> Function {
    Function {
        name: function.name.clone(),
        instructions: gen_instructions(function.instructions, stack),
    }
}
fn gen_instructions(
    instructions: Vec<generator::Instruction>,
    stack: &mut StackGen,
) -> Vec<Instruction> {
    let mut assembly_instructions: Vec<Instruction> = Vec::new();
    assembly_instructions.push(Instruction::STACKALLOCATE(0));
    for instruction in instructions {
        match instruction {
            generator::Instruction::RETURN(val) => {
                let val = gen_operand(val, stack);
                gen_move(
                    &mut assembly_instructions,
                    &val,
                    &Operand::REGISTER(Register::AX),
                );
                assembly_instructions.push(Instruction::RET);
            }
            generator::Instruction::UNARYOP(val) => {
                let operator = match val.operator {
                    generator::UnaryOperator::COMPLEMENT => UnaryOperator::NOT,
                    generator::UnaryOperator::NEGATE => UnaryOperator::NEG,
                    generator::UnaryOperator::LOGICALNOT => {
                        let src = gen_operand(val.src, stack);
                        let dst = gen_operand(val.dst, stack);
                        gen_compare(&mut assembly_instructions, &Operand::IMM(0), &src);
                        gen_move(&mut assembly_instructions, &Operand::IMM(0), &dst);
                        assembly_instructions.push(Instruction::SetCond(Condition::EQUAL, dst));
                        continue;
                    }
                };
                let src = gen_operand(val.src, stack);
                let dst = gen_operand(val.dst, stack);
                gen_move(&mut assembly_instructions, &src, &dst);
                assembly_instructions.push(Instruction::UNARY(operator, dst));
            }
            generator::Instruction::BINARYOP(val) => {
                let src1 = gen_operand(val.src1, stack);
                let src2 = gen_operand(val.src2, stack);
                let dst = gen_operand(val.dst, stack);
                let operator = match val.operator {
                    // Handle simple binary operators
                    parser::BinaryOperator::ADD => BinaryOperator::ADD,
                    parser::BinaryOperator::MULTIPLY => BinaryOperator::MULT,
                    parser::BinaryOperator::SUBTRACT => BinaryOperator::SUB,
                    parser::BinaryOperator::BITAND => BinaryOperator::BITAND,
                    parser::BinaryOperator::BITXOR => BinaryOperator::BITXOR,
                    parser::BinaryOperator::BITOR => BinaryOperator::BITOR,
                    parser::BinaryOperator::LEFTSHIFT => {
                        gen_shift(
                            &mut assembly_instructions,
                            BinaryOperator::LEFTSHIFT,
                            src1,
                            src2,
                            dst,
                        );
                        continue;
                    }
                    parser::BinaryOperator::RIGHTSHIFT => {
                        gen_shift(
                            &mut assembly_instructions,
                            BinaryOperator::RIGHTSHIFT,
                            src1,
                            src2,
                            dst,
                        );
                        continue;
                    }
                    // Division is handled separately
                    parser::BinaryOperator::DIVIDE => {
                        gen_division(&mut assembly_instructions, src1, src2, dst, Register::AX);
                        continue;
                    }
                    parser::BinaryOperator::REMAINDER => {
                        gen_division(&mut assembly_instructions, src1, src2, dst, Register::DX);
                        continue;
                    }
                    parser::BinaryOperator::GREATERTHAN
                    | parser::BinaryOperator::GREATERTHANEQUAL
                    | parser::BinaryOperator::ISEQUAL
                    | parser::BinaryOperator::LESSTHAN
                    | parser::BinaryOperator::LESSTHANEQUAL
                    | parser::BinaryOperator::LOGICALAND
                    | parser::BinaryOperator::LOGICALOR
                    | parser::BinaryOperator::NOTEQUAL => {
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
            generator::Instruction::JUMP(jump) => {
                let condition = match jump.jump_type {
                    generator::JumpType::JUMP => {
                        assembly_instructions.push(Instruction::JMP(jump.target));
                        continue;
                    }
                    generator::JumpType::JUMPIFZERO => Condition::EQUAL,
                    generator::JumpType::JUMPIFNOTZERO => Condition::NOTEQUAL,
                };
                gen_compare(
                    &mut assembly_instructions,
                    &Operand::IMM(0),
                    &gen_operand(jump.condition, stack),
                );
                assembly_instructions.push(Instruction::JMPCond(condition, jump.target));
            }
            generator::Instruction::COPY(copy) => {
                let src = gen_operand(copy.src, stack);
                let dst = gen_operand(copy.dst, stack);
                gen_move(&mut assembly_instructions, &src, &dst);
            }
            generator::Instruction::LABEL(target) => {
                assembly_instructions.push(Instruction::LABEL(target));
            }
        }
    }
    assembly_instructions
}

fn gen_move(instructions: &mut Vec<Instruction>, src: &Operand, dst: &Operand) {
    match (src, dst) {
        (Operand::STACK(_), Operand::STACK(_)) => {
            instructions.push(Instruction::MOV(
                src.clone(),
                Operand::REGISTER(Register::R10),
            ));
            instructions.push(Instruction::MOV(
                Operand::REGISTER(Register::R10),
                dst.clone(),
            ));
        }
        _ => {
            instructions.push(Instruction::MOV(src.clone(), dst.clone()));
        }
    }
}

fn gen_compare(instructions: &mut Vec<Instruction>, src: &Operand, dst: &Operand) {
    match (src, dst) {
        (Operand::STACK(_), Operand::STACK(_)) => {
            gen_move(instructions, &src, &Operand::REGISTER(Register::R10));
            instructions.push(Instruction::COMPARE(
                Operand::REGISTER(Register::R10),
                dst.clone(),
            ));
        }
        (_, Operand::IMM(_)) => {
            gen_move(instructions, dst, &Operand::REGISTER(Register::R11));
            instructions.push(Instruction::COMPARE(
                src.clone(),
                Operand::REGISTER(Register::R11),
            ));
        }
        _ => {
            instructions.push(Instruction::COMPARE(src.clone(), dst.clone()));
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
    gen_move(instructions, &src1, &Operand::REGISTER(Register::AX));
    gen_move(instructions, &src2, &Operand::REGISTER(Register::CX));
    instructions.push(Instruction::BINARY(
        operator,
        Operand::REGISTER(Register::CL),
        Operand::REGISTER(Register::AX),
    ));
    gen_move(instructions, &Operand::REGISTER(Register::AX), &dst);
}

fn gen_binary_op(
    instructions: &mut Vec<Instruction>,
    operator: BinaryOperator,
    src1: Operand,
    src2: Operand,
) {
    match (&operator, &src1, &src2) {
        (BinaryOperator::MULT, _, Operand::STACK(_)) => {
            gen_move(instructions, &src2, &Operand::REGISTER(Register::R11));
            instructions.push(Instruction::BINARY(
                operator,
                src1,
                Operand::REGISTER(Register::R11),
            ));
            gen_move(instructions, &Operand::REGISTER(Register::R11), &src2);
        }
        (_, Operand::STACK(_), Operand::STACK(_)) => {
            gen_move(instructions, &src1, &Operand::REGISTER(Register::R10));
            instructions.push(Instruction::BINARY(
                operator,
                Operand::REGISTER(Register::R10),
                src2,
            ));
        }
        _ => {
            instructions.push(Instruction::BINARY(operator, src1, src2));
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
        parser::BinaryOperator::GREATERTHAN => Condition::GREATERTHAN,
        parser::BinaryOperator::GREATERTHANEQUAL => Condition::GREATERTHANEQUAL,
        parser::BinaryOperator::LESSTHAN => Condition::LESSTHAN,
        parser::BinaryOperator::LESSTHANEQUAL => Condition::LESSTHANEQUAL,
        parser::BinaryOperator::NOTEQUAL => Condition::NOTEQUAL,
        parser::BinaryOperator::ISEQUAL => Condition::EQUAL,
        _ => panic!("Expected relational operator!"),
    };
    gen_compare(instructions, &src2, &src1);
    instructions.push(Instruction::MOV(Operand::IMM(0), dst.clone()));
    instructions.push(Instruction::SetCond(condition, dst));
}

fn gen_division(
    instructions: &mut Vec<Instruction>,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    result_reg: Register,
) {
    gen_move(instructions, &src1, &Operand::REGISTER(Register::AX));
    instructions.push(Instruction::CDQ);
    if let Operand::IMM(_) = src2 {
        gen_move(instructions, &src2, &Operand::REGISTER(Register::R10));
        instructions.push(Instruction::IDIV(Operand::REGISTER(Register::R10)));
    } else {
        instructions.push(Instruction::IDIV(src2.clone()));
    }
    gen_move(instructions, &Operand::REGISTER(result_reg), &dst);
}

fn gen_operand(value: generator::Value, stack: &mut StackGen) -> Operand {
    match value {
        generator::Value::CONSTANT(val) => Operand::IMM(val),
        generator::Value::VARIABLE(name) => {
            if let Some(location) = stack.variables.get(&name) {
                Operand::STACK(*location)
            } else {
                stack.stack_offset += 4;
                stack.variables.insert(name.to_string(), stack.stack_offset);
                Operand::STACK(stack.stack_offset)
            }
        }
    }
}

pub fn add_stack_offset(program: &mut Program, stack: StackGen) {
    program.main.instructions[0] = Instruction::STACKALLOCATE(stack.stack_offset);
}
