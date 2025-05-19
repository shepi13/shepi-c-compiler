use std::collections::HashMap;

use crate::generator;
use crate::parser;

pub type Program<'a> = Vec<Function<'a>>;
#[derive(Debug)]
pub struct Function<'a> {
    pub name: &'a str,
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
    STACKDEALLOCATE(usize),
    PUSH(Operand),
    CALL(String),
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
    STACK(isize),
    REGISTER(Register),
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
    EQUAL,
    NOTEQUAL,
    GREATERTHAN,
    GREATERTHANEQUAL,
    LESSTHAN,
    LESSTHANEQUAL,
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

pub fn gen_assembly_tree<'a>(ast: generator::Program<'a>) -> Program<'a> {
    let mut program: Program = Vec::new();
    for function in ast.into_iter() {
        if function.instructions.is_empty() {
            continue;
        }
        let stack = &mut StackGen::new();
        let mut instructions: Vec<Instruction> = Vec::new();
        instructions.push(Instruction::STACKALLOCATE(0));
        for (i, param) in function.params.iter().enumerate() {
            let param = gen_operand(generator::Value::VARIABLE(param.to_string()), stack);
            match i {
                0 => gen_move(&mut instructions, &Operand::REGISTER(Register::DI), &param),
                1 => gen_move(&mut instructions, &Operand::REGISTER(Register::SI), &param),
                2 => gen_move(&mut instructions, &Operand::REGISTER(Register::DX), &param),
                3 => gen_move(&mut instructions, &Operand::REGISTER(Register::CX), &param),
                4 => gen_move(&mut instructions, &Operand::REGISTER(Register::R8), &param),
                5 => gen_move(&mut instructions, &Operand::REGISTER(Register::R9), &param),
                _ => gen_move(
                    &mut instructions,
                    &Operand::STACK((4 - i as isize) * 8),
                    &param,
                ),
            }
        }
        instructions.append(&mut gen_instructions(function.instructions, stack));
        instructions[0] =
            Instruction::STACKALLOCATE(stack.stack_offset + 16 - stack.stack_offset % 16);
        program.push(Function {
            name: function.name,
            instructions,
        });
    }
    program
}
fn gen_instructions(
    instructions: Vec<generator::Instruction>,
    stack: &mut StackGen,
) -> Vec<Instruction> {
    let mut assembly_instructions: Vec<Instruction> = Vec::new();
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
            generator::Instruction::JUMP(target) => {
                assembly_instructions.push(Instruction::JMP(target));
            }
            generator::Instruction::JUMPCOND(jump) => {
                let condition = match jump.jump_type {
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
            generator::Instruction::FUNCTION(name, args, dst) => {
                gen_func_call(&mut assembly_instructions, stack, name, args, dst);
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
        instructions.push(Instruction::STACKALLOCATE(stack_padding));
    }
    for (i, arg) in args.iter().enumerate() {
        if i >= 6 {
            break;
        }
        gen_move(
            instructions,
            &gen_operand(arg.clone(), stack),
            &Operand::REGISTER(arg_registers[i].clone()),
        );
    }
    let mut i = args.len() as isize - 1;
    while i >= 6 {
        let operand = gen_operand(args[i as usize].clone(), stack);
        match operand {
            Operand::IMM(_) | Operand::REGISTER(_) => {
                instructions.push(Instruction::PUSH(operand));
            }
            _ => {
                gen_move(instructions, &operand, &Operand::REGISTER(Register::AX));
                instructions.push(Instruction::PUSH(Operand::REGISTER(Register::AX)));
            }
        }
        i -= 1;
    }
    instructions.push(Instruction::CALL(name));
    let extra_bytes = if args.len() > 6 {
        8 * (args.len() - 6) + stack_padding
    } else {
        stack_padding
    };
    if extra_bytes != 0 {
        instructions.push(Instruction::STACKDEALLOCATE(extra_bytes));
    }
    gen_move(
        instructions,
        &Operand::REGISTER(Register::AX),
        &gen_operand(dst, stack),
    )
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
                Operand::STACK(*location as isize)
            } else {
                stack.stack_offset += 4;
                stack.variables.insert(name.to_string(), stack.stack_offset);
                Operand::STACK(stack.stack_offset as isize)
            }
        }
    }
}
