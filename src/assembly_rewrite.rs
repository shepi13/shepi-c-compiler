use crate::assembly::{
    self, AssemblyType::Longword, AssemblyType::Quadword, Function, Instruction, Operand,
    Operand::IMM, Operand::Register as Reg, Program, TopLevelDecl,
};
use crate::assembly::{AssemblyType, BinaryOperator};
use assembly::Register::*;

pub fn rewrite_assembly(program: Program) -> Program {
    let decls = program.program.into_iter().map(rewrite_decl);
    Program {
        program: decls.collect(),
        backend_symbols: program.backend_symbols,
    }
}

fn rewrite_decl(decl: TopLevelDecl) -> TopLevelDecl {
    match decl {
        TopLevelDecl::FUNCTION(function) => TopLevelDecl::FUNCTION(Function {
            name: function.name,
            instructions: rewrite_instructions(function.instructions),
            global: function.global,
        }),
        _ => decl,
    }
}

fn rewrite_instructions(old_instructions: Vec<Instruction>) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    for instruction in old_instructions {
        match &instruction {
            Instruction::Mov(_, _, _) => rewrite_mov(&mut instructions, instruction),
            Instruction::Div(_, _) | Instruction::IDiv(_, _) => {
                rewrite_div(&mut instructions, instruction)
            }
            // Arithmetic and bitwise binary operations
            Instruction::Binary(op, _, _, _) => {
                use BinaryOperator::*;
                match op {
                    Add | BitAnd | BitOr | BitXor | Mult | Sub => {
                        rewrite_arithmetic_binary(&mut instructions, instruction)
                    }
                    _ => instructions.push(instruction),
                }
            }
            Instruction::Compare(_, _, _) => rewrite_cmp(&mut instructions, instruction),
            Instruction::Push(_) => rewrite_push(&mut instructions, instruction),
            Instruction::MovZeroExtend(_, _) => rewrite_zero_extend(&mut instructions, instruction),
            _ => instructions.push(instruction),
        }
    }
    instructions
}

fn rewrite_mov(instructions: &mut Vec<Instruction>, mov: Instruction) {
    let Instruction::Mov(src, dst, mov_type) = mov else {
        panic!("Expected mov!")
    };
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let operands_in_mem = is_mem_operand(&src) && is_mem_operand(&dst);
    if operands_in_mem || check_overflow(&src, i32::MAX as i128) {
        instructions.push(Instruction::Mov(src, Reg(R10), mov_type.clone()));
        instructions.push(Instruction::Mov(Reg(R10), dst, mov_type));
    } else {
        instructions.push(Instruction::Mov(src, dst, mov_type));
    }
}

fn rewrite_zero_extend(instructions: &mut Vec<Instruction>, zero_x: Instruction) {
    let Instruction::MovZeroExtend(src, dst) = zero_x else {
        panic!("Expected movZX!")
    };
    if is_mem_operand(&dst) {
        instructions.push(Instruction::Mov(src, Reg(R11), Longword));
        instructions.push(Instruction::Mov(Reg(R11), dst, Quadword));
    } else {
        instructions.push(Instruction::Mov(src, dst, Longword));
    }
}

fn rewrite_push(instructions: &mut Vec<Instruction>, push: Instruction) {
    let Instruction::Push(operand) = push else {
        panic!("Expected push!")
    };
    if check_overflow(&operand, i32::MAX as i128) {
        instructions.push(Instruction::Mov(operand, Reg(R10), Quadword));
        instructions.push(Instruction::Push(Reg(R10)));
    } else {
        instructions.push(Instruction::Push(operand));
    }
}

fn rewrite_cmp(instructions: &mut Vec<Instruction>, cmp: Instruction) {
    let Instruction::Compare(mut src, dst, cmp_type) = cmp else {
        panic!("Expected cmp!")
    };
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let operands_in_mem = is_mem_operand(&src) && is_mem_operand(&dst);
    if operands_in_mem || check_overflow(&src, i32::MAX as i128) {
        instructions.push(Instruction::Mov(src, Reg(R10), cmp_type.clone()));
        src = Reg(R10);
    }
    // Rewrite dst if it's a constant
    if matches!(dst, IMM(_)) {
        instructions.push(Instruction::Mov(dst, Reg(R11), cmp_type.clone()));
        instructions.push(Instruction::Compare(src, Reg(R11), cmp_type));
    } else {
        instructions.push(Instruction::Compare(src, dst, cmp_type));
    }
}

fn rewrite_div(instructions: &mut Vec<Instruction>, div: Instruction) {
    match div {
        Instruction::Div(IMM(val), asm_type) => {
            instructions.push(Instruction::Mov(IMM(val), Reg(R10), asm_type.clone()));
            instructions.push(Instruction::Div(Reg(R10), asm_type));
        }
        Instruction::IDiv(IMM(val), asm_type) => {
            instructions.push(Instruction::Mov(IMM(val), Reg(R10), asm_type.clone()));
            instructions.push(Instruction::IDiv(Reg(R10), asm_type));
        }
        _ => instructions.push(div),
    };
}

fn rewrite_arithmetic_binary(instructions: &mut Vec<Instruction>, bin_op: Instruction) {
    let Instruction::Binary(operator, mut src, dst, op_type) = bin_op else {
        panic!("Expected binary operator!")
    };
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let operands_in_mem = is_mem_operand(&src) && is_mem_operand(&dst);
    if operands_in_mem || check_overflow(&src, i32::MAX as i128) {
        instructions.push(Instruction::Mov(src, Reg(R10), op_type.clone()));
        src = Reg(R10);
    }
    // Rewrite dst (currently only if mult tries to put result in memory)
    if operator == BinaryOperator::Mult && matches!(dst, Operand::Data(_) | Operand::Stack(_)) {
        instructions.push(Instruction::Mov(dst.clone(), Reg(R11), op_type.clone()));
        instructions.push(Instruction::Binary(
            operator,
            src,
            Reg(R11),
            op_type.clone(),
        ));
        instructions.push(Instruction::Mov(Reg(R11), dst, op_type));
    } else {
        instructions.push(Instruction::Binary(operator, src, dst, op_type));
    }
}

fn check_overflow(operand: &Operand, max_size: i128) -> bool {
    match operand {
        Operand::IMM(val) => *val > max_size,
        _ => false,
    }
}

fn is_mem_operand(operand: &Operand) -> bool {
    matches!(operand, Operand::Data(_) | Operand::Stack(_))
}
