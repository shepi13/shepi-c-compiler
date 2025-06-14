use super::assembly_ast::{AssemblyType, Register::*};
use super::assembly_ast::{
    AssemblyType::Longword, AssemblyType::Quadword, BinaryOperator, Function, Instruction, Operand,
    Operand::Imm, Operand::Register as Reg, Program, TopLevelDecl,
};

pub fn rewrite_assembly(program: Program) -> Program {
    let decls = program.program.into_iter().map(rewrite_decl);
    Program {
        program: decls.collect(),
        backend_symbols: program.backend_symbols,
    }
}

fn rewrite_decl(decl: TopLevelDecl) -> TopLevelDecl {
    match decl {
        TopLevelDecl::FunctionDecl(function) => TopLevelDecl::FunctionDecl(Function {
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
            Instruction::Binary(op, _, _, asm_type) => {
                use BinaryOperator::*;
                match op {
                    Add | BitAnd | BitOr | BitXor | Mult | Sub | DoubleDiv => {
                        if *asm_type == AssemblyType::Double {
                            rewrite_arithmetic_binary_f(&mut instructions, instruction);
                        } else {
                            rewrite_arithmetic_binary(&mut instructions, instruction)
                        }
                    }
                    _ => instructions.push(instruction),
                }
            }
            Instruction::Compare(_, _, AssemblyType::Double) => {
                rewrite_comisd(&mut instructions, instruction)
            }
            Instruction::Compare(_, _, _) => rewrite_cmp(&mut instructions, instruction),
            Instruction::Push(_) => rewrite_push(&mut instructions, instruction),
            Instruction::MovSignExtend(_, _, _, _) => {
                rewrite_sign_extend(&mut instructions, instruction)
            }
            Instruction::MovZeroExtend(_, _, _, _) => {
                rewrite_zero_extend(&mut instructions, instruction)
            }
            Instruction::Cvttsd2si(_, _, _) => rewrite_cvttsd2si(&mut instructions, instruction),
            Instruction::Cvtsi2sd(_, _, _) => rewrite_cvtsi2sd(&mut instructions, instruction),
            Instruction::Lea(_, _) => rewrite_lea(&mut instructions, instruction),
            _ => instructions.push(instruction),
        }
    }
    instructions
}

fn rewrite_mov(instructions: &mut Vec<Instruction>, mov: Instruction) {
    let Instruction::Mov(mut src, dst, mov_type) = mov else { panic!("Expected mov!") };
    // Rewrite IMM if it doesn't fit in a byte
    if mov_type == AssemblyType::Byte && check_overflow(&src, u8::MAX as i128) {
        let Imm(val) = src else { panic!("Can only overflow if IMM!") };
        src = Imm(val % 256);
    }
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let scratch = if mov_type == AssemblyType::Double { XMM14 } else { R10 };
    let operands_in_mem = is_mem_operand(&src) && is_mem_operand(&dst);
    if operands_in_mem || check_overflow(&src, i32::MAX as i128) {
        instructions.push(Instruction::Mov(src, Reg(scratch), mov_type));
        instructions.push(Instruction::Mov(Reg(scratch), dst, mov_type));
    } else {
        instructions.push(Instruction::Mov(src, dst, mov_type));
    }
}

fn rewrite_lea(instructions: &mut Vec<Instruction>, lea: Instruction) {
    let Instruction::Lea(src, dst) = lea else { panic!("Expected lea") };
    if !matches!(dst, Reg(_)) {
        instructions.push(Instruction::Lea(src, Reg(R11)));
        instructions.push(Instruction::Mov(Reg(R11), dst, AssemblyType::Quadword));
    } else {
        instructions.push(Instruction::Lea(src, dst));
    }
}

fn rewrite_cvttsd2si(instructions: &mut Vec<Instruction>, cvt: Instruction) {
    let Instruction::Cvttsd2si(src, dst, cvt_type) = cvt else {
        panic!("Expected Cvttsd2si");
    };
    if !matches!(dst, Reg(_)) {
        instructions.push(Instruction::Cvttsd2si(src, Reg(R11), cvt_type));
        instructions.push(Instruction::Mov(Reg(R11), dst, cvt_type));
    } else {
        instructions.push(Instruction::Cvttsd2si(src, dst, cvt_type));
    }
}

fn rewrite_cvtsi2sd(instructions: &mut Vec<Instruction>, cvt: Instruction) {
    let Instruction::Cvtsi2sd(mut src, dst, cvt_type) = cvt else {
        panic!("Expected Cvtsi2sd");
    };
    if matches!(src, Imm(_)) {
        instructions.push(Instruction::Mov(src, Reg(R10), cvt_type));
        src = Reg(R10);
    }
    if !matches!(dst, Reg(_)) {
        instructions.push(Instruction::Cvtsi2sd(src, Reg(XMM15), cvt_type));
        instructions.push(Instruction::Mov(Reg(XMM15), dst, AssemblyType::Double));
    } else {
        instructions.push(Instruction::Cvtsi2sd(src, dst, cvt_type));
    }
}

fn rewrite_sign_extend(instructions: &mut Vec<Instruction>, sign_x: Instruction) {
    let Instruction::MovSignExtend(mut src, dst, src_t, dst_t) = sign_x else {
        panic!("Expected moveSX!")
    };
    // src cannot be constant
    if matches!(src, Imm(_)) {
        instructions.push(Instruction::Mov(src, Reg(R10), src_t));
        src = Reg(R10);
    }
    // dst cannot be in memory
    if is_mem_operand(&dst) {
        instructions.push(Instruction::MovSignExtend(src, Reg(R11), src_t, dst_t));
        instructions.push(Instruction::Mov(Reg(R11), dst, dst_t))
    } else {
        instructions.push(Instruction::MovSignExtend(src, dst, src_t, dst_t));
    }
}

fn rewrite_zero_extend(instructions: &mut Vec<Instruction>, zero_x: Instruction) {
    let Instruction::MovZeroExtend(mut src, dst, src_t, dst_t) = zero_x else {
        panic!("Expected movZX!")
    };
    // Zero extend for longs is just a move using an intermediate register if necessary
    if src_t == Longword {
        if is_mem_operand(&dst) {
            instructions.push(Instruction::Mov(src, Reg(R11), Longword));
            instructions.push(Instruction::Mov(Reg(R11), dst, Quadword));
        } else {
            instructions.push(Instruction::Mov(src, dst, Longword));
        }
    } else {
        // src cannot be constant
        if matches!(src, Imm(_)) {
            instructions.push(Instruction::Mov(src, Reg(R10), src_t));
            src = Reg(R10);
        }
        // destination must be register
        if is_mem_operand(&dst) {
            instructions.push(Instruction::MovZeroExtend(src, Reg(R11), src_t, dst_t));
            instructions.push(Instruction::Mov(Reg(R11), dst, dst_t));
        } else {
            instructions.push(Instruction::MovZeroExtend(src, dst, src_t, dst_t));
        }
    }
}

fn rewrite_push(instructions: &mut Vec<Instruction>, push: Instruction) {
    let Instruction::Push(operand) = push else { panic!("Expected push!") };
    // Rewrite push if it pushes an xmm register (need to use mov instead)
    if matches!(
        operand,
        Reg(XMM0)
            | Reg(XMM1)
            | Reg(XMM2)
            | Reg(XMM3)
            | Reg(XMM4)
            | Reg(XMM5)
            | Reg(XMM6)
            | Reg(XMM7)
            | Reg(XMM14)
            | Reg(XMM15)
    ) {
        instructions.push(Instruction::Binary(
            BinaryOperator::Sub,
            Imm(8),
            Reg(SP),
            AssemblyType::Quadword,
        ));
        instructions.push(Instruction::Mov(
            operand,
            Operand::Memory(SP, 0),
            AssemblyType::Quadword,
        ));
    }
    // Immediate operands to push need to be size int, otherwise we use a register
    else if check_overflow(&operand, i32::MAX as i128) {
        instructions.push(Instruction::Mov(operand, Reg(R10), Quadword));
        instructions.push(Instruction::Push(Reg(R10)));
    } else {
        instructions.push(Instruction::Push(operand));
    }
}

fn rewrite_cmp(instructions: &mut Vec<Instruction>, cmp: Instruction) {
    let Instruction::Compare(mut src, dst, cmp_type) = cmp else { panic!("Expected cmp!") };
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    let operands_in_mem = is_mem_operand(&src) && is_mem_operand(&dst);
    if operands_in_mem || check_overflow(&src, i32::MAX as i128) {
        instructions.push(Instruction::Mov(src, Reg(R10), cmp_type));
        src = Reg(R10);
    }
    // Rewrite dst if it's a constant
    if matches!(dst, Imm(_)) {
        instructions.push(Instruction::Mov(dst, Reg(R11), cmp_type));
        instructions.push(Instruction::Compare(src, Reg(R11), cmp_type));
    } else {
        instructions.push(Instruction::Compare(src, dst, cmp_type));
    }
}

fn rewrite_comisd(instructions: &mut Vec<Instruction>, cmp: Instruction) {
    let Instruction::Compare(src, dst, cmp_type) = cmp else { panic!("Expected cmp!") };
    if !matches!(dst, Reg(_)) {
        instructions.push(Instruction::Mov(dst, Reg(XMM14), cmp_type));
        instructions.push(Instruction::Compare(src, Reg(XMM14), cmp_type));
    } else {
        instructions.push(Instruction::Compare(src, dst, cmp_type));
    }
}

fn rewrite_div(instructions: &mut Vec<Instruction>, div: Instruction) {
    // Div and IDiv cannot take immediate values as operands
    match div {
        Instruction::Div(Imm(val), asm_type) => {
            instructions.push(Instruction::Mov(Imm(val), Reg(R10), asm_type));
            instructions.push(Instruction::Div(Reg(R10), asm_type));
        }
        Instruction::IDiv(Imm(val), asm_type) => {
            instructions.push(Instruction::Mov(Imm(val), Reg(R10), asm_type));
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
        instructions.push(Instruction::Mov(src, Reg(R10), op_type));
        src = Reg(R10);
    }
    // Rewrite dst (currently only if mult tries to put result in memory)
    if operator == BinaryOperator::Mult && is_mem_operand(&dst) {
        instructions.push(Instruction::Mov(dst.clone(), Reg(R11), op_type));
        instructions.push(Instruction::Binary(operator, src, Reg(R11), op_type));
        instructions.push(Instruction::Mov(Reg(R11), dst, op_type));
    } else {
        instructions.push(Instruction::Binary(operator, src, dst, op_type));
    }
}

fn rewrite_arithmetic_binary_f(instructions: &mut Vec<Instruction>, bin_op: Instruction) {
    let Instruction::Binary(operator, src, dst, op_type) = bin_op else {
        panic!("Expected binary operator!")
    };
    if !matches!(dst, Reg(_)) {
        instructions.push(Instruction::Mov(dst.clone(), Reg(XMM15), op_type));
        instructions.push(Instruction::Binary(operator, src, Reg(XMM15), op_type));
        instructions.push(Instruction::Mov(Reg(XMM15), dst, op_type));
    } else {
        instructions.push(Instruction::Binary(operator, src, dst, op_type));
    }
}

pub fn check_overflow(operand: &Operand, max_size: i128) -> bool {
    match operand {
        Operand::Imm(val) => *val > max_size,
        _ => false,
    }
}

pub fn is_mem_operand(operand: &Operand) -> bool {
    matches!(operand, Operand::Data(_) | Operand::Memory(_, _))
}
