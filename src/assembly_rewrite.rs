use crate::assembly::{self, Function, Instruction, Operand, Program, TopLevelDecl};

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
        match instruction {
            Instruction::Mov(_, _, _) => {
                rewrite_mov(&mut instructions, instruction);
            }
            _ => instructions.push(instruction),
        }
    }
    instructions
}

fn rewrite_mov(instructions: &mut Vec<Instruction>, mov: Instruction) {
    // Rewrite src (if it is larger than max int, or if both src and dst are in memory)
    use Operand::Register;
    use assembly::Register::*;
    let Instruction::Mov(src, dst, mov_type) = mov else {
        panic!("Expected mov!")
    };
    let operands_in_mem = is_mem_operand(&src) && is_mem_operand(&dst);
    if operands_in_mem || check_overflow(&src, i32::MAX as i128) {
        instructions.push(Instruction::Mov(src, Register(R10), mov_type.clone()));
        instructions.push(Instruction::Mov(Register(R10), dst, mov_type));
    } else {
        instructions.push(Instruction::Mov(src, dst, mov_type));
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
