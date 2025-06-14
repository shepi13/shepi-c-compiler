use super::assembly_ast::{
    self, AsmSymbol, AssemblyType, BackendSymbols, BinaryOperator, Condition, Function,
    StaticConstant, StaticVar,
};
use crate::{helpers::lib::escape_asm_string, validate::ctype::Initializer};
use std::error::Error;

type Result = std::result::Result<(), Box<dyn Error>>;

pub fn emit_program<T>(output: &mut T, program: assembly_ast::Program) -> Result
where
    T: std::fmt::Write,
{
    for top_level in program.program {
        match top_level {
            assembly_ast::TopLevelDecl::FunctionDecl(function) => {
                emit_function(output, function, &program.backend_symbols)?;
            }
            assembly_ast::TopLevelDecl::Var(var) => emit_static_var(output, var)?,
            assembly_ast::TopLevelDecl::Constant(var) => emit_static_const(output, var)?,
        }
    }
    writeln!(output, "    .section .note.GNU-stack,\"\",@progbits")?;
    Ok(())
}

fn emit_initializer<T>(output: &mut T, initializer: Initializer) -> Result
where
    T: std::fmt::Write,
{
    use Initializer::*;
    match initializer {
        ZeroInit(size) => writeln!(output, "    .zero {}", size)?,
        Char(_) | UChar(_) => writeln!(output, "    .byte {}", initializer.int_value())?,
        Int(_) | UInt(_) => writeln!(output, "    .long {}", initializer.int_value())?,
        Long(_) | ULong(_) => writeln!(output, "    .quad {}", initializer.int_value())?,
        Double(val) => writeln!(output, "    .double {}", val)?,
        PointerInit(label) => writeln!(output, "    .quad {}", label)?,
        StringInit { data, null_terminated } => {
            let new_data = escape_asm_string(data);
            if null_terminated {
                writeln!(output, "    .asciz \"{}\"", new_data)?;
            } else {
                writeln!(output, "    .ascii \"{}\"", new_data)?;
            }
        }
    }
    Ok(())
}

fn emit_static_var<T>(output: &mut T, var: StaticVar) -> Result
where
    T: std::fmt::Write,
{
    if var.global {
        writeln!(output, "    .globl {}", var.name)?;
    }
    if var.init.len() == 1 && matches!(var.init[0], Initializer::ZeroInit(_)) {
        writeln!(output, "    .bss")?;
    } else {
        writeln!(output, "    .data")?;
    }
    writeln!(output, "    .align {}", var.alignment)?;
    writeln!(output, "{}:", var.name)?;
    var.init.into_iter().try_for_each(|initializer| emit_initializer(output, initializer))
}

fn emit_static_const<T>(output: &mut T, var: StaticConstant) -> Result
where
    T: std::fmt::Write,
{
    writeln!(output, "    .section .rodata")?;
    writeln!(output, "    .align {}", var.alignment)?;
    writeln!(output, "{}:", var.name)?;
    emit_initializer(output, var.init)
}

fn emit_function<T>(output: &mut T, function: Function, symbols: &BackendSymbols) -> Result
where
    T: std::fmt::Write,
{
    if function.global {
        writeln!(output, "    .globl {}", function.name)?
    }
    writeln!(output, "    .text")?;
    writeln!(output, "{}:", function.name)?;
    writeln!(output, "    pushq %rbp")?;
    writeln!(output, "    movq %rsp, %rbp")?;
    emit_instructions(output, &function.instructions, symbols)?;
    Ok(())
}
fn emit_instructions<T: std::fmt::Write>(
    output: &mut T,
    instructions: &Vec<assembly_ast::Instruction>,
    symbols: &BackendSymbols,
) -> Result {
    for instruction in instructions {
        match instruction {
            assembly_ast::Instruction::Mov(src, dst, move_type) => {
                let src = get_operand(src, move_type);
                let dst = get_operand(dst, move_type);
                let t = get_size_suffix(move_type);
                writeln!(output, "    mov{t} {src}, {dst}")?;
            }
            assembly_ast::Instruction::MovSignExtend(src, dst, src_type, dst_type) => {
                let src_t = get_size_suffix(src_type);
                let dst_t = get_size_suffix(dst_type);
                let src = get_operand(src, src_type);
                let dst = get_operand(dst, dst_type);
                writeln!(output, "    movs{src_t}{dst_t} {src}, {dst}")?;
            }
            assembly_ast::Instruction::MovZeroExtend(src, dst, src_type, dst_type) => {
                let src_t = get_size_suffix(src_type);
                let dst_t = get_size_suffix(dst_type);
                let src = get_operand(src, src_type);
                let dst = get_operand(dst, dst_type);
                writeln!(output, "    movz{src_t}{dst_t} {src}, {dst}")?;
            }
            assembly_ast::Instruction::Ret => {
                writeln!(output, "    movq %rbp, %rsp")?;
                writeln!(output, "    popq %rbp")?;
                writeln!(output, "    ret")?;
            }
            assembly_ast::Instruction::Unary(operator, operand, asm_type) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(output, "    {operator}{t} {operand}")?;
            }
            assembly_ast::Instruction::Cdq(asm_type) => match asm_type {
                AssemblyType::Longword => writeln!(output, "    cdq")?,
                AssemblyType::Quadword => writeln!(output, "    cqo")?,
                AssemblyType::Double => panic!("Cdq not supported for double!"),
                AssemblyType::ByteArray(_, _) => panic!("Cdq not supported for array!"),
                AssemblyType::Byte => todo!("Byte cdq!"),
            },
            assembly_ast::Instruction::IDiv(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(output, "    idiv{t} {operand}")?;
            }
            assembly_ast::Instruction::Div(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(output, "    div{t} {operand}")?;
            }
            assembly_ast::Instruction::Binary(operator, left, right, asm_type) => {
                let src = get_operand(left, asm_type);
                let dst = get_operand(right, asm_type);
                // Xor and Mult have special impls for double
                if *asm_type == AssemblyType::Double && *operator == BinaryOperator::BitXor {
                    writeln!(output, "    xorpd {src}, {dst}")?;
                } else if *asm_type == AssemblyType::Double && *operator == BinaryOperator::Mult {
                    writeln!(output, "    mulsd {src}, {dst}")?;
                } else {
                    let operator = get_binary_operator(operator);
                    let t = get_size_suffix(asm_type);
                    writeln!(output, "    {operator}{t} {src}, {dst}")?;
                }
            }
            assembly_ast::Instruction::SetCond(cond, val) => {
                let code = get_cond_code(cond);
                let reg = get_short_reg(val);
                writeln!(output, "    set{} {}", code, reg)?;
            }
            assembly_ast::Instruction::Compare(left, right, cmp_type) => {
                let left = get_operand(left, cmp_type);
                let right = get_operand(right, cmp_type);
                let t = get_size_suffix(cmp_type);
                if *cmp_type == AssemblyType::Double {
                    writeln!(output, "    comisd {left}, {right}")?;
                } else {
                    writeln!(output, "    cmp{t} {left}, {right}")?;
                }
            }
            assembly_ast::Instruction::JmpCond(cond, target) => {
                writeln!(output, "    j{} .L_{}", get_cond_code(cond), target)?;
            }
            assembly_ast::Instruction::Jmp(target) => {
                writeln!(output, "    jmp .L_{}", target)?;
            }
            assembly_ast::Instruction::Label(target) => {
                writeln!(output, ".L_{}:", target)?;
            }
            assembly_ast::Instruction::Push(operand) => {
                writeln!(output, "    pushq {}", get_operand_quadword(operand))?;
            }
            assembly_ast::Instruction::Call(label) => {
                if let AsmSymbol::FunctionEntry(true) = &symbols[label] {
                    writeln!(output, "    call {}", label)?;
                } else {
                    writeln!(output, "    call {}@PLT", label)?;
                }
            }
            assembly_ast::Instruction::Cvttsd2si(src, dst, asm_type) => {
                let src = get_operand(src, asm_type);
                let dst = get_operand(dst, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(output, "    cvttsd2si{t} {src}, {dst}")?;
            }
            assembly_ast::Instruction::Cvtsi2sd(src, dst, asm_type) => {
                let src = get_operand(src, asm_type);
                let dst = get_operand(dst, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(output, "    cvtsi2sd{t} {src}, {dst}")?;
            }
            assembly_ast::Instruction::Lea(src, dst) => {
                let src = get_operand_quadword(src);
                let dst = get_operand_quadword(dst);
                writeln!(output, "    leaq {src}, {dst}")?;
            }
            assembly_ast::Instruction::Nop => panic!("Noop should just be a placeholder"),
        }
    }
    Ok(())
}

fn get_operand_quadword(operand: &assembly_ast::Operand) -> String {
    use assembly_ast::Operand::*;
    use assembly_ast::Register::*;
    match operand {
        Register(AX) => String::from("%rax"),
        Register(CX) => String::from("%rcx"),
        Register(DX) => String::from("%rdx"),
        Register(DI) => String::from("%rdi"),
        Register(SI) => String::from("%rsi"),
        Register(R8) => String::from("%r8"),
        Register(R9) => String::from("%r9"),
        Register(R10) => String::from("%r10"),
        Register(R11) => String::from("%r11"),
        Register(SP) => String::from("%rsp"),
        Register(BP) => String::from("%rbp"),
        Register(CL) => String::from("%cl"),
        Indexed(reg1, reg2, loc) => {
            let reg1 = get_operand_quadword(&Register(*reg1));
            let reg2 = get_operand_quadword(&Register(*reg2));
            format!("({reg1}, {reg2}, {loc})")
        }
        _ => get_operand_longword(operand),
    }
}
fn get_operand_longword(operand: &assembly_ast::Operand) -> String {
    use assembly_ast::Operand::*;
    use assembly_ast::Register::*;
    match operand {
        Imm(val) => format!("${val}"),
        // x86 int registers
        Register(AX) => String::from("%eax"),
        Register(CX) => String::from("%ecx"),
        Register(DX) => String::from("%edx"),
        Register(DI) => String::from("%edi"),
        Register(SI) => String::from("%esi"),
        Register(R8) => String::from("%r8d"),
        Register(R9) => String::from("%r9d"),
        Register(R10) => String::from("%r10d"),
        Register(R11) => String::from("%r11d"),
        Register(SP) => String::from("%esp"),
        Register(BP) => String::from("%ebp"),
        //Double Registers
        Register(XMM0) => String::from("%xmm0"),
        Register(XMM1) => String::from("%xmm1"),
        Register(XMM2) => String::from("%xmm2"),
        Register(XMM3) => String::from("%xmm3"),
        Register(XMM4) => String::from("%xmm4"),
        Register(XMM5) => String::from("%xmm5"),
        Register(XMM6) => String::from("%xmm6"),
        Register(XMM7) => String::from("%xmm7"),
        Register(XMM14) => String::from("%xmm14"),
        Register(XMM15) => String::from("%xmm15"),
        // Short reg for shifts
        Register(CL) => String::from("%cl"),
        Memory(reg, val) => {
            let reg = get_operand_quadword(&Register(*reg));
            format!("{val}({reg})")
        }
        Data(name) => format!("{name}(%rip)"),
        Indexed(reg1, reg2, loc) => {
            let reg1 = get_operand_longword(&Register(*reg1));
            let reg2 = get_operand_longword(&Register(*reg2));
            format!("({reg1}, {reg2}, {loc})")
        }
    }
}

fn get_operand(operand: &assembly_ast::Operand, asm_type: &AssemblyType) -> String {
    match asm_type {
        AssemblyType::Longword => get_operand_longword(operand),
        AssemblyType::Quadword | AssemblyType::Double => get_operand_quadword(operand),
        AssemblyType::ByteArray(_, _) => get_operand_longword(operand),
        AssemblyType::Byte => get_short_reg(operand),
    }
}

fn get_short_reg(operand: &assembly_ast::Operand) -> String {
    match operand {
        assembly_ast::Operand::Register(assembly_ast::Register::AX) => String::from("%al"),
        assembly_ast::Operand::Register(assembly_ast::Register::CX) => String::from("%cl"),
        assembly_ast::Operand::Register(assembly_ast::Register::DX) => String::from("%dl"),
        assembly_ast::Operand::Register(assembly_ast::Register::DI) => String::from("%dil"),
        assembly_ast::Operand::Register(assembly_ast::Register::SI) => String::from("%sil"),
        assembly_ast::Operand::Register(assembly_ast::Register::R8) => String::from("%r8b"),
        assembly_ast::Operand::Register(assembly_ast::Register::R9) => String::from("%r9b"),
        assembly_ast::Operand::Register(assembly_ast::Register::R10) => String::from("%r10b"),
        assembly_ast::Operand::Register(assembly_ast::Register::R11) => String::from("%r11b"),
        assembly_ast::Operand::Register(assembly_ast::Register::CL) => String::from("%cl"),
        assembly_ast::Operand::Register(assembly_ast::Register::SP) => String::from("%rsp"),
        _ => get_operand_longword(operand),
    }
}

fn get_cond_code(condition: &assembly_ast::Condition) -> &str {
    match condition {
        Condition::Equal => "e",
        Condition::NotEqual => "ne",
        Condition::GreaterThan => "g",
        Condition::GreaterThanEqual => "ge",
        Condition::LessThan => "l",
        Condition::LessThanEqual => "le",
        Condition::UnsignedGreaterThan => "a",
        Condition::UnsignedGreaterEqual => "ae",
        Condition::UnsignedLessThan => "b",
        Condition::UnsignedLessEqual => "be",
        Condition::Parity => "p",
    }
}

fn get_unary_operator(operator: &assembly_ast::UnaryOperator) -> &str {
    match operator {
        assembly_ast::UnaryOperator::Neg => "neg",
        assembly_ast::UnaryOperator::Not => "not",
        assembly_ast::UnaryOperator::Shr => "shr",
    }
}

fn get_binary_operator(operator: &assembly_ast::BinaryOperator) -> &str {
    match operator {
        assembly_ast::BinaryOperator::Add => "add",
        assembly_ast::BinaryOperator::Sub => "sub",
        assembly_ast::BinaryOperator::Mult => "imul",
        assembly_ast::BinaryOperator::DoubleDiv => "div",
        assembly_ast::BinaryOperator::BitAnd => "and",
        assembly_ast::BinaryOperator::BitOr => "or",
        assembly_ast::BinaryOperator::BitXor => "xor",
        assembly_ast::BinaryOperator::LeftShift => "sal",
        assembly_ast::BinaryOperator::LeftShiftUnsigned => "shl",
        assembly_ast::BinaryOperator::RightShift => "sar",
        assembly_ast::BinaryOperator::RightShiftUnsigned => "shr",
    }
}

fn get_size_suffix(asm_type: &AssemblyType) -> &str {
    match asm_type {
        AssemblyType::Byte => "b",
        AssemblyType::Longword => "l",
        AssemblyType::Quadword => "q",
        AssemblyType::Double => "sd",
        AssemblyType::ByteArray(_, _) => panic!("Expected var, found array!"),
    }
}
