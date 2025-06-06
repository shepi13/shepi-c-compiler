use super::assembly_gen::{
    self, AsmSymbol, AssemblyType, BackendSymbols, Condition, Function, StaticConstant, StaticVar,
};
use crate::{codegen::assembly_gen::BinaryOperator, validate::type_check::Initializer};
use std::error::Error;

type EmptyResult = Result<(), Box<dyn Error>>;

pub fn emit_program<T>(output: &mut T, program: assembly_gen::Program) -> EmptyResult
where
    T: std::fmt::Write,
{
    for top_level in program.program {
        match top_level {
            assembly_gen::TopLevelDecl::FunctionDecl(function) => {
                emit_function(output, function, &program.backend_symbols)?;
            }
            assembly_gen::TopLevelDecl::Var(var) => emit_static_var(output, var)?,
            assembly_gen::TopLevelDecl::Constant(var) => emit_static_const(output, var)?,
        }
    }
    writeln!(output, "    .section .note.GNU-stack,\"\",@progbits")?;
    Ok(())
}

fn emit_static_var<T>(output: &mut T, var: StaticVar) -> EmptyResult
where
    T: std::fmt::Write,
{
    use Initializer::*;
    if var.global {
        writeln!(output, "    .globl {}", var.name)?;
    }
    if var.init.len() == 1 && matches!(var.init[0], ZeroInit(_)) {
        writeln!(output, "    .bss")?;
    } else {
        writeln!(output, "    .data")?;
    }
    writeln!(output, "    .align {}", var.alignment)?;
    writeln!(output, "{}:", var.name)?;
    for init in var.init {
        let init_str = match init {
            Int(_) | UnsignedInt(_) => format!(".long {}", init.int_value()),
            Long(_) | UnsignedLong(_) => format!(".quad {}", init.int_value()),
            Double(val) => format!(".double {}", val),
            ZeroInit(size) => format!(".zero {size}"),
        };
        writeln!(output, "    {}", init_str)?;
    }
    Ok(())
}

fn emit_static_const<T>(output: &mut T, var: StaticConstant) -> EmptyResult
where
    T: std::fmt::Write,
{
    writeln!(output, "    .section .rodata")?;
    writeln!(output, "    .align {}", var.alignment)?;
    writeln!(output, "{}:", var.name)?;
    match var.init {
        Initializer::Double(val) => {
            writeln!(output, "    .double {val}")?;
        }
        _ => panic!("Not a double"),
    }
    Ok(())
}

fn emit_function<T>(output: &mut T, function: Function, symbols: &BackendSymbols) -> EmptyResult
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
    instructions: &Vec<assembly_gen::Instruction>,
    symbols: &BackendSymbols,
) -> EmptyResult {
    for instruction in instructions {
        match instruction {
            assembly_gen::Instruction::Mov(src, dst, move_type) => {
                let src = get_operand(src, move_type);
                let dst = get_operand(dst, move_type);
                let t = get_size_suffix(move_type);
                writeln!(output, "    mov{t} {src}, {dst}")?;
            }
            assembly_gen::Instruction::MovSignExtend(src, dst) => {
                let src = get_operand(src, &AssemblyType::Longword);
                let dst = get_operand(dst, &AssemblyType::Quadword);
                writeln!(output, "    movslq {src}, {dst}")?;
            }
            assembly_gen::Instruction::Ret => {
                writeln!(output, "    movq %rbp, %rsp")?;
                writeln!(output, "    popq %rbp")?;
                writeln!(output, "    ret")?;
            }
            assembly_gen::Instruction::Unary(operator, operand, asm_type) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(output, "    {operator}{t} {operand}")?;
            }
            assembly_gen::Instruction::Cdq(asm_type) => match asm_type {
                AssemblyType::Longword => writeln!(output, "    cdq")?,
                AssemblyType::Quadword => writeln!(output, "    cqo")?,
                AssemblyType::Double => panic!("Cdq not supported for double!"),
                AssemblyType::ByteArray(_, _) => panic!("Cdq not supported for array!"),
            },
            assembly_gen::Instruction::IDiv(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(output, "    idiv{t} {operand}")?;
            }
            assembly_gen::Instruction::Div(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(output, "    div{t} {operand}")?;
            }
            assembly_gen::Instruction::Binary(operator, left, right, asm_type) => {
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
            assembly_gen::Instruction::SetCond(cond, val) => {
                let code = get_cond_code(cond);
                let reg = get_short_reg(val);
                writeln!(output, "    set{} {}", code, reg)?;
            }
            assembly_gen::Instruction::Compare(left, right, cmp_type) => {
                let left = get_operand(left, cmp_type);
                let right = get_operand(right, cmp_type);
                let t = get_size_suffix(cmp_type);
                if *cmp_type == AssemblyType::Double {
                    writeln!(output, "    comisd {left}, {right}")?;
                } else {
                    writeln!(output, "    cmp{t} {left}, {right}")?;
                }
            }
            assembly_gen::Instruction::JmpCond(cond, target) => {
                writeln!(output, "    j{} .L_{}", get_cond_code(cond), target)?;
            }
            assembly_gen::Instruction::Jmp(target) => {
                writeln!(output, "    jmp .L_{}", target)?;
            }
            assembly_gen::Instruction::Label(target) => {
                writeln!(output, ".L_{}:", target)?;
            }
            assembly_gen::Instruction::Push(operand) => {
                writeln!(output, "    pushq {}", get_operand_quadword(operand))?;
            }
            assembly_gen::Instruction::Call(label) => {
                if let AsmSymbol::FunctionEntry(true) = &symbols[label] {
                    writeln!(output, "    call {}", label)?;
                } else {
                    writeln!(output, "    call {}@PLT", label)?;
                }
            }
            assembly_gen::Instruction::MovZeroExtend(_, _) => {
                panic!("Should be replaced with Mov instructions by rewriter!")
            }
            assembly_gen::Instruction::Cvttsd2si(src, dst, asm_type) => {
                let src = get_operand(src, asm_type);
                let dst = get_operand(dst, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(output, "    cvttsd2si{t} {src}, {dst}")?;
            }
            assembly_gen::Instruction::Cvtsi2sd(src, dst, asm_type) => {
                let src = get_operand(src, asm_type);
                let dst = get_operand(dst, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(output, "    cvtsi2sd{t} {src}, {dst}")?;
            }
            assembly_gen::Instruction::Lea(src, dst) => {
                let src = get_operand_quadword(src);
                let dst = get_operand_quadword(dst);
                writeln!(output, "    leaq {src}, {dst}")?;
            }
            assembly_gen::Instruction::Nop => panic!("Noop should just be a placeholder"),
        }
    }
    Ok(())
}

fn get_operand_quadword(operand: &assembly_gen::Operand) -> String {
    use assembly_gen::Operand::*;
    use assembly_gen::Register::*;
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
fn get_operand_longword(operand: &assembly_gen::Operand) -> String {
    use assembly_gen::Operand::*;
    use assembly_gen::Register::*;
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

fn get_operand(operand: &assembly_gen::Operand, asm_type: &AssemblyType) -> String {
    match asm_type {
        AssemblyType::Longword => get_operand_longword(operand),
        AssemblyType::Quadword | AssemblyType::Double => get_operand_quadword(operand),
        AssemblyType::ByteArray(_, _) => get_operand_longword(operand),
    }
}

fn get_short_reg(operand: &assembly_gen::Operand) -> String {
    match operand {
        assembly_gen::Operand::Register(assembly_gen::Register::AX) => String::from("%al"),
        assembly_gen::Operand::Register(assembly_gen::Register::CX) => String::from("%cl"),
        assembly_gen::Operand::Register(assembly_gen::Register::DX) => String::from("%dl"),
        assembly_gen::Operand::Register(assembly_gen::Register::DI) => String::from("%dil"),
        assembly_gen::Operand::Register(assembly_gen::Register::SI) => String::from("%sil"),
        assembly_gen::Operand::Register(assembly_gen::Register::R8) => String::from("%r8b"),
        assembly_gen::Operand::Register(assembly_gen::Register::R9) => String::from("%r9b"),
        assembly_gen::Operand::Register(assembly_gen::Register::R10) => String::from("%r10b"),
        assembly_gen::Operand::Register(assembly_gen::Register::R11) => String::from("%r11b"),
        assembly_gen::Operand::Register(assembly_gen::Register::CL) => String::from("%cl"),
        assembly_gen::Operand::Register(assembly_gen::Register::SP) => String::from("%rsp"),
        _ => get_operand_longword(operand),
    }
}

fn get_cond_code(condition: &assembly_gen::Condition) -> &str {
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

fn get_unary_operator(operator: &assembly_gen::UnaryOperator) -> &str {
    match operator {
        assembly_gen::UnaryOperator::Neg => "neg",
        assembly_gen::UnaryOperator::Not => "not",
        assembly_gen::UnaryOperator::Shr => "shr",
    }
}

fn get_binary_operator(operator: &assembly_gen::BinaryOperator) -> &str {
    match operator {
        assembly_gen::BinaryOperator::Add => "add",
        assembly_gen::BinaryOperator::Sub => "sub",
        assembly_gen::BinaryOperator::Mult => "imul",
        assembly_gen::BinaryOperator::DoubleDiv => "div",
        assembly_gen::BinaryOperator::BitAnd => "and",
        assembly_gen::BinaryOperator::BitOr => "or",
        assembly_gen::BinaryOperator::BitXor => "xor",
        assembly_gen::BinaryOperator::LeftShift => "sal",
        assembly_gen::BinaryOperator::LeftShiftUnsigned => "shl",
        assembly_gen::BinaryOperator::RightShift => "sar",
        assembly_gen::BinaryOperator::RightShiftUnsigned => "shr",
    }
}

fn get_size_suffix(asm_type: &AssemblyType) -> &str {
    match asm_type {
        AssemblyType::Longword => "l",
        AssemblyType::Quadword => "q",
        AssemblyType::Double => "sd",
        AssemblyType::ByteArray(_, _) => panic!("Expected var, found array!"),
    }
}
