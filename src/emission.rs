use crate::{
    assembly::{self, AsmSymbol, AssemblyType, BackendSymbols, Condition, StaticVar},
    type_check::Initializer,
};
use std::{fs::File, io::Write};

pub fn emit_program(output_filename: &str, program: assembly::Program) {
    let mut file = File::create(output_filename).expect("Failed to create file!");
    for top_level in program.program {
        match top_level {
            assembly::TopLevelDecl::FUNCTION(function) => {
                emit_function(&mut file, function, &program.backend_symbols);
            }
            assembly::TopLevelDecl::STATICVAR(var) => emit_static_var(&mut file, var),
        }
    }
    writeln!(file, "    .section .note.GNU-stack,\"\",@progbits").unwrap();
}
fn emit_static_var(file: &mut File, var: StaticVar) {
    use Initializer::*;
    if var.global {
        writeln!(file, "    .globl {}", var.name).unwrap();
    }
    let init = match var.init {
        Int(0) | UnsignedInt(0) => String::from(".zero 4"),
        Long(0) | UnsignedLong(0) => String::from(".zero 8"),
        Int(_) | UnsignedInt(_) => format!(".long {}", var.init.value()),
        Long(_) | UnsignedLong(_) => format!(".quad {}", var.init.value()),
    };
    if var.init.value() == 0 {
        writeln!(file, "    .bss").unwrap();
    } else {
        writeln!(file, "    .data").unwrap();
    }
    writeln!(file, "    .align {}", var.alignment).unwrap();
    writeln!(file, "{}:", var.name).unwrap();
    writeln!(file, "    {}", init).unwrap();
}
fn emit_function(file: &mut File, function: assembly::Function, symbols: &BackendSymbols) {
    if function.global {
        writeln!(file, "    .globl {}", function.name).unwrap()
    }
    writeln!(file, "    .text").unwrap();
    writeln!(file, "{}:", function.name).unwrap();
    writeln!(file, "    pushq %rbp").unwrap();
    writeln!(file, "    movq %rsp, %rbp").unwrap();
    emit_instructions(file, &function.instructions, symbols);
}
fn emit_instructions(
    file: &mut File,
    instructions: &Vec<assembly::Instruction>,
    symbols: &BackendSymbols,
) {
    for instruction in instructions {
        match instruction {
            assembly::Instruction::Mov(src, dst, move_type) => {
                let src = get_operand(src, move_type);
                let dst = get_operand(dst, move_type);
                let t = get_size_suffix(move_type);
                writeln!(file, "    mov{t} {src}, {dst}").unwrap();
            }
            assembly::Instruction::MovSignExtend(src, dst) => {
                let src = get_operand(src, &AssemblyType::Longword);
                let dst = get_operand(dst, &AssemblyType::Quadword);
                writeln!(file, "    movslq {src}, {dst}").unwrap();
            }
            assembly::Instruction::Ret => {
                writeln!(file, "    movq %rbp, %rsp").unwrap();
                writeln!(file, "    popq %rbp").unwrap();
                writeln!(file, "    ret").unwrap();
            }
            assembly::Instruction::Unary(operator, operand, asm_type) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(file, "    {operator}{t} {operand}").unwrap();
            }
            assembly::Instruction::Cdq(asm_type) => match asm_type {
                AssemblyType::Longword => writeln!(file, "    cdq").unwrap(),
                AssemblyType::Quadword => writeln!(file, "    cqo").unwrap(),
            },
            assembly::Instruction::IDiv(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(file, "    idiv{t} {operand}").unwrap();
            }
            assembly::Instruction::Div(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(file, "    div{t} {operand}").unwrap();
            }
            assembly::Instruction::Binary(operator, left, right, asm_type) => {
                let operator = get_binary_operator(operator);
                let src = get_operand(left, asm_type);
                let dst = get_operand(right, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(file, "    {operator}{t} {src}, {dst}").unwrap();
            }
            assembly::Instruction::SetCond(cond, val) => {
                let code = get_cond_code(cond);
                let reg = get_short_reg(val);
                writeln!(file, "    set{} {}", code, reg).unwrap();
            }
            assembly::Instruction::Compare(left, right, cmp_type) => {
                let left = get_operand(left, cmp_type);
                let right = get_operand(right, cmp_type);
                let t = get_size_suffix(cmp_type);
                writeln!(file, "    cmp{t} {left}, {right}").unwrap();
            }
            assembly::Instruction::JmpCond(cond, target) => {
                writeln!(file, "    j{} .L_{}", get_cond_code(cond), target).unwrap();
            }
            assembly::Instruction::Jmp(target) => {
                writeln!(file, "    jmp .L_{}", target).unwrap();
            }
            assembly::Instruction::Label(target) => {
                writeln!(file, ".L_{}:", target).unwrap();
            }
            assembly::Instruction::Push(operand) => {
                writeln!(file, "    pushq {}", get_operand_quadword(operand)).unwrap();
            }
            assembly::Instruction::Call(label) => {
                if let AsmSymbol::FunctionEntry(true) = &symbols[label] {
                    writeln!(file, "    call {}", label).unwrap();
                } else {
                    writeln!(file, "    call {}@PLT", label).unwrap();
                }
            }
            assembly::Instruction::MovZeroExtend(_, _) => {
                panic!("Should be replaced with Mov instructions by rewriter!")
            }
            assembly::Instruction::NOP => panic!("Noop should just be a placeholder"),
        }
    }
}

fn get_operand_quadword(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::Register(assembly::Register::AX) => String::from("%rax"),
        assembly::Operand::Register(assembly::Register::CX) => String::from("%rcx"),
        assembly::Operand::Register(assembly::Register::DX) => String::from("%rdx"),
        assembly::Operand::Register(assembly::Register::DI) => String::from("%rdi"),
        assembly::Operand::Register(assembly::Register::SI) => String::from("%rsi"),
        assembly::Operand::Register(assembly::Register::R8) => String::from("%r8"),
        assembly::Operand::Register(assembly::Register::R9) => String::from("%r9"),
        assembly::Operand::Register(assembly::Register::R10) => String::from("%r10"),
        assembly::Operand::Register(assembly::Register::R11) => String::from("%r11"),
        assembly::Operand::Register(assembly::Register::SP) => String::from("%rsp"),
        assembly::Operand::Register(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::Stack(val) => format!("-{val}(%rbp)"),
        assembly::Operand::Data(name) => format!("{name}(%rip)"),
    }
}
fn get_operand_longword(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::Register(assembly::Register::AX) => String::from("%eax"),
        assembly::Operand::Register(assembly::Register::CX) => String::from("%ecx"),
        assembly::Operand::Register(assembly::Register::DX) => String::from("%edx"),
        assembly::Operand::Register(assembly::Register::DI) => String::from("%edi"),
        assembly::Operand::Register(assembly::Register::SI) => String::from("%esi"),
        assembly::Operand::Register(assembly::Register::R8) => String::from("%r8d"),
        assembly::Operand::Register(assembly::Register::R9) => String::from("%r9d"),
        assembly::Operand::Register(assembly::Register::R10) => String::from("%r10d"),
        assembly::Operand::Register(assembly::Register::R11) => String::from("%r11d"),
        assembly::Operand::Register(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::Register(assembly::Register::SP) => String::from("%rsp"),
        assembly::Operand::Stack(val) => format!("-{val}(%rbp)"),
        assembly::Operand::Data(name) => format!("{name}(%rip)"),
    }
}

fn get_operand(operand: &assembly::Operand, asm_type: &AssemblyType) -> String {
    match asm_type {
        AssemblyType::Longword => get_operand_longword(operand),
        AssemblyType::Quadword => get_operand_quadword(operand),
    }
}

fn get_short_reg(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::Register(assembly::Register::AX) => String::from("%al"),
        assembly::Operand::Register(assembly::Register::CX) => String::from("%cl"),
        assembly::Operand::Register(assembly::Register::DX) => String::from("%dl"),
        assembly::Operand::Register(assembly::Register::DI) => String::from("%dil"),
        assembly::Operand::Register(assembly::Register::SI) => String::from("%sil"),
        assembly::Operand::Register(assembly::Register::R8) => String::from("%r8b"),
        assembly::Operand::Register(assembly::Register::R9) => String::from("%r9b"),
        assembly::Operand::Register(assembly::Register::R10) => String::from("%r10b"),
        assembly::Operand::Register(assembly::Register::R11) => String::from("%r11b"),
        assembly::Operand::Register(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::Register(assembly::Register::SP) => String::from("%rsp"),
        assembly::Operand::Stack(val) => format!("-{val}(%rbp)"),
        assembly::Operand::Data(name) => format!("{name}(%rip)"),
    }
}

fn get_cond_code(condition: &assembly::Condition) -> &str {
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
    }
}

fn get_unary_operator(operator: &assembly::UnaryOperator) -> &str {
    match operator {
        assembly::UnaryOperator::Neg => "neg",
        assembly::UnaryOperator::Not => "not",
    }
}

fn get_binary_operator(operator: &assembly::BinaryOperator) -> &str {
    match operator {
        assembly::BinaryOperator::Add => "add",
        assembly::BinaryOperator::Sub => "sub",
        assembly::BinaryOperator::Mult => "imul",
        assembly::BinaryOperator::BitAnd => "and",
        assembly::BinaryOperator::BitOr => "or",
        assembly::BinaryOperator::BitXor => "xor",
        assembly::BinaryOperator::LeftShift => "sal",
        assembly::BinaryOperator::LeftShiftUnsigned => "shl",
        assembly::BinaryOperator::RightShift => "sar",
        assembly::BinaryOperator::RightShiftUnsigned => "shr",
    }
}

fn get_size_suffix(asm_type: &AssemblyType) -> &str {
    match asm_type {
        AssemblyType::Longword => "l",
        AssemblyType::Quadword => "q",
    }
}
