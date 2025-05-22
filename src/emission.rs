use crate::{
    assembly::{self, AsmSymbol, AssemblyType, BackendSymbols, Condition, StaticVar},
};
use std::{fs::File, io::Write};

pub fn emit_program(output_filename: &str, program: assembly::Program) {
    let mut file = File::create(output_filename).expect("Failed to create file!");
    for top_level in program.program {
        match top_level {
            assembly::TopLevelDecl::FUNCTION(function) => {
                emit_function(&mut file, function, &program.backend_symbols);
            }
            assembly::TopLevelDecl::STATICVAR(var) => emit_static_var(&mut file, var, &program.backend_symbols),
        }
    }
    writeln!(file, "    .section .note.GNU-stack,\"\",@progbits").unwrap();
}
fn emit_static_var(file: &mut File, var: StaticVar, symbols: &BackendSymbols) {
    if var.global {
        writeln!(file, "    .globl {}", var.name).unwrap();
    }
    let initializer = var.init.value();
    let (section, data_type) = match initializer {
        Some(0) => (".bss", ".zero 4".to_string()),
        Some(val) => (".data", format!(".long {}", val)),
        None => panic!("No static initializer!"),
    };
    writeln!(file, "    {}", section).unwrap();
    writeln!(file, "    .align 4").unwrap();
    writeln!(file, "{}:", var.name).unwrap();
    writeln!(file, "    {}", data_type).unwrap();
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
            assembly::Instruction::Mov(src, dst, _) => {
                let src = get_operand(src);
                let dst = get_operand(dst);
                writeln!(file, "    movl {src}, {dst}").unwrap();
            }
            assembly::Instruction::Ret => {
                writeln!(file, "    movq %rbp, %rsp").unwrap();
                writeln!(file, "    popq %rbp").unwrap();
                writeln!(file, "    ret").unwrap();
            }
            assembly::Instruction::Unary(operator, operand, _) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand);
                writeln!(file, "    {} {}", operator, operand).unwrap();
            }
            assembly::Instruction::Cdq(_) => {
                writeln!(file, "    cdq").unwrap();
            }
            assembly::Instruction::IDiv(operand, _) => {
                writeln!(file, "    idivl {}", get_operand(operand)).unwrap();
            }
            assembly::Instruction::Binary(operator, left, right, asm_type) => {
                let mut operator = get_binary_operator(operator);
                if *asm_type == AssemblyType::Quadword {
                    match operator {
                        "subl" => {operator = "subq"},
                        "addl" => {operator = "addq"},
                        _ => ()
                    }
                }
                let src1 = get_operand(left);
                let src2 = get_operand(right);
                writeln!(file, "    {} {}, {}", operator, src1, src2).unwrap();
            }
            assembly::Instruction::SetCond(cond, val) => {
                let code = get_cond_code(cond);
                let reg = get_short_reg(val);
                writeln!(file, "    set{} {}", code, reg).unwrap();
            }
            assembly::Instruction::Compare(left, right, _) => {
                let left = get_operand(left);
                let right = get_operand(right);
                writeln!(file, "    cmpl {}, {}", left, right).unwrap();
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
                writeln!(file, "    pushq {}", get_long_reg(operand)).unwrap();
            }
            assembly::Instruction::Call(label) => {
                if let AsmSymbol::FunctionEntry(true) = &symbols[label] {
                    writeln!(file, "    call {}", label).unwrap();
                } else {
                    writeln!(file, "    call {}@PLT", label).unwrap();
                }
            }
            _ => panic!("Not implemented!")
        }
    }
}

fn get_long_reg(operand: &assembly::Operand) -> String {
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

fn get_operand(operand: &assembly::Operand) -> String {
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
    }
}

fn get_unary_operator(operator: &assembly::UnaryOperator) -> &str {
    match operator {
        assembly::UnaryOperator::Neg => "negl",
        assembly::UnaryOperator::Not => "notl",
    }
}

fn get_binary_operator(operator: &assembly::BinaryOperator) -> &str {
    match operator {
        assembly::BinaryOperator::Add => "addl",
        assembly::BinaryOperator::Sub => "subl",
        assembly::BinaryOperator::Mult => "imull",
        assembly::BinaryOperator::BitAnd => "andl",
        assembly::BinaryOperator::BitOr => "orl",
        assembly::BinaryOperator::BitXor => "xorl",
        assembly::BinaryOperator::LeftShift => "sal",
        assembly::BinaryOperator::RightShift => "sar",
    }
}
