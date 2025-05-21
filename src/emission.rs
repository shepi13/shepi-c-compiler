use crate::{
    assembly::{self, Condition},
    generator::StaticVariable,
    type_check::{SymbolAttr, Symbols},
};
use std::{fs::File, io::Write};

pub fn emit_program(output_filename: &str, program: assembly::Program, symbols: &Symbols) {
    let mut file = File::create(output_filename).expect("Failed to create file!");
    for top_level in program {
        match top_level {
            assembly::TopLevelDecl::FUNCTION(function) => {
                emit_function(&mut file, function, symbols);
            }
            assembly::TopLevelDecl::STATICVAR(var) => emit_static_var(&mut file, var),
        }
    }
    writeln!(file, "    .section .note.GNU-stack,\"\",@progbits").unwrap();
}
fn emit_static_var(file: &mut File, var: StaticVariable) {
    if var.global {
        writeln!(file, "    .globl {}", var.identifier).unwrap();
    }
    let (section, data_type) = if var.initializer == 0 {
        (".bss", ".zero 4".to_string())
    } else {
        (".data", format!(".long {}", var.initializer))
    };
    writeln!(file, "    {}", section).unwrap();
    writeln!(file, "    .align 4").unwrap();
    writeln!(file, "{}:", var.identifier).unwrap();
    writeln!(file, "    {}", data_type).unwrap();
}
fn emit_function(file: &mut File, function: assembly::Function, symbols: &Symbols) {
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
    symbols: &Symbols,
) {
    for instruction in instructions {
        match instruction {
            assembly::Instruction::Mov(src, dst) => {
                let src = get_operand(src);
                let dst = get_operand(dst);
                writeln!(file, "    movl {src}, {dst}").unwrap();
            }
            assembly::Instruction::Ret => {
                writeln!(file, "    movq %rbp, %rsp").unwrap();
                writeln!(file, "    popq %rbp").unwrap();
                writeln!(file, "    ret").unwrap();
            }
            assembly::Instruction::StackAllocate(val) => {
                writeln!(file, "    subq ${}, %rsp", val).unwrap();
            }
            assembly::Instruction::StackDeallocate(val) => {
                writeln!(file, "    addq ${}, %rsp", val).unwrap();
            }
            assembly::Instruction::Unary(operator, operand) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand);
                writeln!(file, "    {} {}", operator, operand).unwrap();
            }
            assembly::Instruction::Cdq => {
                writeln!(file, "    cdq").unwrap();
            }
            assembly::Instruction::IDiv(operand) => {
                writeln!(file, "    idivl {}", get_operand(operand)).unwrap();
            }
            assembly::Instruction::Binary(operator, left, right) => {
                let operator = get_binary_operator(operator);
                let src1 = get_operand(left);
                let src2 = get_operand(right);
                writeln!(file, "    {} {}, {}", operator, src1, src2).unwrap();
            }
            assembly::Instruction::SetCond(cond, val) => {
                let code = get_cond_code(cond);
                let reg = get_short_reg(val);
                writeln!(file, "    set{} {}", code, reg).unwrap();
            }
            assembly::Instruction::Compare(left, right) => {
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
                if let SymbolAttr::Function(attrs) = &symbols[label].attrs {
                    if attrs.defined {
                        writeln!(file, "    call {}", label).unwrap();
                    } else {
                        writeln!(file, "    call {}@PLT", label).unwrap();
                    }
                } else {
                    panic!("Expected function attributes!")
                }
            }
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
