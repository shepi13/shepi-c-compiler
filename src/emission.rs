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
            assembly::Instruction::MOV(src, dst) => {
                let src = get_operand(src);
                let dst = get_operand(dst);
                writeln!(file, "    movl {src}, {dst}").unwrap();
            }
            assembly::Instruction::RET => {
                writeln!(file, "    movq %rbp, %rsp").unwrap();
                writeln!(file, "    popq %rbp").unwrap();
                writeln!(file, "    ret").unwrap();
            }
            assembly::Instruction::STACKALLOCATE(val) => {
                writeln!(file, "    subq ${}, %rsp", val).unwrap();
            }
            assembly::Instruction::STACKDEALLOCATE(val) => {
                writeln!(file, "    addq ${}, %rsp", val).unwrap();
            }
            assembly::Instruction::UNARY(operator, operand) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand);
                writeln!(file, "    {} {}", operator, operand).unwrap();
            }
            assembly::Instruction::CDQ => {
                writeln!(file, "    cdq").unwrap();
            }
            assembly::Instruction::IDIV(operand) => {
                writeln!(file, "    idivl {}", get_operand(operand)).unwrap();
            }
            assembly::Instruction::BINARY(operator, left, right) => {
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
            assembly::Instruction::COMPARE(left, right) => {
                let left = get_operand(left);
                let right = get_operand(right);
                writeln!(file, "    cmpl {}, {}", left, right).unwrap();
            }
            assembly::Instruction::JMPCond(cond, target) => {
                writeln!(file, "    j{} .L_{}", get_cond_code(cond), target).unwrap();
            }
            assembly::Instruction::JMP(target) => {
                writeln!(file, "    jmp .L_{}", target).unwrap();
            }
            assembly::Instruction::LABEL(target) => {
                writeln!(file, ".L_{}:", target).unwrap();
            }
            assembly::Instruction::PUSH(operand) => {
                writeln!(file, "    pushq {}", get_long_reg(operand)).unwrap();
            }
            assembly::Instruction::CALL(label) => {
                if let SymbolAttr::FUNCTION(attrs) = &symbols[label].attrs {
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
        assembly::Operand::REGISTER(assembly::Register::AX) => String::from("%rax"),
        assembly::Operand::REGISTER(assembly::Register::CX) => String::from("%rcx"),
        assembly::Operand::REGISTER(assembly::Register::DX) => String::from("%rdx"),
        assembly::Operand::REGISTER(assembly::Register::DI) => String::from("%rdi"),
        assembly::Operand::REGISTER(assembly::Register::SI) => String::from("%rsi"),
        assembly::Operand::REGISTER(assembly::Register::R8) => String::from("%r8"),
        assembly::Operand::REGISTER(assembly::Register::R9) => String::from("%r9"),
        assembly::Operand::REGISTER(assembly::Register::R10) => String::from("%r10"),
        assembly::Operand::REGISTER(assembly::Register::R11) => String::from("%r11"),
        assembly::Operand::REGISTER(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::STACK(val) => format!("-{val}(%rbp)"),
        assembly::Operand::DATA(name) => format!("{name}(%rip)"),
    }
}

fn get_operand(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::REGISTER(assembly::Register::AX) => String::from("%eax"),
        assembly::Operand::REGISTER(assembly::Register::CX) => String::from("%ecx"),
        assembly::Operand::REGISTER(assembly::Register::DX) => String::from("%edx"),
        assembly::Operand::REGISTER(assembly::Register::DI) => String::from("%edi"),
        assembly::Operand::REGISTER(assembly::Register::SI) => String::from("%esi"),
        assembly::Operand::REGISTER(assembly::Register::R8) => String::from("%r8d"),
        assembly::Operand::REGISTER(assembly::Register::R9) => String::from("%r9d"),
        assembly::Operand::REGISTER(assembly::Register::R10) => String::from("%r10d"),
        assembly::Operand::REGISTER(assembly::Register::R11) => String::from("%r11d"),
        assembly::Operand::REGISTER(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::STACK(val) => format!("-{val}(%rbp)"),
        assembly::Operand::DATA(name) => format!("{name}(%rip)"),
    }
}

fn get_short_reg(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::REGISTER(assembly::Register::AX) => String::from("%al"),
        assembly::Operand::REGISTER(assembly::Register::CX) => String::from("%cl"),
        assembly::Operand::REGISTER(assembly::Register::DX) => String::from("%dl"),
        assembly::Operand::REGISTER(assembly::Register::DI) => String::from("%dil"),
        assembly::Operand::REGISTER(assembly::Register::SI) => String::from("%sil"),
        assembly::Operand::REGISTER(assembly::Register::R8) => String::from("%r8b"),
        assembly::Operand::REGISTER(assembly::Register::R9) => String::from("%r9b"),
        assembly::Operand::REGISTER(assembly::Register::R10) => String::from("%r10b"),
        assembly::Operand::REGISTER(assembly::Register::R11) => String::from("%r11b"),
        assembly::Operand::REGISTER(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::STACK(val) => format!("-{val}(%rbp)"),
        assembly::Operand::DATA(name) => format!("{name}(%rip)"),
    }
}

fn get_cond_code(condition: &assembly::Condition) -> &str {
    match condition {
        Condition::EQUAL => "e",
        Condition::NOTEQUAL => "ne",
        Condition::GREATERTHAN => "g",
        Condition::GREATERTHANEQUAL => "ge",
        Condition::LESSTHAN => "l",
        Condition::LESSTHANEQUAL => "le",
    }
}

fn get_unary_operator(operator: &assembly::UnaryOperator) -> &str {
    match operator {
        assembly::UnaryOperator::NEG => "negl",
        assembly::UnaryOperator::NOT => "notl",
    }
}

fn get_binary_operator(operator: &assembly::BinaryOperator) -> &str {
    match operator {
        assembly::BinaryOperator::ADD => "addl",
        assembly::BinaryOperator::SUB => "subl",
        assembly::BinaryOperator::MULT => "imull",
        assembly::BinaryOperator::BITAND => "andl",
        assembly::BinaryOperator::BITOR => "orl",
        assembly::BinaryOperator::BITXOR => "xorl",
        assembly::BinaryOperator::LEFTSHIFT => "sal",
        assembly::BinaryOperator::RIGHTSHIFT => "sar",
    }
}
