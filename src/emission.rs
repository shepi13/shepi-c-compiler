use crate::assembly::{self, Condition};
use std::{fs::File, io::Write};

pub struct CodeEmission {
    file: File,
}

impl CodeEmission {
    pub fn from(output_filename: &str) -> CodeEmission {
        let file = File::create(output_filename).expect("Failed to create file!");
        CodeEmission { file }
    }
    pub fn emit(mut self, program: assembly::Program) {
        self.emit_function(&program.main);
        writeln!(
            &mut self.file,
            "    .section .note.GNU-stack,\"\",@progbits"
        )
        .expect("Write Failed!");
    }

    fn emit_function(&mut self, function: &assembly::Function) {
        writeln!(&mut self.file, "    .globl {}", function.name).expect("Write failed!");
        writeln!(&mut self.file, "{}:", function.name).expect("Write failed!");
        writeln!(&mut self.file, "    pushq %rbp").expect("Write failed!");
        writeln!(&mut self.file, "    movq %rsp, %rbp").expect("Write failed!");
        self.emit_instructions(&function.instructions);
    }
    fn emit_instructions(&mut self, instructions: &Vec<assembly::Instruction>) {
        for instruction in instructions {
            match instruction {
                assembly::Instruction::MOV(src, dst) => {
                    let src = get_operand(src);
                    let dst = get_operand(dst);
                    writeln!(&mut self.file, "    movl {src}, {dst}").expect("Write failed!");
                }
                assembly::Instruction::RET => {
                    writeln!(&mut self.file, "    movq %rbp, %rsp").expect("Write failed!");
                    writeln!(&mut self.file, "    popq %rbp").expect("Write failed!");
                    writeln!(&mut self.file, "    ret").expect("Write failed!");
                }
                assembly::Instruction::STACKALLOCATE(val) => {
                    writeln!(&mut self.file, "    subq ${}, %rsp", val).expect("Write failed!");
                }
                assembly::Instruction::UNARY(operator, operand) => {
                    let operator = get_unary_operator(operator);
                    let operand = get_operand(operand);
                    writeln!(&mut self.file, "    {} {}", operator, operand)
                        .expect("Write failed!");
                }
                assembly::Instruction::CDQ => {
                    writeln!(&mut self.file, "    cdq").expect("Write failed!");
                }
                assembly::Instruction::IDIV(operand) => {
                    writeln!(&mut self.file, "    idivl {}", get_operand(operand))
                        .expect("Write failed!");
                }
                assembly::Instruction::BINARY(operator, left, right) => {
                    let operator = get_binary_operator(operator);
                    let src1 = get_operand(left);
                    let src2 = get_operand(right);
                    writeln!(&mut self.file, "    {} {}, {}", operator, src1, src2)
                        .expect("Write failed!");
                }
                assembly::Instruction::SetCond(cond, val) => {
                    writeln!(
                        &mut self.file,
                        "    set{} {}",
                        get_cond_code(cond),
                        get_short_reg(val)
                    )
                    .expect("Write failed!");
                }
                assembly::Instruction::COMPARE(left, right) => {
                    writeln!(
                        &mut self.file,
                        "    cmpl {}, {}",
                        get_operand(left),
                        get_operand(right)
                    )
                    .expect("Write failed!");
                }
                assembly::Instruction::JMPCond(cond, target) => {
                    writeln!(&mut self.file, "    j{} .L_{}", get_cond_code(cond), target)
                        .expect("Write failed!");
                }
                assembly::Instruction::JMP(target) => {
                    writeln!(&mut self.file, "    jmp .L_{}", target).expect("Write failed!");
                }
                assembly::Instruction::LABEL(target) => {
                    writeln!(&mut self.file, ".L_{}:", target).expect("Write failed!");
                }
            }
        }
    }
}

fn get_operand(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::REGISTER(assembly::Register::AX) => String::from("%eax"),
        assembly::Operand::REGISTER(assembly::Register::CX) => String::from("%ecx"),
        assembly::Operand::REGISTER(assembly::Register::DX) => String::from("%edx"),
        assembly::Operand::REGISTER(assembly::Register::R10) => String::from("%r10d"),
        assembly::Operand::REGISTER(assembly::Register::R11) => String::from("%r11d"),
        assembly::Operand::REGISTER(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::STACK(val) => format!("-{val}(%rbp)"),
    }
}

fn get_short_reg(operand: &assembly::Operand) -> String {
    match operand {
        assembly::Operand::IMM(val) => format!("${val}"),
        assembly::Operand::REGISTER(assembly::Register::AX) => String::from("%al"),
        assembly::Operand::REGISTER(assembly::Register::CX) => String::from("%cl"),
        assembly::Operand::REGISTER(assembly::Register::DX) => String::from("%dl"),
        assembly::Operand::REGISTER(assembly::Register::R10) => String::from("%r10b"),
        assembly::Operand::REGISTER(assembly::Register::R11) => String::from("%r11b"),
        assembly::Operand::REGISTER(assembly::Register::CL) => String::from("%cl"),
        assembly::Operand::STACK(val) => format!("-{val}(%rbp)"),
    }
}

fn get_cond_code(condition: &assembly::Condition) -> String {
    match condition {
        Condition::EQUAL => String::from("e"),
        Condition::NOTEQUAL => String::from("ne"),
        Condition::GREATERTHAN => String::from("g"),
        Condition::GREATERTHANEQUAL => String::from("ge"),
        Condition::LESSTHAN => String::from("l"),
        Condition::LESSTHANEQUAL => String::from("le"),
    }
}

fn get_unary_operator(operator: &assembly::UnaryOperator) -> String {
    match operator {
        assembly::UnaryOperator::NEG => String::from("negl"),
        assembly::UnaryOperator::NOT => String::from("notl"),
    }
}

fn get_binary_operator(operator: &assembly::BinaryOperator) -> String {
    match operator {
        assembly::BinaryOperator::ADD => String::from("addl"),
        assembly::BinaryOperator::SUB => String::from("subl"),
        assembly::BinaryOperator::MULT => String::from("imull"),
        assembly::BinaryOperator::BITAND => String::from("andl"),
        assembly::BinaryOperator::BITOR => String::from("orl"),
        assembly::BinaryOperator::BITXOR => String::from("xorl"),
        assembly::BinaryOperator::LEFTSHIFT => String::from("sal"),
        assembly::BinaryOperator::RIGHTSHIFT => String::from("sar"),
    }
}
