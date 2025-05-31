use super::assembly_gen::{
    self, AsmSymbol, AssemblyType, BackendSymbols, Condition, StaticConstant, StaticVar,
};
use crate::{codegen::assembly_gen::BinaryOperator, validate::type_check::Initializer};
use std::{fs::File, io::Write};

pub fn emit_program(output_filename: &str, program: assembly_gen::Program) {
    let mut file = File::create(output_filename).expect("Failed to create file!");
    for top_level in program.program {
        match top_level {
            assembly_gen::TopLevelDecl::FunctionDecl(function) => {
                emit_function(&mut file, function, &program.backend_symbols);
            }
            assembly_gen::TopLevelDecl::Var(var) => emit_static_var(&mut file, var),
            assembly_gen::TopLevelDecl::Constant(var) => emit_static_const(&mut file, var),
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
        Int(_) | UnsignedInt(_) => format!(".long {}", var.init.int_value()),
        Long(_) | UnsignedLong(_) => format!(".quad {}", var.init.int_value()),
        Double(val) => format!(".double {}", val),
        ZeroInit(size) => format!(".zero {size}"),
    };
    if matches!(var.init, ZeroInit(_)) {
        writeln!(file, "    .bss").unwrap();
    } else {
        writeln!(file, "    .data").unwrap();
    }
    writeln!(file, "    .align {}", var.alignment).unwrap();
    writeln!(file, "{}:", var.name).unwrap();
    writeln!(file, "    {}", init).unwrap();
}

fn emit_static_const(file: &mut File, var: StaticConstant) {
    writeln!(file, "    .section .rodata").unwrap();
    writeln!(file, "    .align {}", var.alignment).unwrap();
    writeln!(file, "{}:", var.name).unwrap();
    match var.init {
        Initializer::Double(val) => {
            writeln!(file, "    .double {val}").unwrap();
        }
        _ => panic!("Not a double"),
    }
}

fn emit_function(file: &mut File, function: assembly_gen::Function, symbols: &BackendSymbols) {
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
    instructions: &Vec<assembly_gen::Instruction>,
    symbols: &BackendSymbols,
) {
    for instruction in instructions {
        match instruction {
            assembly_gen::Instruction::Mov(src, dst, move_type) => {
                let src = get_operand(src, move_type);
                let dst = get_operand(dst, move_type);
                let t = get_size_suffix(move_type);
                writeln!(file, "    mov{t} {src}, {dst}").unwrap();
            }
            assembly_gen::Instruction::MovSignExtend(src, dst) => {
                let src = get_operand(src, &AssemblyType::Longword);
                let dst = get_operand(dst, &AssemblyType::Quadword);
                writeln!(file, "    movslq {src}, {dst}").unwrap();
            }
            assembly_gen::Instruction::Ret => {
                writeln!(file, "    movq %rbp, %rsp").unwrap();
                writeln!(file, "    popq %rbp").unwrap();
                writeln!(file, "    ret").unwrap();
            }
            assembly_gen::Instruction::Unary(operator, operand, asm_type) => {
                let operator = get_unary_operator(operator);
                let operand = get_operand(operand, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(file, "    {operator}{t} {operand}").unwrap();
            }
            assembly_gen::Instruction::Cdq(asm_type) => match asm_type {
                AssemblyType::Longword => writeln!(file, "    cdq").unwrap(),
                AssemblyType::Quadword => writeln!(file, "    cqo").unwrap(),
                AssemblyType::Double => panic!("Cdq not supported for double!"),
            },
            assembly_gen::Instruction::IDiv(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(file, "    idiv{t} {operand}").unwrap();
            }
            assembly_gen::Instruction::Div(operand, asm_type) => {
                let t = get_size_suffix(asm_type);
                let operand = get_operand(operand, asm_type);
                writeln!(file, "    div{t} {operand}").unwrap();
            }
            assembly_gen::Instruction::Binary(operator, left, right, asm_type) => {
                let src = get_operand(left, asm_type);
                let dst = get_operand(right, asm_type);
                // Xor and Mult have special impls for double
                if *asm_type == AssemblyType::Double && *operator == BinaryOperator::BitXor {
                    writeln!(file, "    xorpd {src}, {dst}").unwrap();
                } else if *asm_type == AssemblyType::Double && *operator == BinaryOperator::Mult {
                    writeln!(file, "    mulsd {src}, {dst}").unwrap();
                } else {
                    let operator = get_binary_operator(operator);
                    let t = get_size_suffix(asm_type);
                    writeln!(file, "    {operator}{t} {src}, {dst}").unwrap();
                }
            }
            assembly_gen::Instruction::SetCond(cond, val) => {
                let code = get_cond_code(cond);
                let reg = get_short_reg(val);
                writeln!(file, "    set{} {}", code, reg).unwrap();
            }
            assembly_gen::Instruction::Compare(left, right, cmp_type) => {
                let left = get_operand(left, cmp_type);
                let right = get_operand(right, cmp_type);
                let t = get_size_suffix(cmp_type);
                if *cmp_type == AssemblyType::Double {
                    writeln!(file, "    comisd {left}, {right}").unwrap();
                } else {
                    writeln!(file, "    cmp{t} {left}, {right}").unwrap();
                }
            }
            assembly_gen::Instruction::JmpCond(cond, target) => {
                writeln!(file, "    j{} .L_{}", get_cond_code(cond), target).unwrap();
            }
            assembly_gen::Instruction::Jmp(target) => {
                writeln!(file, "    jmp .L_{}", target).unwrap();
            }
            assembly_gen::Instruction::Label(target) => {
                writeln!(file, ".L_{}:", target).unwrap();
            }
            assembly_gen::Instruction::Push(operand) => {
                writeln!(file, "    pushq {}", get_operand_quadword(operand)).unwrap();
            }
            assembly_gen::Instruction::Call(label) => {
                if let AsmSymbol::FunctionEntry(true) = &symbols[label] {
                    writeln!(file, "    call {}", label).unwrap();
                } else {
                    writeln!(file, "    call {}@PLT", label).unwrap();
                }
            }
            assembly_gen::Instruction::MovZeroExtend(_, _) => {
                panic!("Should be replaced with Mov instructions by rewriter!")
            }
            assembly_gen::Instruction::Cvttsd2si(src, dst, asm_type) => {
                let src = get_operand(src, asm_type);
                let dst = get_operand(dst, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(file, "    cvttsd2si{t} {src}, {dst}").unwrap();
            }
            assembly_gen::Instruction::Cvtsi2sd(src, dst, asm_type) => {
                let src = get_operand(src, asm_type);
                let dst = get_operand(dst, asm_type);
                let t = get_size_suffix(asm_type);
                writeln!(file, "    cvtsi2sd{t} {src}, {dst}").unwrap();
            }
            assembly_gen::Instruction::Lea(src, dst) => {
                let src = get_operand_quadword(src);
                let dst = get_operand_quadword(dst);
                writeln!(file, "    leaq {src}, {dst}").unwrap();
            }
            assembly_gen::Instruction::Nop => panic!("Noop should just be a placeholder"),
        }
    }
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
    }
}

fn get_operand(operand: &assembly_gen::Operand, asm_type: &AssemblyType) -> String {
    match asm_type {
        AssemblyType::Longword => get_operand_longword(operand),
        AssemblyType::Quadword | AssemblyType::Double => get_operand_quadword(operand),
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
    }
}
