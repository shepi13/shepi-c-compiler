use std::collections::HashMap;
use std::iter::zip;

use super::assembly_rewrite::check_overflow;
use crate::codegen::assembly_ast::{
    self, AsmSymbol, AssemblyType, BinaryOperator, Condition, FLOAT_REGISTERS, Function,
    INT_REGISTERS, Instruction, Operand, Program, Register, StackGen, StaticVar, TopLevelDecl,
    UnaryOperator,
};
use crate::parse::parse_tree;
use crate::tac_generation::generator::gen_label;
use crate::tac_generation::tac_ast::{self, Value};
use crate::validate::ctype::{CType, SymbolAttr, Symbols};

use assembly_ast::Operand::{Data, Imm, Memory, Register as Reg};
use assembly_ast::Register::*;

pub fn gen_assembly_tree(ast: tac_ast::Program, symbols: Symbols) -> Program {
    let mut program = Vec::new();
    let mut stack = StackGen::default();
    for decl in ast.into_iter() {
        match decl {
            tac_ast::TopLevelDecl::Function { name, params, instructions, global } => {
                if instructions.is_empty() {
                    continue;
                }
                // Setup initial state (including nop that will be replaced by stack allocation later)
                let mut new_instructions: Vec<Instruction> = Vec::new();
                new_instructions.push(Instruction::Nop);
                stack.reset_stack();
                // Pull arguments from registers/stack
                let mut param_setup = set_up_parameters(params, &mut stack, &symbols);
                new_instructions.append(&mut param_setup);
                // Generate instructions for function body
                let mut body = gen_instructions(instructions, &mut stack, &symbols);
                new_instructions.append(&mut body);
                // Replace Nop with stack allocation, and add instructions to program
                new_instructions[0] = Instruction::Binary(
                    BinaryOperator::Sub,
                    Imm((stack.stack_offset + 16 - stack.stack_offset % 16) as i128),
                    Reg(SP),
                    AssemblyType::Quadword,
                );
                program.push(TopLevelDecl::FunctionDecl(Function {
                    name,
                    global,
                    instructions: new_instructions,
                }));
            }
            tac_ast::TopLevelDecl::StaticDecl { identifier, global, ctype, initializer } => {
                // Calculate smaller valid alignment if possible TODO:
                let alignment = AssemblyType::from(ctype).get_alignment() as u64;
                program.push(TopLevelDecl::Var(StaticVar {
                    name: identifier,
                    global,
                    alignment,
                    init: initializer,
                }));
            }
        }
    }
    program.append(
        &mut stack.static_constants.into_iter().map(|var| TopLevelDecl::Constant(var.1)).collect(),
    );
    Program {
        program,
        backend_symbols: gen_backend_symbols(symbols),
    }
}

fn set_up_parameters(
    params: Vec<tac_ast::Value>,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    let param_groups = classify_parameters(params, stack, symbols);
    //Copy params from registers
    let mut copy_params = |param_group, registers: &[Register]| {
        for ((asm_type, param), reg) in zip(param_group, registers) {
            instructions.push(Instruction::Mov(Reg(*reg), param, asm_type));
        }
    };
    copy_params(param_groups.int_args, &INT_REGISTERS);
    copy_params(param_groups.float_args, &FLOAT_REGISTERS);
    // Copy remaining params from stack
    let mut offset = 16;
    for (asm_type, param) in param_groups.stack_args {
        instructions.push(Instruction::Mov(Operand::Memory(BP, offset), param, asm_type));
        offset += 8;
    }
    instructions
}

fn gen_backend_symbols(symbols: Symbols) -> HashMap<String, AsmSymbol> {
    let mut backend_symbols = HashMap::new();
    for (name, symbol) in symbols {
        match symbol.attrs {
            SymbolAttr::Function { defined, global: _ } => {
                backend_symbols.insert(name, AsmSymbol::FunctionEntry(defined));
            }
            _ => {
                let is_static = symbol.attrs != SymbolAttr::Local;
                backend_symbols
                    .insert(name, AsmSymbol::ObjectEntry(symbol.ctype.into(), is_static));
            }
        }
    }
    backend_symbols
}

fn gen_instructions(
    instructions: Vec<tac_ast::Instruction>,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> Vec<Instruction> {
    let mut asm_instructions: Vec<Instruction> = Vec::new();
    for instruction in instructions {
        use Instruction::*;
        match instruction {
            tac_ast::Instruction::Return(val) => {
                let val_type = AssemblyType::from(get_type(&val, symbols));
                let val = gen_operand(val, stack, symbols);
                let reg = if val_type == AssemblyType::Double { Reg(XMM0) } else { Reg(AX) };
                asm_instructions.push(Mov(val, reg, val_type));
                asm_instructions.push(Ret);
            }
            tac_ast::Instruction::UnaryOp { operator, src, dst } => {
                use BinaryOperator::BitXor;
                let src_type = AssemblyType::from(get_type(&src, symbols));
                let dst_type = AssemblyType::from(get_type(&dst, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                let operator = match operator {
                    tac_ast::UnaryOperator::Complement => UnaryOperator::Not,
                    tac_ast::UnaryOperator::Negate => {
                        if src_type == AssemblyType::Double {
                            // Doubles use xor with -0.0 for negation
                            let neg_zero = Operand::Data(stack.static_constant(-0.0, true));
                            asm_instructions.push(Mov(src, dst.clone(), AssemblyType::Double));
                            asm_instructions.push(Binary(BitXor, neg_zero, dst, src_type));
                            continue;
                        } else {
                            UnaryOperator::Neg
                        }
                    }
                    tac_ast::UnaryOperator::LogicalNot => {
                        // Logical not uses cmp instead of unary operators
                        if src_type == AssemblyType::Double {
                            let nan_label = gen_label("nan");
                            let end_label = gen_label("end");
                            asm_instructions.push(Binary(BitXor, Reg(XMM0), Reg(XMM0), src_type));
                            asm_instructions.push(Compare(src, Reg(XMM0), src_type));
                            asm_instructions.push(JmpCond(Condition::Parity, nan_label.clone()));
                            asm_instructions.push(Mov(Imm(0), dst.clone(), dst_type));
                            asm_instructions.push(SetCond(Condition::Equal, dst.clone()));
                            asm_instructions.push(Jmp(end_label.clone()));
                            asm_instructions.push(Label(nan_label));
                            asm_instructions.push(Mov(Imm(0), dst, dst_type));
                            asm_instructions.push(Label(end_label));
                        } else {
                            asm_instructions.push(Compare(Imm(0), src, src_type));
                            asm_instructions.push(Mov(Imm(0), dst.clone(), dst_type));
                            asm_instructions.push(SetCond(Condition::Equal, dst));
                        }
                        continue;
                    }
                };
                asm_instructions.push(Mov(src, dst.clone(), src_type));
                asm_instructions.push(Unary(operator, dst, src_type));
            }
            tac_ast::Instruction::BinaryOp { operator: _, src1: _, src2: _, dst: _ } => {
                gen_binary_op(&mut asm_instructions, instruction, stack, symbols);
            }
            tac_ast::Instruction::Jump(target) => {
                asm_instructions.push(Jmp(target));
            }
            tac_ast::Instruction::JumpCond { jump_type, condition, target } => {
                use BinaryOperator::BitXor;
                let condition_t = match jump_type {
                    tac_ast::JumpType::JumpIfZero => Condition::Equal,
                    tac_ast::JumpType::JumpIfNotZero => Condition::NotEqual,
                };
                let cmp_type = AssemblyType::from(get_type(&condition, symbols));
                let dst = gen_operand(condition, stack, symbols);
                if cmp_type == AssemblyType::Double {
                    let end_label = gen_label("jumpcond_end");
                    asm_instructions.push(Binary(BitXor, Reg(XMM0), Reg(XMM0), cmp_type));
                    asm_instructions.push(Compare(dst, Reg(XMM0), cmp_type));
                    match jump_type {
                        tac_ast::JumpType::JumpIfNotZero => {
                            asm_instructions.push(JmpCond(Condition::Parity, target.clone()))
                        }
                        tac_ast::JumpType::JumpIfZero => {
                            asm_instructions.push(JmpCond(Condition::Parity, end_label.clone()))
                        }
                    }
                    asm_instructions.push(JmpCond(condition_t, target));
                    asm_instructions.push(Label(end_label));
                } else {
                    asm_instructions.push(Compare(Imm(0), dst, cmp_type));
                    asm_instructions.push(JmpCond(condition_t, target));
                }
            }
            tac_ast::Instruction::Copy(src, dst) => {
                let src_type = AssemblyType::from(get_type(&src, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Mov(src, dst, src_type));
            }
            tac_ast::Instruction::Label(target) => {
                asm_instructions.push(Label(target));
            }
            tac_ast::Instruction::Function(name, args, dst) => {
                gen_func_call(&mut asm_instructions, stack, name, args, dst, symbols);
            }
            tac_ast::Instruction::SignExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(MovSignExtend(src, dst));
            }
            tac_ast::Instruction::Truncate(src, dst) => {
                let mut src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                if check_overflow(&src, i32::MAX as i128) {
                    let Imm(val) = src else { panic!("Overflow must be IMM") };
                    src = Imm(val & 0xFFFFFFFF);
                }
                asm_instructions.push(Mov(src, dst, AssemblyType::Longword));
            }
            tac_ast::Instruction::ZeroExtend(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(MovZeroExtend(src, dst));
            }
            tac_ast::Instruction::DoubleToInt(src, dst) => {
                let dst_type = AssemblyType::from(get_type(&dst, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Cvttsd2si(src, dst, dst_type));
            }
            tac_ast::Instruction::IntToDouble(src, dst) => {
                let src_type = AssemblyType::from(get_type(&src, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Cvtsi2sd(src, dst, src_type));
            }
            tac_ast::Instruction::DoubleToUInt(src, dst) => match get_type(&dst, symbols) {
                CType::UnsignedInt => {
                    let src = gen_operand(src, stack, symbols);
                    let dst = gen_operand(dst, stack, symbols);
                    asm_instructions.push(Cvttsd2si(src, Reg(AX), AssemblyType::Quadword));
                    asm_instructions.push(Mov(Reg(AX), dst, AssemblyType::Longword));
                }
                CType::UnsignedLong => {
                    gen_double2ull(&mut asm_instructions, src, dst, stack, symbols)
                }
                _ => panic!("Expected an unsigned dst"),
            },
            tac_ast::Instruction::UIntToDouble(src, dst) => match get_type(&src, symbols) {
                CType::UnsignedInt => {
                    let src = gen_operand(src, stack, symbols);
                    let dst = gen_operand(dst, stack, symbols);
                    asm_instructions.push(MovZeroExtend(src, Reg(AX)));
                    asm_instructions.push(Cvtsi2sd(Reg(AX), dst, AssemblyType::Quadword));
                }
                CType::UnsignedLong => {
                    gen_ull2double(&mut asm_instructions, src, dst, stack, symbols)
                }
                _ => panic!("Expected an unsigned src!"),
            },
            tac_ast::Instruction::GetAddress(src, dst) => {
                let src = gen_operand(src, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Lea(src, dst));
            }
            tac_ast::Instruction::Load(ptr, dst) => {
                let dst_type = AssemblyType::from(get_type(&dst, symbols));
                let ptr = gen_operand(ptr, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Mov(ptr, Reg(AX), AssemblyType::Quadword));
                asm_instructions.push(Mov(Memory(AX, 0), dst, dst_type));
            }
            tac_ast::Instruction::Store(src, ptr) => {
                let src_type = AssemblyType::from(get_type(&src, symbols));
                let src = gen_operand(src, stack, symbols);
                let ptr = gen_operand(ptr, stack, symbols);
                asm_instructions.push(Mov(ptr, Reg(AX), AssemblyType::Quadword));
                asm_instructions.push(Mov(src, Memory(AX, 0), src_type));
            }
            tac_ast::Instruction::AddPtr(ptr, index, scale, dst) => {
                let ptr = gen_operand(ptr, stack, symbols);
                let index = gen_operand(index, stack, symbols);
                let dst = gen_operand(dst, stack, symbols);
                asm_instructions.push(Mov(ptr, Reg(AX), AssemblyType::Quadword));
                if let Imm(index_val) = index {
                    let offset = index_val as isize * scale as isize;
                    asm_instructions.push(Lea(Memory(AX, offset), dst));
                } else if [1, 2, 4, 8].contains(&scale) {
                    asm_instructions.push(Mov(index, Reg(DX), AssemblyType::Quadword));
                    asm_instructions.push(Lea(Operand::Indexed(AX, DX, scale), dst));
                } else {
                    asm_instructions.push(Mov(index, Reg(DX), AssemblyType::Quadword));
                    asm_instructions.push(Binary(
                        BinaryOperator::Mult,
                        Imm(scale.into()),
                        Reg(DX),
                        AssemblyType::Quadword,
                    ));
                    asm_instructions.push(Lea(Operand::Indexed(AX, DX, 1), dst));
                }
            }
            tac_ast::Instruction::CopyToOffset(src, dst, offset) => {
                let src_type = AssemblyType::from(get_type(&src, symbols));
                let src = gen_operand(src, stack, symbols);
                let dst = gen_mem_offset(dst, offset as isize, stack, symbols);
                asm_instructions.push(Mov(src, dst, src_type));
            }
        }
    }
    asm_instructions
}

fn gen_double2ull(
    instructions: &mut Vec<Instruction>,
    src: Value,
    dst: Value,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    use BinaryOperator::{Add, Sub};
    use Instruction::*;
    let src = gen_operand(src, stack, symbols);
    let dst = gen_operand(dst, stack, symbols);
    // Long max + 1
    let upper_bound = stack.static_constant(9223372036854775808.0, false);

    let out_of_range_lbl = gen_label("out_of_range");
    let end_lbl = gen_label("end");
    // If it fits in signed long, conversion is trivial
    instructions.push(Compare(Data(upper_bound.clone()), src.clone(), AssemblyType::Double));
    instructions.push(JmpCond(Condition::UnsignedGreaterEqual, out_of_range_lbl.clone()));
    instructions.push(Cvttsd2si(src.clone(), Reg(AX), AssemblyType::Quadword));
    instructions.push(Jmp(end_lbl.clone()));

    // Otherwise subtract long_max+1, convert, and add it again
    instructions.push(Label(out_of_range_lbl));
    instructions.push(Mov(src, Reg(XMM1), AssemblyType::Double));
    instructions.push(Binary(Sub, Data(upper_bound), Reg(XMM1), AssemblyType::Double));
    instructions.push(Cvttsd2si(Reg(XMM1), Reg(AX), AssemblyType::Quadword));
    instructions.push(Mov(Imm(9223372036854775808), Reg(DX), AssemblyType::Quadword));
    instructions.push(Binary(Add, Reg(DX), Reg(AX), AssemblyType::Quadword));

    // Result is in AX
    instructions.push(Label(end_lbl));
    instructions.push(Mov(Reg(AX), dst, AssemblyType::Quadword));
}

fn gen_ull2double(
    instructions: &mut Vec<Instruction>,
    src: Value,
    dst: Value,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    use Instruction::*;
    let src = gen_operand(src, stack, symbols);
    let dst = gen_operand(dst, stack, symbols);

    let out_of_range_lbl = gen_label("out_of_range");
    let end_lbl = gen_label("end");
    // Check if it fits in a signed long, if so conversion is trivial with Cvtsi2sd
    instructions.push(Compare(Imm(0), src.clone(), AssemblyType::Quadword));
    instructions.push(JmpCond(Condition::LessThan, out_of_range_lbl.clone()));
    instructions.push(Cvtsi2sd(src.clone(), Reg(XMM0), AssemblyType::Quadword));
    instructions.push(Jmp(end_lbl.clone()));

    // Otherwise Divide by 2 and round to odd by using bitwise and/or
    instructions.push(Label(out_of_range_lbl));
    instructions.push(Mov(src, Reg(AX), AssemblyType::Quadword));
    instructions.push(Mov(Reg(AX), Reg(DX), AssemblyType::Quadword));
    instructions.push(Unary(UnaryOperator::Shr, Reg(DX), AssemblyType::Quadword));
    instructions.push(Binary(BinaryOperator::BitAnd, Imm(1), Reg(AX), AssemblyType::Quadword));
    instructions.push(Binary(BinaryOperator::BitOr, Reg(AX), Reg(DX), AssemblyType::Quadword));
    // Convert divided value and multiply by 2
    instructions.push(Cvtsi2sd(Reg(DX), Reg(XMM0), AssemblyType::Quadword));
    instructions.push(Binary(BinaryOperator::Add, Reg(XMM0), Reg(XMM0), AssemblyType::Double));

    // Result is in XMM0
    instructions.push(Label(end_lbl));
    instructions.push(Mov(Reg(XMM0), dst, AssemblyType::Double));
}

struct ParamGroups {
    int_args: Vec<(AssemblyType, Operand)>,
    float_args: Vec<(AssemblyType, Operand)>,
    stack_args: Vec<(AssemblyType, Operand)>,
}
fn classify_parameters(
    parameters: Vec<Value>,
    stack: &mut StackGen,
    symbols: &Symbols,
) -> ParamGroups {
    let mut int_args = Vec::new();
    let mut float_args = Vec::new();
    let mut stack_args = Vec::new();
    for param in parameters {
        let param_type = AssemblyType::from(get_type(&param, symbols));
        let operand = gen_operand(param, stack, symbols);
        if param_type == AssemblyType::Double && float_args.len() < 8 {
            float_args.push((param_type, operand));
        } else if param_type != AssemblyType::Double && int_args.len() < 6 {
            int_args.push((param_type, operand));
        } else {
            stack_args.push((param_type, operand))
        }
    }
    ParamGroups { int_args, float_args, stack_args }
}

fn gen_func_call(
    instructions: &mut Vec<Instruction>,
    stack: &mut StackGen,
    name: String,
    args: Vec<tac_ast::Value>,
    dst: tac_ast::Value,
    symbols: &Symbols,
) {
    use AssemblyType::*;
    use Instruction::*;
    let param_groups = classify_parameters(args, stack, symbols);
    // Calculate and adjust stack alignment
    let padding = if param_groups.stack_args.len() % 2 == 1 { 8 } else { 0 };
    let bytes_to_remove = 8 * param_groups.stack_args.len() as i128 + padding;
    if padding != 0 {
        instructions.push(Binary(BinaryOperator::Sub, Imm(padding), Reg(SP), Quadword));
    }
    // Pass arguments in registers
    let mut pass_args = |param_group, registers: &[Register]| {
        for ((arg_type, arg), reg) in zip(param_group, registers) {
            instructions.push(Mov(arg, Reg(*reg), arg_type));
        }
    };
    pass_args(param_groups.int_args, &INT_REGISTERS);
    pass_args(param_groups.float_args, &FLOAT_REGISTERS);
    // Pass arguments on stack
    for (arg_type, arg) in param_groups.stack_args.into_iter().rev() {
        if matches!(arg, Reg(_) | Imm(_)) || matches!(arg_type, Double | Quadword) {
            instructions.push(Push(arg));
        } else {
            instructions.push(Mov(arg, Reg(AX), arg_type));
            instructions.push(Push(Reg(AX)));
        }
    }
    // Call function
    instructions.push(Call(name));
    // Clean up stack
    if bytes_to_remove != 0 {
        instructions.push(Binary(BinaryOperator::Add, Imm(bytes_to_remove), Reg(SP), Quadword));
    }
    // Get return value from eax or xmm0
    let dst_type = AssemblyType::from(get_type(&dst, symbols));
    let dst = gen_operand(dst, stack, symbols);
    let result_reg = if dst_type == AssemblyType::Double { Reg(XMM0) } else { Reg(AX) };
    instructions.push(Instruction::Mov(result_reg, dst, dst_type));
}

fn gen_binary_op(
    instructions: &mut Vec<Instruction>,
    binary_instruction: tac_ast::Instruction,
    stack: &mut StackGen,
    symbols: &Symbols,
) {
    let tac_ast::Instruction::BinaryOp { operator, src1, src2, dst } = binary_instruction else {
        panic!("Expected binary instruction!")
    };
    use parse_tree::BinaryOperator::*;
    let src_ctype = get_type(&src1, symbols);
    let asm_type = AssemblyType::from(src_ctype.clone());
    let src1 = gen_operand(src1, stack, symbols);
    let src2 = gen_operand(src2, stack, symbols);
    let dst = gen_operand(dst, stack, symbols);
    // Instructions for generic binary ops (addition, subtraction, mult, bitwise)
    let mut gen_arithmetic = |operator| {
        instructions.push(Instruction::Mov(src1.clone(), dst.clone(), asm_type));
        instructions.push(Instruction::Binary(operator, src2.clone(), dst.clone(), asm_type));
    };
    match &operator {
        // Handle arithmetic binary operators
        Add => gen_arithmetic(BinaryOperator::Add),
        Multiply => gen_arithmetic(BinaryOperator::Mult),
        Subtract => gen_arithmetic(BinaryOperator::Sub),
        BitAnd => gen_arithmetic(BinaryOperator::BitAnd),
        BitOr => gen_arithmetic(BinaryOperator::BitOr),
        BitXor => gen_arithmetic(BinaryOperator::BitXor),
        LeftShift => {
            let operator = match src_ctype.is_signed() {
                true => BinaryOperator::LeftShift,
                false => BinaryOperator::LeftShiftUnsigned,
            };
            gen_shift(instructions, operator, src1, src2, dst, asm_type);
        }
        RightShift => {
            let operator = match src_ctype.is_signed() {
                true => BinaryOperator::RightShift,
                false => BinaryOperator::RightShiftUnsigned,
            };
            gen_shift(instructions, operator, src1, src2, dst, asm_type);
        }
        // Division is handled separately
        Divide => {
            if asm_type == AssemblyType::Double {
                gen_arithmetic(BinaryOperator::DoubleDiv);
            } else {
                let sign = src_ctype.is_signed();
                gen_division(instructions, src1, src2, dst, Register::AX, asm_type, sign);
            }
        }
        Remainder => {
            let sign = src_ctype.is_signed();
            gen_division(instructions, src1, src2, dst, Register::DX, asm_type, sign);
        }
        GreaterThan | GreaterThanEqual | IsEqual | LessThan | LessThanEqual | LogicalAnd
        | LogicalOr | NotEqual => {
            use AssemblyType::Longword;
            let signed = if asm_type == AssemblyType::Double {
                false
            } else {
                src_ctype.is_arithmetic() && src_ctype.is_signed()
            };
            // NAN returns true for !=, false otherwise
            let nan_result = (operator == NotEqual) as i128;
            let condition = get_condition(operator, signed);
            let nan_label = gen_label("is_nan");
            let end_label = gen_label("end");
            instructions.push(Instruction::Compare(src2, src1, asm_type));
            // Double skips the set and defaults to false if either value is NaN
            if asm_type == AssemblyType::Double {
                instructions.push(Instruction::JmpCond(Condition::Parity, nan_label.clone()));
            }
            instructions.push(Instruction::Mov(Imm(0), dst.clone(), Longword));
            instructions.push(Instruction::SetCond(condition, dst.clone()));
            if asm_type == AssemblyType::Double {
                instructions.push(Instruction::Jmp(end_label.clone()));
                instructions.push(Instruction::Label(nan_label));
                instructions.push(Instruction::Mov(Imm(nan_result), dst, Longword));
                instructions.push(Instruction::Label(end_label));
            }
        }
    };
}

fn get_condition(op: parse_tree::BinaryOperator, signed: bool) -> Condition {
    match signed {
        true => match op {
            parse_tree::BinaryOperator::GreaterThan => Condition::GreaterThan,
            parse_tree::BinaryOperator::GreaterThanEqual => Condition::GreaterThanEqual,
            parse_tree::BinaryOperator::LessThan => Condition::LessThan,
            parse_tree::BinaryOperator::LessThanEqual => Condition::LessThanEqual,
            parse_tree::BinaryOperator::NotEqual => Condition::NotEqual,
            parse_tree::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected relational operator!"),
        },
        false => match op {
            parse_tree::BinaryOperator::GreaterThan => Condition::UnsignedGreaterThan,
            parse_tree::BinaryOperator::GreaterThanEqual => Condition::UnsignedGreaterEqual,
            parse_tree::BinaryOperator::LessThan => Condition::UnsignedLessThan,
            parse_tree::BinaryOperator::LessThanEqual => Condition::UnsignedLessEqual,
            parse_tree::BinaryOperator::NotEqual => Condition::NotEqual,
            parse_tree::BinaryOperator::IsEqual => Condition::Equal,
            _ => panic!("Expected releational operator"),
        },
    }
}

fn gen_shift(
    instructions: &mut Vec<Instruction>,
    operator: BinaryOperator,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    shift_type: AssemblyType,
) {
    instructions.push(Instruction::Mov(src1, Reg(AX), shift_type));
    instructions.push(Instruction::Mov(src2, Reg(CX), shift_type));
    instructions.push(Instruction::Binary(operator, Reg(CL), Reg(AX), shift_type));
    instructions.push(Instruction::Mov(Reg(AX), dst, shift_type));
}

fn gen_division(
    instructions: &mut Vec<Instruction>,
    src1: Operand,
    src2: Operand,
    dst: Operand,
    result_reg: Register,
    div_type: AssemblyType,
    is_signed: bool,
) {
    // Move dividend into AX
    instructions.push(Instruction::Mov(src1, Reg(AX), div_type));

    // USE CDQ or 0 extend to setup registers for division
    // Use IDiv for signed and Div for unsigned
    if is_signed {
        instructions.push(Instruction::Cdq(div_type));
        instructions.push(Instruction::IDiv(src2, div_type));
    } else {
        instructions.push(Instruction::Mov(Imm(0), Reg(DX), div_type));
        instructions.push(Instruction::Div(src2, div_type));
    }
    // Division result is in AX, Remainder in DX
    instructions.push(Instruction::Mov(Reg(result_reg), dst, div_type));
}

fn get_type(value: &tac_ast::Value, symbols: &Symbols) -> CType {
    match &value {
        tac_ast::Value::Variable(name) => symbols[name].ctype.clone(),
        tac_ast::Value::ConstValue(constexpr) => match constexpr {
            parse_tree::Constant::Int(_) => CType::Int,
            parse_tree::Constant::UInt(_) => CType::UnsignedInt,
            parse_tree::Constant::Long(_) => CType::Long,
            parse_tree::Constant::ULong(_) => CType::UnsignedLong,
            parse_tree::Constant::Double(_) => CType::Double,
            parse_tree::Constant::Char(_) | parse_tree::Constant::UChar(_) => {
                todo!("Implement char types!")
            }
        },
    }
}

fn gen_mem_offset(name: String, offset: isize, stack: &mut StackGen, symbols: &Symbols) -> Operand {
    let operand = gen_operand(Value::Variable(name), stack, symbols);
    match operand {
        Operand::Data(_) => operand,
        Operand::Memory(BP, location) => Operand::Memory(BP, location + offset),
        _ => panic!("Expected operand in memory"),
    }
}

fn gen_operand(value: tac_ast::Value, stack: &mut StackGen, symbols: &Symbols) -> Operand {
    match value {
        tac_ast::Value::ConstValue(constexpr) => match constexpr {
            parse_tree::Constant::Int(val) | parse_tree::Constant::Long(val) => {
                Operand::Imm(val.into())
            }
            parse_tree::Constant::UInt(val) | parse_tree::Constant::ULong(val) => {
                Operand::Imm(val.into())
            }
            parse_tree::Constant::Double(val) => {
                let name = stack.static_constant(val, false);
                Operand::Data(name)
            }
            parse_tree::Constant::Char(_) | parse_tree::Constant::UChar(_) => {
                todo!("Implement char types!")
            }
        },
        tac_ast::Value::Variable(name) => {
            let symbol = &symbols[&name];
            let var_type = AssemblyType::from(symbol.ctype.clone());
            if matches!(symbol.attrs, SymbolAttr::Static { init: _, global: _ }) {
                Operand::Data(name)
            } else if let Some(location) = stack.stack_variables.get(&name) {
                Operand::Memory(BP, -(*location as isize))
            } else {
                match var_type {
                    AssemblyType::Longword => stack.stack_offset += 4,
                    AssemblyType::Quadword | AssemblyType::Double => {
                        stack.stack_offset += 8 + (8 - stack.stack_offset % 8);
                    }
                    AssemblyType::ByteArray(size, alignment) => {
                        stack.stack_offset += size as usize;
                        stack.stack_offset += alignment - stack.stack_offset % alignment;
                    }
                }
                stack.stack_variables.insert(name.to_string(), stack.stack_offset);
                Operand::Memory(BP, -(stack.stack_offset as isize))
            }
        }
    }
}
