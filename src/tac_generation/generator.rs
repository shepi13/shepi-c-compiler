// Generates Three Adress Code Intermediate representation (Step between Parser AST and Assembly AST)

use std::{
    cmp::min,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    parse::parse_tree::{
        self, BinaryExpression, BinaryOperator, Constant, Expression, ForInit, IncrementType,
        Statement, TypedExpression, VariableInitializer,
    },
    tac_generation::tac_ast::{Instruction, JumpType, Program, TopLevelDecl, Value},
    validate::ctype::{
        CType, Initializer, StaticInitializer, Symbol, SymbolAttr, Symbols, get_common_type,
        string_name,
    },
};

pub fn gen_tac_ast(parser_ast: parse_tree::Program, symbols: &mut Symbols) -> Program {
    let mut program: Program = Vec::new();

    for decl in parser_ast {
        if let parse_tree::Declaration::Function(function) = decl {
            program.push(gen_function(function, symbols));
        }
    }
    for (name, entry) in symbols {
        if let SymbolAttr::Static { init, global } = &entry.attrs {
            match &init {
                StaticInitializer::Initialized(initializer) => {
                    program.push(TopLevelDecl::StaticDecl {
                        identifier: name.clone(),
                        global: *global,
                        ctype: entry.ctype.clone(),
                        initializer: initializer.clone(),
                    });
                }
                StaticInitializer::Tentative => {
                    let initializer = match &entry.ctype {
                        CType::Double => Initializer::Double(0.0),
                        CType::Int
                        | CType::Long
                        | CType::UnsignedInt
                        | CType::UnsignedLong
                        | CType::Pointer(_)
                        | CType::Array(_, _) => Initializer::ZeroInit(entry.ctype.size()),
                        _ => panic!("Not a variable"),
                    };
                    program.push(TopLevelDecl::StaticDecl {
                        identifier: name.clone(),
                        global: *global,
                        ctype: entry.ctype.clone(),
                        initializer: vec![initializer],
                    });
                }
                StaticInitializer::None => (),
            }
        } else if let SymbolAttr::Constant(init) = &entry.attrs {
            program.push(TopLevelDecl::StaticConstant {
                identifier: name.clone(),
                ctype: entry.ctype.clone(),
                initializer: init.clone(),
            })
        }
    }
    program
}
fn gen_function(function: parse_tree::FunctionDeclaration, symbols: &mut Symbols) -> TopLevelDecl {
    let mut instructions: Vec<Instruction> = Vec::new();
    if let Some(body) = function.body {
        gen_block(body, &mut instructions, symbols);
        let ret_val = match function.ctype {
            CType::Function(_, ret_t) if *ret_t == CType::Void => None,
            _ => Some(Value::ConstValue(parse_tree::Constant::Int(0))),
        };
        instructions.push(Instruction::Return(ret_val));
    }
    let SymbolAttr::Function { defined: _, global } = symbols[&function.name].attrs else {
        panic!("Expected function!")
    };
    let params = function.params.into_iter().map(Value::Variable).collect();
    TopLevelDecl::Function {
        name: function.name,
        params,
        global,
        instructions,
    }
}
fn gen_block(block: parse_tree::Block, instructions: &mut Vec<Instruction>, symbols: &mut Symbols) {
    for block_item in block {
        match block_item {
            parse_tree::BlockItem::StatementItem(statement, _) => {
                gen_instructions(statement, instructions, symbols)
            }
            parse_tree::BlockItem::DeclareItem(parse_tree::Declaration::Variable(decl)) => {
                gen_declaration(decl, instructions, symbols);
            }
            parse_tree::BlockItem::DeclareItem(parse_tree::Declaration::Function(_)) => (),
            parse_tree::BlockItem::DeclareItem(parse_tree::Declaration::Struct {
                tag: _,
                members: _,
            }) => {
                todo!("Structure declaration tac!")
            }
            parse_tree::BlockItem::DeclareItem(parse_tree::Declaration::Union {
                tag: _,
                members: _,
            }) => {
                todo!("Union declaration tac!")
            }
        }
    }
}
fn gen_declaration(
    declaration: parse_tree::VariableDeclaration,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) {
    if declaration.storage.is_some() {
        return;
    }
    match declaration.init {
        Some(VariableInitializer::SingleElem(TypedExpression {
            ctype: _,
            expr: Expression::StringLiteral(_),
        })) => {
            let v_init = vec![declaration.init.expect("Is some")];
            gen_init_list(instructions, v_init, declaration.name, &declaration.ctype, 0, symbols);
        }
        Some(VariableInitializer::SingleElem(value)) => {
            let result = gen_expression_and_convert(value, instructions, symbols);
            instructions.push(Instruction::Copy(result, Value::Variable(declaration.name)));
        }
        Some(VariableInitializer::CompoundInit(v_init)) => {
            gen_init_list(instructions, v_init, declaration.name, &declaration.ctype, 0, symbols);
        }
        None => (),
    }
}

fn gen_init_list(
    instructions: &mut Vec<Instruction>,
    initializers: Vec<VariableInitializer>,
    name: String,
    ctype: &CType,
    initial_offset: u64,
    symbols: &mut Symbols,
) {
    let CType::Array(elem_t, _) = ctype else { panic!("Expected array") };
    for (i, initializer) in initializers.into_iter().enumerate() {
        let offset = initial_offset + (i as u64) * elem_t.size();
        match initializer {
            VariableInitializer::SingleElem(expr) => {
                if let Expression::StringLiteral(s_data) = &expr.expr {
                    const CHUNK_SIZE: usize = 8;
                    // Adjust string data to size of array by truncating or adding 0-byte padding
                    let buffer_len = min(s_data.len(), ctype.size() as usize);
                    let mut data: Vec<u8> = s_data.as_bytes()[..buffer_len].to_vec();
                    data.resize(expr.get_type().size() as usize, 0);
                    // Loop over padded string and generate copy to offset instructions
                    for (i, chunk) in data.chunks(CHUNK_SIZE).enumerate() {
                        let offset = offset + i as u64 * 8;
                        let (chunk, remainder) = chunk.split_at(chunk.len() & 12);
                        // Copy chunk
                        if chunk.len() == 8 {
                            let val = Constant::Long(i64::from_le_bytes(chunk.try_into().unwrap()));
                            let val = Value::ConstValue(val);
                            instructions.push(Instruction::CopyToOffset(val, name.clone(), offset));
                        } else if chunk.len() == 4 {
                            let val =
                                Constant::Int(i32::from_le_bytes(chunk.try_into().unwrap()) as i64);
                            let val = Value::ConstValue(val);
                            instructions.push(Instruction::CopyToOffset(val, name.clone(), offset));
                        }
                        // Copy remainder
                        for (byte_offset, byte) in remainder.iter().enumerate() {
                            let offset = offset + (byte_offset + chunk.len()) as u64;
                            let val = Value::ConstValue(Constant::Char(*byte as i64));
                            instructions.push(Instruction::CopyToOffset(val, name.clone(), offset))
                        }
                    }
                } else {
                    let result = gen_expression_and_convert(expr, instructions, symbols);
                    instructions.push(Instruction::CopyToOffset(result, name.clone(), offset));
                }
            }
            VariableInitializer::CompoundInit(init_list) => {
                gen_init_list(instructions, init_list, name.clone(), elem_t, offset, symbols);
            }
        }
    }
}

fn gen_instructions(
    statement: Statement,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) {
    match statement {
        Statement::Return(Some(value)) => {
            let dst = gen_expression_and_convert(value, instructions, symbols);
            instructions.push(Instruction::Return(Some(dst)));
        }
        Statement::Return(None) => instructions.push(Instruction::Return(None)),
        Statement::Null => (),
        Statement::ExprStmt(value) => {
            gen_expression_and_convert(value, instructions, symbols);
        }
        Statement::If(condition, if_true, if_false) => {
            let end_label = gen_label("end");
            let else_label = gen_label("else");
            let condition = gen_expression_and_convert(condition, instructions, symbols);
            instructions.push(Instruction::JumpCond {
                jump_type: JumpType::JumpIfZero,
                condition,
                target: else_label.clone(),
            });
            gen_instructions(*if_true, instructions, symbols);
            instructions.push(Instruction::Jump(end_label.clone()));
            instructions.push(Instruction::Label(else_label));
            if let Some(false_statement) = *if_false {
                gen_instructions(false_statement, instructions, symbols);
            }
            instructions.push(Instruction::Label(end_label));
        }
        Statement::Goto(target) => {
            instructions.push(Instruction::Jump(target.to_string()));
        }
        Statement::Label(name, statement) => {
            instructions.push(Instruction::Label(name.to_string()));
            gen_instructions(*statement, instructions, symbols);
        }
        Statement::Compound(block) => gen_block(block, instructions, symbols),
        Statement::Break(name) => {
            let target = format!("break_{}", name);
            instructions.push(Instruction::Jump(target));
        }
        Statement::Continue(name) => {
            let target = format!("continue_{}", name);
            instructions.push(Instruction::Jump(target));
        }
        Statement::DoWhile(loop_data) => {
            let start = format!("start_{}", loop_data.label);
            instructions.push(Instruction::Label(start.clone()));
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Label(format!("continue_{}", loop_data.label)));
            let result = gen_expression_and_convert(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond {
                jump_type: JumpType::JumpIfNotZero,
                condition: result,
                target: start,
            });
            instructions.push(Instruction::Label(format!("break_{}", loop_data.label)));
        }
        Statement::While(loop_data) => {
            let break_label = format!("break_{}", loop_data.label);
            let continue_label = format!("continue_{}", loop_data.label);
            instructions.push(Instruction::Label(continue_label.clone()));
            let result = gen_expression_and_convert(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond {
                jump_type: JumpType::JumpIfZero,
                condition: result,
                target: break_label.clone(),
            });
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Jump(continue_label));
            instructions.push(Instruction::Label(break_label));
        }
        Statement::For(init, loop_data, post_loop) => {
            match *init {
                ForInit::Decl(decl) => {
                    gen_declaration(decl, instructions, symbols);
                }
                ForInit::Expr(Some(expr)) => {
                    gen_expression_and_convert(expr, instructions, symbols);
                }
                _ => (),
            };
            let break_label = format!("break_{}", loop_data.label);
            let start_label = format!("start_{}", loop_data.label);
            instructions.push(Instruction::Label(start_label.clone()));
            let condition = gen_expression_and_convert(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond {
                jump_type: JumpType::JumpIfZero,
                condition,
                target: break_label.clone(),
            });
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Label(format!("continue_{}", loop_data.label)));
            if let Some(post) = post_loop {
                gen_expression_and_convert(post, instructions, symbols);
            }
            instructions.push(Instruction::Jump(start_label));
            instructions.push(Instruction::Label(break_label));
        }
        Statement::Switch {
            label,
            condition,
            cases,
            statement,
            default,
        } => {
            let src1 = gen_expression_and_convert(condition, instructions, symbols);
            let dst = gen_temp_var(CType::Int, symbols);
            for case in cases {
                let src2 = gen_expression_and_convert(case.1, instructions, symbols);
                instructions.push(Instruction::BinaryOp {
                    operator: BinaryOperator::IsEqual,
                    src1: src1.clone(),
                    src2: src2.clone(),
                    dst: dst.clone(),
                });
                instructions.push(Instruction::JumpCond {
                    jump_type: JumpType::JumpIfNotZero,
                    condition: dst.clone(),
                    target: case.0,
                });
            }
            if let Some(target) = default {
                instructions.push(Instruction::Jump(target));
            } else {
                instructions.push(Instruction::Jump(format!("break_{}", label)));
            }
            gen_instructions(*statement, instructions, symbols);
            instructions.push(Instruction::Label(format!("break_{}", label)));
        }
        Statement::Case(_, _) | Statement::Default(_) => {
            panic!("Compiler error: case/default should be replaced in typecheck pass")
        }
    }
}

fn lvalue_convert(
    instructions: &mut Vec<Instruction>,
    result: ExpResult,
    ctype: Option<CType>,
    symbols: &mut Symbols,
) -> Value {
    match result {
        ExpResult::Operand(val) => val,
        ExpResult::DereferencedPointer(ptr) => {
            let dst = gen_temp_var(ctype.expect("Undefined type!"), symbols);
            instructions.push(Instruction::Load(ptr, dst.clone()));
            dst
        }
    }
}

fn gen_expression_and_convert(
    expression: TypedExpression,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) -> Value {
    let expr_type = expression.ctype.clone();
    let result = gen_expression(expression, instructions, symbols);
    lvalue_convert(instructions, result, expr_type, symbols)
}
#[derive(Debug, Clone)]
enum ExpResult {
    Operand(Value),
    DereferencedPointer(Value),
}
fn gen_expression(
    expression: TypedExpression,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) -> ExpResult {
    match expression.expr {
        Expression::Constant(constexpr) => ExpResult::Operand(Value::ConstValue(constexpr)),
        Expression::Unary(parse_tree::UnaryOperator::Increment(increment_type), expr) => {
            gen_increment(instructions, *expr, increment_type, symbols)
        }
        Expression::Unary(operator, expr) => {
            let expr_type = expression.ctype.expect("Undefined type!");
            let src = gen_expression_and_convert(*expr, instructions, symbols);
            let dst = gen_temp_var(expr_type, symbols);
            instructions.push(Instruction::UnaryOp {
                operator: operator.into(),
                src,
                dst: dst.clone(),
            });
            ExpResult::Operand(dst)
        }
        Expression::Binary(binary) => {
            use BinaryOperator::{Add, LogicalAnd, LogicalOr, Subtract};
            let expr_t = expression.ctype.expect("Undefined type!");
            // Short circuiting needs special handling
            if let LogicalAnd | LogicalOr = binary.operator {
                gen_short_circuit(instructions, *binary, symbols)
            } else if binary.operator == Add && expr_t.is_pointer() {
                gen_pointer_addition(instructions, *binary, symbols)
            } else if binary.operator == Subtract && binary.left.get_type().is_pointer() {
                gen_pointer_subtraction(instructions, *binary, symbols)
            } else if binary.is_assignment {
                gen_compound_assignment(instructions, *binary, symbols)
            } else {
                let src1 = gen_expression_and_convert(binary.left, instructions, symbols);
                let src2 = gen_expression_and_convert(binary.right, instructions, symbols);
                let dst = gen_temp_var(expr_t, symbols);
                instructions.push(Instruction::BinaryOp {
                    operator: binary.operator,
                    src1,
                    src2,
                    dst: dst.clone(),
                });
                ExpResult::Operand(dst)
            }
        }
        Expression::Variable(name) => ExpResult::Operand(Value::Variable(name.to_string())),
        Expression::Assignment(assignment) => {
            let lval = gen_expression(assignment.left, instructions, symbols);
            let rval = gen_expression_and_convert(assignment.right, instructions, symbols);
            match &lval {
                ExpResult::Operand(val) => {
                    instructions.push(Instruction::Copy(rval, val.clone()));
                    lval
                }
                ExpResult::DereferencedPointer(ptr) => {
                    instructions.push(Instruction::Store(rval.clone(), ptr.clone()));
                    ExpResult::Operand(rval)
                }
            }
        }
        Expression::Condition(condition) => {
            let expr_type = expression.ctype.expect("Undefined type!");
            let end_label = gen_label("cond_end");
            let e2_label = gen_label("cond_e2");
            let cond = gen_expression_and_convert(condition.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond {
                jump_type: JumpType::JumpIfZero,
                condition: cond,
                target: e2_label.clone(),
            });
            if expr_type == CType::Void {
                gen_expression_and_convert(condition.if_true, instructions, symbols);
                instructions.push(Instruction::Jump(end_label.clone()));
                instructions.push(Instruction::Label(e2_label));
                gen_expression_and_convert(condition.if_false, instructions, symbols);
                instructions.push(Instruction::Label(end_label));
                ExpResult::Operand(Value::Variable(String::from("DummyVar")))
            } else {
                let dst = gen_temp_var(expr_type, symbols);
                let e1 = gen_expression_and_convert(condition.if_true, instructions, symbols);
                instructions.push(Instruction::Copy(e1, dst.clone()));
                instructions.push(Instruction::Jump(end_label.clone()));
                instructions.push(Instruction::Label(e2_label));
                let e2 = gen_expression_and_convert(condition.if_false, instructions, symbols);
                instructions.push(Instruction::Copy(e2, dst.clone()));
                instructions.push(Instruction::Label(end_label));
                ExpResult::Operand(dst)
            }
        }
        Expression::FunctionCall(name, args) => {
            let expr_type = expression.ctype.expect("Undefined type!");
            let results = args
                .into_iter()
                .map(|arg| gen_expression_and_convert(arg, instructions, symbols))
                .collect();
            match &symbols[&name].ctype {
                CType::Function(_, ret_t) if **ret_t == CType::Void => {
                    instructions.push(Instruction::Function(name.to_string(), results, None));
                    ExpResult::Operand(Value::Variable(String::from("DummyVar")))
                }
                _ => {
                    let dst = gen_temp_var(expr_type, symbols);
                    instructions.push(Instruction::Function(
                        name.to_string(),
                        results,
                        Some(dst.clone()),
                    ));
                    ExpResult::Operand(dst)
                }
            }
        }
        Expression::Cast(new_type, castexpr) => {
            let old_type = castexpr.get_type();
            let result = gen_expression_and_convert(*castexpr, instructions, symbols);
            let dst = gen_cast(instructions, new_type, old_type, result, symbols);
            ExpResult::Operand(dst)
        }
        Expression::Dereference(inner) => {
            let result = gen_expression_and_convert(*inner, instructions, symbols);
            ExpResult::DereferencedPointer(result)
        }
        Expression::AddrOf(inner) => {
            let expr_type = expression.ctype.expect("Undefined type!");
            let result = gen_expression(*inner, instructions, symbols);
            match result {
                ExpResult::Operand(val) => {
                    let dst = gen_temp_var(expr_type, symbols);
                    instructions.push(Instruction::GetAddress(val, dst.clone()));
                    ExpResult::Operand(dst)
                }
                ExpResult::DereferencedPointer(ptr) => ExpResult::Operand(ptr),
            }
        }
        Expression::Subscript(ptr, offset) => {
            let binary = BinaryExpression {
                operator: BinaryOperator::Add,
                left: *ptr,
                right: *offset,
                is_assignment: false,
            };
            let ExpResult::Operand(result) = gen_pointer_addition(instructions, binary, symbols)
            else {
                panic!("Expected operand")
            };
            ExpResult::DereferencedPointer(result)
        }
        Expression::StringLiteral(s_data) => {
            let name = string_name();
            symbols.insert(
                name.clone(),
                Symbol {
                    ctype: CType::Array(CType::Char.into(), s_data.len() as u64 + 1),
                    attrs: SymbolAttr::Constant(Initializer::StringInit {
                        data: s_data.clone(),
                        null_terminated: true,
                    }),
                },
            );
            ExpResult::Operand(Value::Variable(name))
        }
        Expression::SizeOf(expr) => {
            let size = expr.get_type().size();
            ExpResult::Operand(Value::ConstValue(Constant::ULong(size)))
        }
        Expression::SizeOfT(type_t) => {
            ExpResult::Operand(Value::ConstValue(Constant::ULong(type_t.size())))
        }
        Expression::DotAccess(_, _) => todo!("Member access TAC"),
        Expression::Arrow(_, _) => todo!("Member pointer access TAC"),
    }
}

fn gen_increment(
    instructions: &mut Vec<Instruction>,
    expression: TypedExpression,
    increment_type: IncrementType,
    symbols: &mut Symbols,
) -> ExpResult {
    use parse_tree::IncrementType::*;
    let expr_t = expression.get_type();
    let lval = gen_expression(expression, instructions, symbols);
    let src1 = lvalue_convert(instructions, lval.clone(), Some(expr_t.clone()), symbols);
    let old_value = gen_temp_var(expr_t.clone(), symbols);
    // For post operators, save a copy of the initial value to return after the increment
    if matches!(increment_type, PostDecrement | PostIncrement) {
        instructions.push(Instruction::Copy(src1.clone(), old_value.clone()));
    }
    // Normal increment is a binary addition, pointer increment uses AddPtr
    let dst = gen_temp_var(expr_t.clone(), symbols);
    if expr_t.is_pointer() {
        let CType::Pointer(ref ptr_t) = expr_t else { panic!("Not pointer") };
        let inc_val = match increment_type {
            PreIncrement | PostIncrement => Value::ConstValue(Constant::Long(1)),
            PreDecrement | PostDecrement => Value::ConstValue(Constant::Long(-1)),
        };
        instructions.push(Instruction::AddPtr(src1, inc_val, ptr_t.size(), dst.clone()));
    } else {
        let src2 = if expr_t == CType::Double { Constant::Double(1.0) } else { Constant::Int(1) };
        let operator = match increment_type {
            PreIncrement | PostIncrement => BinaryOperator::Add,
            PreDecrement | PostDecrement => BinaryOperator::Subtract,
        };
        instructions.push(Instruction::BinaryOp {
            operator,
            src1,
            src2: Value::ConstValue(src2),
            dst: dst.clone(),
        });
    }
    // Assign result or store if lvalue is pointer
    let result = match &lval {
        ExpResult::Operand(val) => {
            instructions.push(Instruction::Copy(dst, val.clone()));
            lval
        }
        ExpResult::DereferencedPointer(ptr) => {
            instructions.push(Instruction::Store(dst.clone(), ptr.clone()));
            ExpResult::Operand(dst)
        }
    };
    // Return new or old result for pre/post increment
    match increment_type {
        PostDecrement | PostIncrement => ExpResult::Operand(old_value),
        _ => result,
    }
}

fn gen_pointer_addition(
    instructions: &mut Vec<Instruction>,
    binary: BinaryExpression,
    symbols: &mut Symbols,
) -> ExpResult {
    use Instruction::*;
    let left_t = binary.left.get_type();
    let CType::Pointer(ref ptr_t) = left_t else { panic!("Left expr not pointer!") };
    let ptr_size = ptr_t.size();
    let lval = gen_expression(binary.left, instructions, symbols);
    let src = lvalue_convert(instructions, lval.clone(), Some(left_t.clone()), symbols);
    let int_val = gen_expression_and_convert(binary.right, instructions, symbols);
    let dst = gen_temp_var(left_t, symbols);
    instructions.push(AddPtr(src, int_val, ptr_size, dst.clone()));
    if binary.is_assignment {
        match lval {
            ExpResult::Operand(ref val) => {
                instructions.push(Copy(dst, val.clone()));
                lval
            }
            ExpResult::DereferencedPointer(ptr) => {
                instructions.push(Store(dst.clone(), ptr));
                ExpResult::Operand(dst)
            }
        }
    } else {
        ExpResult::Operand(dst)
    }
}

fn gen_pointer_subtraction(
    instructions: &mut Vec<Instruction>,
    binary: BinaryExpression,
    symbols: &mut Symbols,
) -> ExpResult {
    use BinaryOperator::{Divide, Subtract};
    use Instruction::*;
    let CType::Pointer(ptr_t) = binary.left.get_type() else { panic!("Expr not pointer!") };
    let ptr_size = ptr_t.size();
    let src1 = gen_expression_and_convert(binary.left, instructions, symbols);
    let src2 = gen_expression_and_convert(binary.right, instructions, symbols);
    // Subtract into temp var
    let tmp = gen_temp_var(CType::Long, symbols);
    instructions.push(BinaryOp {
        src1,
        src2,
        dst: tmp.clone(),
        operator: Subtract,
    });
    // Divide into final result
    let result = gen_temp_var(CType::Long, symbols);
    let src2 = Value::ConstValue(Constant::Long(ptr_size as i64));
    instructions.push(BinaryOp {
        src1: tmp,
        src2,
        dst: result.clone(),
        operator: Divide,
    });
    ExpResult::Operand(result)
}

fn gen_compound_assignment(
    instructions: &mut Vec<Instruction>,
    binary: BinaryExpression,
    symbols: &mut Symbols,
) -> ExpResult {
    use Instruction::*;
    let left_t = binary.left.get_type();
    let expr_t = match binary.operator {
        BinaryOperator::LeftShift | BinaryOperator::RightShift => left_t.clone(),
        _ => get_common_type(&binary.left, &binary.right).expect("Compound type error!"),
    };
    // Generate and cast lvalue to common type
    let lval = gen_expression(binary.left, instructions, symbols);
    let src1 = lvalue_convert(instructions, lval.clone(), Some(left_t.clone()), symbols);
    let src1 = gen_cast(instructions, expr_t.clone(), left_t.clone(), src1, symbols);
    // Generate rvalue/temp var for dst, push binary op and cast result
    let src2 = gen_expression_and_convert(binary.right, instructions, symbols);
    let dst = gen_temp_var(expr_t.clone(), symbols);
    let operator = binary.operator;
    instructions.push(BinaryOp { operator, src1, src2, dst: dst.clone() });
    let result = gen_cast(instructions, left_t, expr_t.clone(), dst, symbols);
    // Handle assigning to pointer.
    match lval {
        ExpResult::Operand(ref val) => {
            instructions.push(Copy(result, val.clone()));
            lval
        }
        ExpResult::DereferencedPointer(ptr) => {
            instructions.push(Store(result.clone(), ptr));
            ExpResult::Operand(result)
        }
    }
}

fn gen_cast(
    instructions: &mut Vec<Instruction>,
    new_type: CType,
    old_type: CType,
    val: Value,
    symbols: &mut Symbols,
) -> Value {
    if new_type == old_type {
        return val;
    }
    if new_type == CType::Void {
        return Value::Variable(String::from("DummyVar"));
    }
    let dst = gen_temp_var(new_type.clone(), symbols);
    if old_type == CType::Double && new_type.is_signed() {
        instructions.push(Instruction::DoubleToInt(val, dst.clone()));
    } else if old_type == CType::Double {
        instructions.push(Instruction::DoubleToUInt(val, dst.clone()));
    } else if new_type == CType::Double && old_type.is_signed() {
        instructions.push(Instruction::IntToDouble(val, dst.clone()))
    } else if new_type == CType::Double {
        instructions.push(Instruction::UIntToDouble(val, dst.clone()))
    } else if new_type.size() == old_type.size() {
        instructions.push(Instruction::Copy(val, dst.clone()));
    } else if new_type.size() < old_type.size() {
        instructions.push(Instruction::Truncate(val, dst.clone()));
    } else if old_type.is_signed() {
        instructions.push(Instruction::SignExtend(val, dst.clone()));
    } else {
        instructions.push(Instruction::ZeroExtend(val, dst.clone()));
    }
    dst
}

fn gen_short_circuit(
    instructions: &mut Vec<Instruction>,
    binary: BinaryExpression,
    symbols: &mut Symbols,
) -> ExpResult {
    use Instruction::*;
    let operator = binary.operator;
    let left = binary.left;
    let right = binary.right;
    let (jump_type, label_type) = match operator {
        BinaryOperator::LogicalAnd => (JumpType::JumpIfZero, false),
        BinaryOperator::LogicalOr => (JumpType::JumpIfNotZero, true),
        _ => panic!("Expected a short circuiting operator!"),
    };
    let target = gen_label(&label_type.to_string());
    let end = gen_label("end");
    let dst = gen_temp_var(CType::Int, symbols);
    let v1 = gen_expression_and_convert(left, instructions, symbols);
    instructions.push(JumpCond {
        jump_type,
        condition: v1,
        target: target.clone(),
    });
    let v2 = gen_expression_and_convert(right, instructions, symbols);
    instructions.push(JumpCond {
        jump_type,
        condition: v2,
        target: target.clone(),
    });
    instructions.push(Copy(Value::ConstValue(Constant::Int(!label_type as i64)), dst.clone()));
    instructions.push(Jump(end.clone()));
    instructions.push(Label(target));
    instructions.push(Copy(Value::ConstValue(Constant::Int(label_type as i64)), dst.clone()));
    instructions.push(Label(end));
    ExpResult::Operand(dst)
}

fn gen_temp_var(ctype: CType, symbols: &mut Symbols) -> Value {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let tmp_name = format!("tmp.{}", COUNTER.fetch_add(1, Ordering::Relaxed));
    symbols.insert(tmp_name.clone(), Symbol { ctype, attrs: SymbolAttr::Local });
    Value::Variable(tmp_name)
}

pub fn gen_label(label_type: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("label_{}.{}", label_type, COUNTER.fetch_add(1, Ordering::Relaxed))
}
