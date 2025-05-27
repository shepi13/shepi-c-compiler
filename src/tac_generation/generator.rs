// Generates Three Adress Code Intermediate representation (Step between Parser AST and Assembly AST)

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    parse::parse_tree::{self, BinaryOperator, CType, StorageClass},
    validate::type_check::{Initializer, StaticInitializer, Symbol, SymbolAttr, Symbols, get_type},
};

pub type Program = Vec<TopLevelDecl>;
#[derive(Debug, Clone)]
pub enum TopLevelDecl {
    Function(Function),
    StaticDecl(StaticVariable),
}
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Value>,
    pub instructions: Vec<Instruction>,
    pub global: bool,
}
#[derive(Debug, Clone)]
pub struct StaticVariable {
    pub identifier: String,
    pub global: bool,
    pub initializer: Initializer,
    pub ctype: CType,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Return(Value),
    SignExtend(Value, Value),
    ZeroExtend(Value, Value),
    Truncate(Value, Value),
    DoubleToInt(Value, Value),
    DoubleToUInt(Value, Value),
    IntToDouble(Value, Value),
    UIntToDouble(Value, Value),
    UnaryOp(InstructionUnary),
    BinaryOp(InstructionBinary),
    Copy(InstructionCopy),
    Label(String),
    Jump(String),
    JumpCond(InstructionJump),
    Function(String, Vec<Value>, Value),
}
#[derive(Debug, Clone)]
pub struct InstructionUnary {
    pub operator: UnaryOperator,
    pub src: Value,
    pub dst: Value,
}
#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    Complement,
    Negate,
    LogicalNot,
}
impl InstructionUnary {
    fn from(operator: parse_tree::UnaryOperator, src: Value, dst: Value) -> Self {
        let operator = match operator {
            parse_tree::UnaryOperator::Complement => UnaryOperator::Complement,
            parse_tree::UnaryOperator::LogicalNot => UnaryOperator::LogicalNot,
            parse_tree::UnaryOperator::Negate => UnaryOperator::Negate,
            _ => panic!("Invalid TAC operator"),
        };
        Self { operator, src, dst }
    }
}
#[derive(Debug, Clone)]
pub struct InstructionBinary {
    pub operator: BinaryOperator,
    pub src1: Value,
    pub src2: Value,
    pub dst: Value,
}
#[derive(Debug, Clone)]
pub struct InstructionCopy {
    pub src: Value,
    pub dst: Value,
}
#[derive(Debug, Clone)]
pub struct InstructionJump {
    pub jump_type: JumpType,
    pub condition: Value,
    pub target: String,
}
#[derive(Debug, Clone, Copy)]
pub enum JumpType {
    JumpIfZero,
    JumpIfNotZero,
}

#[derive(Debug, Clone)]
pub enum Value {
    ConstValue(parse_tree::Constant),
    Variable(String),
}

pub fn gen_tac_ast(parser_ast: parse_tree::Program, symbols: &mut Symbols) -> Program {
    let mut program: Program = Vec::new();

    for decl in parser_ast {
        if let parse_tree::Declaration::Function(function) = decl {
            program.push(TopLevelDecl::Function(gen_function(function, symbols)));
        }
    }
    for (name, entry) in symbols {
        if let SymbolAttr::Static(var_attrs) = &entry.attrs {
            match &var_attrs.init {
                &StaticInitializer::Initialized(initializer) => {
                    program.push(TopLevelDecl::StaticDecl(StaticVariable {
                        identifier: name.clone(),
                        global: var_attrs.global,
                        initializer,
                        ctype: entry.ctype.clone(),
                    }));
                }
                StaticInitializer::Tentative => {
                    let initializer = match &entry.ctype {
                        CType::Int => Initializer::Int(0),
                        CType::Long => Initializer::Long(0),
                        CType::UnsignedInt => Initializer::UnsignedInt(0),
                        CType::UnsignedLong => Initializer::UnsignedLong(0),
                        CType::Double => Initializer::Double(0.0),
                        _ => panic!("Not a variable"),
                    };
                    program.push(TopLevelDecl::StaticDecl(StaticVariable {
                        identifier: name.clone(),
                        global: var_attrs.global,
                        initializer,
                        ctype: entry.ctype.clone(),
                    }));
                }
                StaticInitializer::None => (),
            }
        }
    }
    program
}
fn gen_function(function: parse_tree::FunctionDeclaration, symbols: &mut Symbols) -> Function {
    let mut instructions: Vec<Instruction> = Vec::new();
    if let Some(body) = function.body {
        gen_block(body, &mut instructions, symbols);
        instructions.push(Instruction::Return(Value::ConstValue(parse_tree::Constant::Int(0))));
    }
    let global = symbols[&function.name].get_function_attrs().global;
    let params = function.params.into_iter().map(Value::Variable).collect();
    Function {
        name: function.name,
        params,
        global,
        instructions,
    }
}
fn gen_block(block: parse_tree::Block, instructions: &mut Vec<Instruction>, symbols: &mut Symbols) {
    for block_item in block {
        match block_item {
            parse_tree::BlockItem::StatementItem(statement) => {
                gen_instructions(statement, instructions, symbols)
            }
            parse_tree::BlockItem::DeclareItem(parse_tree::Declaration::Variable(decl)) => {
                gen_declaration(decl, instructions, symbols);
            }
            parse_tree::BlockItem::DeclareItem(parse_tree::Declaration::Function(_)) => (),
        }
    }
}
fn gen_declaration(
    declaration: parse_tree::VariableDeclaration,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) {
    if declaration.storage == Some(StorageClass::Extern)
        || declaration.storage == Some(StorageClass::Static)
    {
        return;
    }
    if let Some(value) = declaration.value {
        let result = gen_expression(value, instructions, symbols);
        instructions.push(Instruction::Copy(InstructionCopy {
            src: result,
            dst: Value::Variable(declaration.name.to_string()),
        }));
    }
}
fn gen_instructions(
    statement: parse_tree::Statement,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) {
    match statement {
        parse_tree::Statement::Return(value) => {
            let dst = gen_expression(value, instructions, symbols);
            instructions.push(Instruction::Return(dst));
        }
        parse_tree::Statement::Null => (),
        parse_tree::Statement::ExprStmt(value) => {
            gen_expression(value, instructions, symbols);
        }
        parse_tree::Statement::If(condition, if_true, if_false) => {
            let end_label = gen_label("end");
            let else_label = gen_label("else");
            let condition = gen_expression(condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfZero,
                condition,
                target: else_label.clone(),
            }));
            gen_instructions(*if_true, instructions, symbols);
            instructions.push(Instruction::Jump(end_label.clone()));
            instructions.push(Instruction::Label(else_label));
            if let Some(false_statement) = *if_false {
                gen_instructions(false_statement, instructions, symbols);
            }
            instructions.push(Instruction::Label(end_label));
        }
        parse_tree::Statement::Goto(target) => {
            instructions.push(Instruction::Jump(target.to_string()));
        }
        parse_tree::Statement::Label(name, statement) => {
            instructions.push(Instruction::Label(name.to_string()));
            gen_instructions(*statement, instructions, symbols);
        }
        parse_tree::Statement::Compound(block) => gen_block(block, instructions, symbols),
        parse_tree::Statement::Break(name) => {
            let target = format!("break_{}", name);
            instructions.push(Instruction::Jump(target));
        }
        parse_tree::Statement::Continue(name) => {
            let target = format!("continue_{}", name);
            instructions.push(Instruction::Jump(target));
        }
        parse_tree::Statement::DoWhile(loop_data) => {
            let start = format!("start_{}", loop_data.label);
            instructions.push(Instruction::Label(start.clone()));
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Label(format!("continue_{}", loop_data.label)));
            let result = gen_expression(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfNotZero,
                condition: result,
                target: start,
            }));
            instructions.push(Instruction::Label(format!("break_{}", loop_data.label)));
        }
        parse_tree::Statement::While(loop_data) => {
            let break_label = format!("break_{}", loop_data.label);
            let continue_label = format!("continue_{}", loop_data.label);
            instructions.push(Instruction::Label(continue_label.clone()));
            let result = gen_expression(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfZero,
                condition: result,
                target: break_label.clone(),
            }));
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Jump(continue_label));
            instructions.push(Instruction::Label(break_label));
        }
        parse_tree::Statement::For(init, loop_data, post_loop) => {
            match init {
                parse_tree::ForInit::Decl(decl) => {
                    gen_declaration(decl, instructions, symbols);
                }
                parse_tree::ForInit::Expr(Some(expr)) => {
                    gen_expression(expr, instructions, symbols);
                }
                _ => (),
            };
            let break_label = format!("break_{}", loop_data.label);
            let start_label = format!("start_{}", loop_data.label);
            instructions.push(Instruction::Label(start_label.clone()));
            let condition = gen_expression(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfZero,
                condition,
                target: break_label.clone(),
            }));
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Label(format!("continue_{}", loop_data.label)));
            if let Some(post) = post_loop {
                gen_expression(post, instructions, symbols);
            }
            instructions.push(Instruction::Jump(start_label));
            instructions.push(Instruction::Label(break_label));
        }
        parse_tree::Statement::Switch(switch) => {
            let src1 = gen_expression(switch.condition, instructions, symbols);
            let dst = gen_temp_var(CType::Int, symbols);
            for case in switch.cases {
                let src2 = gen_expression(case.1, instructions, symbols);
                instructions.push(Instruction::BinaryOp(InstructionBinary {
                    operator: BinaryOperator::IsEqual,
                    src1: src1.clone(),
                    src2: src2.clone(),
                    dst: dst.clone(),
                }));
                instructions.push(Instruction::JumpCond(InstructionJump {
                    jump_type: JumpType::JumpIfNotZero,
                    condition: dst.clone(),
                    target: case.0,
                }));
            }
            if let Some(target) = switch.default {
                instructions.push(Instruction::Jump(target));
            } else {
                instructions.push(Instruction::Jump(format!("break_{}", switch.label)));
            }
            gen_instructions(*switch.statement, instructions, symbols);
            instructions.push(Instruction::Label(format!("break_{}", switch.label)));
        }
        parse_tree::Statement::Case(_, _) | parse_tree::Statement::Default(_) => {
            panic!("Compiler error: case/default should be replaced in typecheck pass")
        }
    }
}
fn gen_expression(
    expression: parse_tree::TypedExpression,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) -> Value {
    let expr_type = || expression.ctype.expect("Undefined type!");
    match expression.expr {
        parse_tree::Expression::Constant(constexpr) => Value::ConstValue(constexpr),
        parse_tree::Expression::Unary(
            parse_tree::UnaryOperator::Increment(increment_type),
            expr,
        ) => {
            use parse_tree::Constant;
            use parse_tree::IncrementType::*;
            let is_double = get_type(&expr) == CType::Double;
            let dst = gen_expression(*expr, instructions, symbols);
            let operator = match increment_type {
                PreIncrement | PostIncrement => BinaryOperator::Add,
                PreDecrement | PostDecrement => BinaryOperator::Subtract,
            };
            let src2 = if is_double { Constant::Double(1.0) } else { Constant::Int(1) };
            let bin_instruction = Instruction::BinaryOp(InstructionBinary {
                operator,
                src1: dst.clone(),
                src2: Value::ConstValue(src2),
                dst: dst.clone(),
            });
            match increment_type {
                PreDecrement | PreIncrement => {
                    instructions.push(bin_instruction);
                    dst
                }
                PostDecrement | PostIncrement => {
                    let old_value = gen_temp_var(expr_type(), symbols);
                    instructions.push(Instruction::Copy(InstructionCopy {
                        src: dst.clone(),
                        dst: old_value.clone(),
                    }));
                    instructions.push(bin_instruction);
                    old_value
                }
            }
        }
        parse_tree::Expression::Unary(operator, expr) => {
            let src = gen_expression(*expr, instructions, symbols);
            let dst = gen_temp_var(expr_type(), symbols);
            instructions.push(Instruction::UnaryOp(InstructionUnary::from(
                operator,
                src,
                dst.clone(),
            )));
            dst
        }
        parse_tree::Expression::Binary(operator) => {
            // Short circuiting needs special handling
            if let parse_tree::BinaryOperator::LogicalAnd | parse_tree::BinaryOperator::LogicalOr =
                operator.operator
            {
                return gen_short_circuit(
                    instructions,
                    operator.operator,
                    operator.left,
                    operator.right,
                    symbols,
                );
            };
            let src1 = gen_expression(operator.left, instructions, symbols);
            let src2 = gen_expression(operator.right, instructions, symbols);
            let dst = if operator.is_assignment {
                src1.clone()
            } else {
                gen_temp_var(expr_type(), symbols)
            };
            instructions.push(Instruction::BinaryOp(InstructionBinary {
                operator: operator.operator.clone(),
                src1,
                src2,
                dst: dst.clone(),
            }));
            dst
        }
        parse_tree::Expression::Variable(name) => Value::Variable(name.to_string()),
        parse_tree::Expression::Assignment(assignment) => {
            let result = gen_expression(assignment.right, instructions, symbols);
            match &assignment.left.expr {
                parse_tree::Expression::Variable(name) => {
                    instructions.push(Instruction::Copy(InstructionCopy {
                        src: result,
                        dst: Value::Variable(name.to_string()),
                    }));
                    Value::Variable(name.to_string())
                }
                _ => panic!("Left hand side of assignment should be variable!"),
            }
        }
        parse_tree::Expression::Condition(condition) => {
            let dst = gen_temp_var(expr_type(), symbols);
            let end_label = gen_label("cond_end");
            let e2_label = gen_label("cond_e2");
            let cond = gen_expression(condition.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfZero,
                condition: cond,
                target: e2_label.clone(),
            }));
            let e1 = gen_expression(condition.if_true, instructions, symbols);
            instructions.push(Instruction::Copy(InstructionCopy { src: e1, dst: dst.clone() }));
            instructions.push(Instruction::Jump(end_label.clone()));
            instructions.push(Instruction::Label(e2_label));
            let e2 = gen_expression(condition.if_false, instructions, symbols);
            instructions.push(Instruction::Copy(InstructionCopy { src: e2, dst: dst.clone() }));
            instructions.push(Instruction::Label(end_label));
            dst
        }
        parse_tree::Expression::FunctionCall(name, args) => {
            let results =
                args.into_iter().map(|arg| gen_expression(arg, instructions, symbols)).collect();
            let dst = gen_temp_var(expr_type(), symbols);
            instructions.push(Instruction::Function(name.to_string(), results, dst.clone()));
            dst
        }
        parse_tree::Expression::Cast(new_type, castexpr) => {
            let old_type = get_type(&castexpr);
            let result = gen_expression(*castexpr, instructions, symbols);
            if new_type == old_type {
                return result;
            }
            let dst = gen_temp_var(new_type.clone(), symbols);

            if old_type == CType::Double && new_type.is_signed() {
                instructions.push(Instruction::DoubleToInt(result, dst.clone()));
            } else if old_type == CType::Double {
                instructions.push(Instruction::DoubleToUInt(result, dst.clone()));
            } else if new_type == CType::Double && old_type.is_signed() {
                instructions.push(Instruction::IntToDouble(result, dst.clone()))
            } else if new_type == CType::Double {
                instructions.push(Instruction::UIntToDouble(result, dst.clone()))
            } else if new_type.size() == old_type.size() {
                instructions
                    .push(Instruction::Copy(InstructionCopy { src: result, dst: dst.clone() }));
            } else if new_type.size() < old_type.size() {
                instructions.push(Instruction::Truncate(result, dst.clone()));
            } else if old_type.is_signed() {
                instructions.push(Instruction::SignExtend(result, dst.clone()));
            } else {
                instructions.push(Instruction::ZeroExtend(result, dst.clone()));
            }
            dst
        }
        parse_tree::Expression::AddrOf(_) | parse_tree::Expression::Dereference(_) => {
            panic!("Not implemented!")
        }
    }
}

fn gen_short_circuit(
    instructions: &mut Vec<Instruction>,
    operator: parse_tree::BinaryOperator,
    left: parse_tree::TypedExpression,
    right: parse_tree::TypedExpression,
    symbols: &mut Symbols,
) -> Value {
    let (jump_type, label_type) = match operator {
        parse_tree::BinaryOperator::LogicalAnd => (JumpType::JumpIfZero, false),
        parse_tree::BinaryOperator::LogicalOr => (JumpType::JumpIfNotZero, true),
        _ => panic!("Expected a short circuiting operator!"),
    };
    let target = gen_label(&label_type.to_string());
    let end = gen_label("end");
    let dst = gen_temp_var(CType::Int, symbols);
    let v1 = gen_expression(left, instructions, symbols);
    instructions.push(Instruction::JumpCond(InstructionJump {
        jump_type: jump_type.clone(),
        condition: v1,
        target: target.clone(),
    }));
    let v2 = gen_expression(right, instructions, symbols);
    instructions.push(Instruction::JumpCond(InstructionJump {
        jump_type,
        condition: v2,
        target: target.clone(),
    }));
    instructions.push(Instruction::Copy(InstructionCopy {
        src: Value::ConstValue(parse_tree::Constant::Int(!label_type as i64)),
        dst: dst.clone(),
    }));
    instructions.push(Instruction::Jump(end.clone()));
    instructions.push(Instruction::Label(target));
    instructions.push(Instruction::Copy(InstructionCopy {
        src: Value::ConstValue(parse_tree::Constant::Int(label_type as i64)),
        dst: dst.clone(),
    }));
    instructions.push(Instruction::Label(end));
    dst
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
