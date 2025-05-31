// Generates Three Adress Code Intermediate representation (Step between Parser AST and Assembly AST)

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    parse::parse_tree::{self, BinaryOperator, CType, StorageClass, VariableInitializer},
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
    GetAddress(Value, Value),
    Load(Value, Value),
    Store(Value, Value),
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
                StaticInitializer::Initialized(initializer) => {
                    program.push(TopLevelDecl::StaticDecl(StaticVariable {
                        identifier: name.clone(),
                        global: var_attrs.global,
                        initializer: initializer[0],
                        ctype: entry.ctype.clone(),
                    }));
                }
                StaticInitializer::Tentative => {
                    let initializer = match &entry.ctype {
                        CType::Double => Initializer::Double(0.0),
                        CType::Int
                        | CType::Long
                        | CType::UnsignedInt
                        | CType::UnsignedLong
                        | CType::Pointer(_) => Initializer::ZeroInit(entry.ctype.size()),
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
    if let Some(VariableInitializer::SingleElem(value)) = declaration.init {
        let result = gen_expression_and_convert(value, instructions, symbols);
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
            let dst = gen_expression_and_convert(value, instructions, symbols);
            instructions.push(Instruction::Return(dst));
        }
        parse_tree::Statement::Null => (),
        parse_tree::Statement::ExprStmt(value) => {
            gen_expression_and_convert(value, instructions, symbols);
        }
        parse_tree::Statement::If(condition, if_true, if_false) => {
            let end_label = gen_label("end");
            let else_label = gen_label("else");
            let condition = gen_expression_and_convert(condition, instructions, symbols);
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
            let result = gen_expression_and_convert(loop_data.condition, instructions, symbols);
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
            let result = gen_expression_and_convert(loop_data.condition, instructions, symbols);
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
                    gen_expression_and_convert(expr, instructions, symbols);
                }
                _ => (),
            };
            let break_label = format!("break_{}", loop_data.label);
            let start_label = format!("start_{}", loop_data.label);
            instructions.push(Instruction::Label(start_label.clone()));
            let condition = gen_expression_and_convert(loop_data.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfZero,
                condition,
                target: break_label.clone(),
            }));
            gen_instructions(*loop_data.body, instructions, symbols);
            instructions.push(Instruction::Label(format!("continue_{}", loop_data.label)));
            if let Some(post) = post_loop {
                gen_expression_and_convert(post, instructions, symbols);
            }
            instructions.push(Instruction::Jump(start_label));
            instructions.push(Instruction::Label(break_label));
        }
        parse_tree::Statement::Switch(switch) => {
            let src1 = gen_expression_and_convert(switch.condition, instructions, symbols);
            let dst = gen_temp_var(CType::Int, symbols);
            for case in switch.cases {
                let src2 = gen_expression_and_convert(case.1, instructions, symbols);
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
    expression: parse_tree::TypedExpression,
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
    expression: parse_tree::TypedExpression,
    instructions: &mut Vec<Instruction>,
    symbols: &mut Symbols,
) -> ExpResult {
    let expr_type = || expression.ctype.expect("Undefined type!");
    match expression.expr {
        parse_tree::Expression::Constant(constexpr) => {
            ExpResult::Operand(Value::ConstValue(constexpr))
        }
        parse_tree::Expression::Unary(
            parse_tree::UnaryOperator::Increment(increment_type),
            expr,
        ) => {
            use parse_tree::Constant;
            use parse_tree::IncrementType::*;
            let inner_type = get_type(&expr);
            let lval = gen_expression(*expr, instructions, symbols);
            let operator = match increment_type {
                PreIncrement | PostIncrement => BinaryOperator::Add,
                PreDecrement | PostDecrement => BinaryOperator::Subtract,
            };
            let src1 =
                lvalue_convert(instructions, lval.clone(), Some(inner_type.clone()), symbols);
            let old_value = gen_temp_var(inner_type.clone(), symbols);
            if matches!(increment_type, PostDecrement | PostIncrement) {
                instructions.push(Instruction::Copy(InstructionCopy {
                    src: src1.clone(),
                    dst: old_value.clone(),
                }));
            }
            let src2 =
                if inner_type == CType::Double { Constant::Double(1.0) } else { Constant::Int(1) };
            let dst = gen_temp_var(inner_type.clone(), symbols);
            instructions.push(Instruction::BinaryOp(InstructionBinary {
                operator,
                src1,
                src2: Value::ConstValue(src2),
                dst: dst.clone(),
            }));
            let result = match &lval {
                ExpResult::Operand(val) => {
                    instructions
                        .push(Instruction::Copy(InstructionCopy { src: dst, dst: val.clone() }));
                    lval
                }
                ExpResult::DereferencedPointer(ptr) => {
                    instructions.push(Instruction::Store(dst.clone(), ptr.clone()));
                    ExpResult::Operand(dst)
                }
            };
            match increment_type {
                PostDecrement | PostIncrement => ExpResult::Operand(old_value),
                _ => result,
            }
        }
        parse_tree::Expression::Unary(operator, expr) => {
            let src = gen_expression_and_convert(*expr, instructions, symbols);
            let dst = gen_temp_var(expr_type(), symbols);
            instructions.push(Instruction::UnaryOp(InstructionUnary::from(
                operator,
                src,
                dst.clone(),
            )));
            ExpResult::Operand(dst)
        }
        parse_tree::Expression::Binary(operator) => {
            // Short circuiting needs special handling
            if let parse_tree::BinaryOperator::LogicalAnd | parse_tree::BinaryOperator::LogicalOr =
                operator.operator
            {
                return ExpResult::Operand(gen_short_circuit(
                    instructions,
                    operator.operator,
                    operator.left,
                    operator.right,
                    symbols,
                ));
            };
            // Handle compound assignment, we have to do some type checking here as we cannot
            // calculate lvalues twice (if they call functions), so need to generate our casts
            // with tmp variables
            if operator.is_assignment {
                let expr_t = expr_type();
                let left_t = get_type(&operator.left);
                // Generate and cast lvalue to common type
                let lval = gen_expression(operator.left, instructions, symbols);
                let src1 =
                    lvalue_convert(instructions, lval.clone(), Some(expr_t.clone()), symbols);
                let src1 = gen_cast(instructions, expr_t.clone(), left_t.clone(), src1, symbols);
                // Generate rvalue/temp var for dst, push binary op
                let src2 = gen_expression_and_convert(operator.right, instructions, symbols);
                let dst = gen_temp_var(expr_t.clone(), symbols);
                instructions.push(Instruction::BinaryOp(InstructionBinary {
                    operator: operator.operator,
                    src1,
                    src2,
                    dst: dst.clone(),
                }));
                // Cast result to assignment type
                let result = gen_cast(instructions, left_t, expr_t.clone(), dst, symbols);
                // Handle assigning to pointer.
                match &lval {
                    ExpResult::Operand(val) => {
                        instructions.push(Instruction::Copy(InstructionCopy {
                            src: result,
                            dst: val.clone(),
                        }));
                        lval
                    }
                    ExpResult::DereferencedPointer(ptr) => {
                        instructions.push(Instruction::Store(result.clone(), ptr.clone()));
                        ExpResult::Operand(result)
                    }
                }
            } else {
                let src1 = gen_expression_and_convert(operator.left, instructions, symbols);
                let src2 = gen_expression_and_convert(operator.right, instructions, symbols);
                let dst = gen_temp_var(expr_type(), symbols);
                instructions.push(Instruction::BinaryOp(InstructionBinary {
                    operator: operator.operator,
                    src1,
                    src2,
                    dst: dst.clone(),
                }));
                ExpResult::Operand(dst)
            }
        }
        parse_tree::Expression::Variable(name) => {
            ExpResult::Operand(Value::Variable(name.to_string()))
        }
        parse_tree::Expression::Assignment(assignment) => {
            let lval = gen_expression(assignment.left, instructions, symbols);
            let rval = gen_expression_and_convert(assignment.right, instructions, symbols);
            match &lval {
                ExpResult::Operand(val) => {
                    instructions
                        .push(Instruction::Copy(InstructionCopy { src: rval, dst: val.clone() }));
                    lval
                }
                ExpResult::DereferencedPointer(ptr) => {
                    instructions.push(Instruction::Store(rval.clone(), ptr.clone()));
                    ExpResult::Operand(rval)
                }
            }
        }
        parse_tree::Expression::Condition(condition) => {
            let dst = gen_temp_var(expr_type(), symbols);
            let end_label = gen_label("cond_end");
            let e2_label = gen_label("cond_e2");
            let cond = gen_expression_and_convert(condition.condition, instructions, symbols);
            instructions.push(Instruction::JumpCond(InstructionJump {
                jump_type: JumpType::JumpIfZero,
                condition: cond,
                target: e2_label.clone(),
            }));
            let e1 = gen_expression_and_convert(condition.if_true, instructions, symbols);
            instructions.push(Instruction::Copy(InstructionCopy { src: e1, dst: dst.clone() }));
            instructions.push(Instruction::Jump(end_label.clone()));
            instructions.push(Instruction::Label(e2_label));
            let e2 = gen_expression_and_convert(condition.if_false, instructions, symbols);
            instructions.push(Instruction::Copy(InstructionCopy { src: e2, dst: dst.clone() }));
            instructions.push(Instruction::Label(end_label));
            ExpResult::Operand(dst)
        }
        parse_tree::Expression::FunctionCall(name, args) => {
            let results = args
                .into_iter()
                .map(|arg| gen_expression_and_convert(arg, instructions, symbols))
                .collect();
            let dst = gen_temp_var(expr_type(), symbols);
            instructions.push(Instruction::Function(name.to_string(), results, dst.clone()));
            ExpResult::Operand(dst)
        }
        parse_tree::Expression::Cast(new_type, castexpr) => {
            let old_type = get_type(&castexpr);
            let result = gen_expression_and_convert(*castexpr, instructions, symbols);
            let dst = gen_cast(instructions, new_type, old_type, result, symbols);
            ExpResult::Operand(dst)
        }
        parse_tree::Expression::Dereference(inner) => {
            let result = gen_expression_and_convert(*inner, instructions, symbols);
            ExpResult::DereferencedPointer(result)
        }
        parse_tree::Expression::AddrOf(inner) => {
            let result = gen_expression(*inner, instructions, symbols);
            match result {
                ExpResult::Operand(val) => {
                    let dst = gen_temp_var(expr_type(), symbols);
                    instructions.push(Instruction::GetAddress(val, dst.clone()));
                    ExpResult::Operand(dst)
                }
                ExpResult::DereferencedPointer(ptr) => ExpResult::Operand(ptr),
            }
        }
        parse_tree::Expression::Subscript(_, _) => panic!("Not implemented!"),
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
        instructions.push(Instruction::Copy(InstructionCopy { src: val, dst: dst.clone() }));
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
    let v1 = gen_expression_and_convert(left, instructions, symbols);
    instructions.push(Instruction::JumpCond(InstructionJump {
        jump_type,
        condition: v1,
        target: target.clone(),
    }));
    let v2 = gen_expression_and_convert(right, instructions, symbols);
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
