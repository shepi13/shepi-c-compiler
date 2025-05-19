// Generates Three Adress Code Intermediate representation (Step between Parser AST and Assembly AST)

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::parser::{self, UnaryOperator};


pub type Program<'a> = Vec<Function<'a>>;
#[derive(Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub params: &'a Vec<String>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug)]
pub enum Instruction {
    RETURN(Value),
    UNARYOP(InstructionUnary),
    BINARYOP(InstructionBinary),
    COPY(InstructionCopy),
    LABEL(String),
    JUMP(String),
    JUMPCOND(InstructionJump),
    FUNCTION(String, Vec<Value>, Value),
}
#[derive(Debug)]
pub struct InstructionUnary {
    pub operator: UnaryOperator,
    pub src: Value,
    pub dst: Value,
}
#[derive(Debug)]
pub struct InstructionBinary {
    pub operator: parser::BinaryOperator,
    pub src1: Value,
    pub src2: Value,
    pub dst: Value,
}
#[derive(Debug)]
pub struct InstructionCopy {
    pub src: Value,
    pub dst: Value,
}
#[derive(Debug)]
pub struct InstructionJump {
    pub jump_type: JumpType,
    pub condition: Value,
    pub target: String,
}
#[derive(Debug, Clone)]
pub enum JumpType {
    JUMPIFZERO,
    JUMPIFNOTZERO,
}

#[derive(Debug, Clone)]
pub enum Value {
    CONSTANT(u32),
    VARIABLE(String),
}

pub fn gen_tac_ast<'a>(parser_ast: &'a parser::Program<'a>) -> Program<'a> {
    let mut program: Program = Vec::new();
    for function in parser_ast {
        program.push(gen_function(function));
    }
    program
}
fn gen_function<'a>(function: &'a parser::FunctionDeclaration<'a>) -> Function<'a> {
    let mut instructions: Vec<Instruction> = Vec::new();
    if let Some(body) = &function.body {
        gen_block(&body, &mut instructions);
        instructions.push(Instruction::RETURN(Value::CONSTANT(0)));
    }
    Function {
        name: function.name,
        params: &function.params,
        instructions,
    }
}
fn gen_block(block: &parser::Block, instructions: &mut Vec<Instruction>) {
    for block_item in block {
        match block_item {
            parser::BlockItem::STATEMENT(statement) => gen_instructions(&statement, instructions),
            parser::BlockItem::DECLARATION(parser::Declaration::VARIABLE(decl)) => {
                gen_declaration(decl, instructions);
            }
            parser::BlockItem::DECLARATION(parser::Declaration::FUNCTION(_)) => ()
        }
    }
}
fn gen_declaration(declaration: &parser::VariableDeclaration, instructions: &mut Vec<Instruction>) {
    if let Some(value) = &declaration.value {
        let result = gen_expression(&value, instructions);
        instructions.push(Instruction::COPY(InstructionCopy {
            src: result,
            dst: Value::VARIABLE(declaration.name.to_string()),
        }));
    }
}
fn gen_instructions(statement: &parser::Statement, instructions: &mut Vec<Instruction>) {
    match statement {
        parser::Statement::RETURN(value) => {
            let dst = gen_expression(value, instructions);
            instructions.push(Instruction::RETURN(dst));
        }
        parser::Statement::NULL => (),
        parser::Statement::EXPRESSION(value) => {
            gen_expression(value, instructions);
        }
        parser::Statement::IF(condition, if_true, if_false) => {
            let end_label = gen_label("end");
            let else_label = gen_label("else");
            let condition = gen_expression(condition, instructions);
            instructions.push(Instruction::JUMPCOND(InstructionJump {
                jump_type: JumpType::JUMPIFZERO,
                condition,
                target: else_label.clone(),
            }));
            gen_instructions(&if_true, instructions);
            instructions.push(Instruction::JUMP(end_label.clone()));
            instructions.push(Instruction::LABEL(else_label));
            if let Some(false_statement) = if_false.as_ref() {
                gen_instructions(&false_statement, instructions);
            }
            instructions.push(Instruction::LABEL(end_label));
        }
        parser::Statement::GOTO(target) => {
            instructions.push(Instruction::JUMP(target.to_string()));
        }
        parser::Statement::LABEL(name, statement) => {
            instructions.push(Instruction::LABEL(name.to_string()));
            gen_instructions(statement, instructions);
        }
        parser::Statement::COMPOUND(block) => gen_block(block, instructions),
        parser::Statement::BREAK(name) => {
            let target = format!("break_{}", name);
            instructions.push(Instruction::JUMP(target));
        }
        parser::Statement::CONTINUE(name) => {
            let target = format!("continue_{}", name);
            instructions.push(Instruction::JUMP(target));
        }
        parser::Statement::DOWHILE(loop_data) => {
            let start = format!("start_{}", loop_data.label);
            instructions.push(Instruction::LABEL(start.clone()));
            gen_instructions(&loop_data.body, instructions);
            instructions.push(Instruction::LABEL(format!("continue_{}", loop_data.label)));
            let result = gen_expression(&loop_data.condition, instructions);
            instructions.push(Instruction::JUMPCOND(InstructionJump {
                jump_type: JumpType::JUMPIFNOTZERO,
                condition: result,
                target: start,
            }));
            instructions.push(Instruction::LABEL(format!("break_{}", loop_data.label)));
        }
        parser::Statement::WHILE(loop_data) => {
            let break_label = format!("break_{}", loop_data.label);
            let continue_label = format!("continue_{}", loop_data.label);
            instructions.push(Instruction::LABEL(continue_label.clone()));
            let result = gen_expression(&loop_data.condition, instructions);
            instructions.push(Instruction::JUMPCOND(InstructionJump {
                jump_type: JumpType::JUMPIFZERO,
                condition: result,
                target: break_label.clone(),
            }));
            gen_instructions(&loop_data.body, instructions);
            instructions.push(Instruction::JUMP(continue_label));
            instructions.push(Instruction::LABEL(break_label));
        }
        parser::Statement::FOR(init, loop_data, post_loop) => {
            match init {
                parser::ForInit::INITDECL(decl) => {
                    gen_declaration(decl, instructions);
                }
                parser::ForInit::INITEXP(Some(expr)) => {
                    gen_expression(expr, instructions);
                }
                _ => (),
            };
            let break_label = format!("break_{}", loop_data.label);
            let start_label = format!("start_{}", loop_data.label);
            instructions.push(Instruction::LABEL(start_label.clone()));
            let condition = gen_expression(&loop_data.condition, instructions);
            instructions.push(Instruction::JUMPCOND(InstructionJump {
                jump_type: JumpType::JUMPIFZERO,
                condition,
                target: break_label.clone(),
            }));
            gen_instructions(&loop_data.body, instructions);
            instructions.push(Instruction::LABEL(format!("continue_{}", loop_data.label)));
            if let Some(post) = post_loop {
                gen_expression(post, instructions);
            }
            instructions.push(Instruction::JUMP(start_label));
            instructions.push(Instruction::LABEL(break_label));
        }
    }
}
fn gen_expression(expression: &parser::Expression, instructions: &mut Vec<Instruction>) -> Value {
    match expression {
        parser::Expression::LITEXP(parser::Literal::INT(val)) => Value::CONSTANT(*val),
        parser::Expression::UNARY(operator, expr) => {
            let src = gen_expression(expr, instructions);
            let dst = Value::VARIABLE(gen_temp_name());
            instructions.push(Instruction::UNARYOP(InstructionUnary {
                operator: operator.clone(),
                src,
                dst: dst.clone(),
            }));
            dst
        }
        parser::Expression::BINARY(operator) => {
            // Short circuiting needs special handling
            if let parser::BinaryOperator::LOGICALAND | parser::BinaryOperator::LOGICALOR =
                operator.operator
            {
                return gen_short_circuit(
                    instructions,
                    &operator.operator,
                    &operator.left,
                    &operator.right,
                );
            };
            let src1 = gen_expression(&operator.left, instructions);
            let src2 = gen_expression(&operator.right, instructions);
            let dst = Value::VARIABLE(gen_temp_name());
            instructions.push(Instruction::BINARYOP(InstructionBinary {
                operator: operator.operator.clone(),
                src1,
                src2,
                dst: dst.clone(),
            }));
            dst
        }
        parser::Expression::VAR(name) => Value::VARIABLE(name.to_string()),
        parser::Expression::ASSIGNMENT(assignment) => {
            let result = gen_expression(&assignment.right, instructions);
            match &assignment.left {
                parser::Expression::VAR(name) => {
                    instructions.push(Instruction::COPY(InstructionCopy {
                        src: result,
                        dst: Value::VARIABLE(name.to_string()),
                    }));
                    Value::VARIABLE(name.to_string())
                }
                _ => panic!("Left hand side of assignment should be variable!"),
            }
        }
        parser::Expression::CONDITION(condition) => {
            let dst = Value::VARIABLE(gen_temp_name());
            let end_label = gen_label("cond_end");
            let e2_label = gen_label("cond_e2");
            let cond = gen_expression(&condition.condition, instructions);
            instructions.push(Instruction::JUMPCOND(InstructionJump {
                jump_type: JumpType::JUMPIFZERO,
                condition: cond,
                target: e2_label.clone(),
            }));
            let e1 = gen_expression(&condition.if_true, instructions);
            instructions.push(Instruction::COPY(InstructionCopy {
                src: e1,
                dst: dst.clone(),
            }));
            instructions.push(Instruction::JUMP(end_label.clone()));
            instructions.push(Instruction::LABEL(e2_label));
            let e2 = gen_expression(&condition.if_false, instructions);
            instructions.push(Instruction::COPY(InstructionCopy {
                src: e2,
                dst: dst.clone(),
            }));
            instructions.push(Instruction::LABEL(end_label));
            dst
        }
        parser::Expression::FUNCTION(name, args) => {
            let results = args.into_iter().map(|arg| gen_expression(arg, instructions)).collect();
            let dst = Value::VARIABLE(gen_temp_name());
            instructions.push(Instruction::FUNCTION(name.to_string(), results, dst.clone()));
            dst
        }
    }
}

fn gen_short_circuit(
    instructions: &mut Vec<Instruction>,
    operator: &parser::BinaryOperator,
    left: &parser::Expression,
    right: &parser::Expression,
) -> Value {
    let (jump_type, label_type) = match operator {
        parser::BinaryOperator::LOGICALAND => (JumpType::JUMPIFZERO, false),
        parser::BinaryOperator::LOGICALOR => (JumpType::JUMPIFNOTZERO, true),
        _ => panic!("Expected a short circuiting operator!"),
    };
    let target = gen_label(&label_type.to_string());
    let end = gen_label("end");
    let dst = Value::VARIABLE(gen_temp_name());
    let v1 = gen_expression(left, instructions);
    instructions.push(Instruction::JUMPCOND(InstructionJump {
        jump_type: jump_type.clone(),
        condition: v1,
        target: target.clone(),
    }));
    let v2 = gen_expression(right, instructions);
    instructions.push(Instruction::JUMPCOND(InstructionJump {
        jump_type,
        condition: v2,
        target: target.clone(),
    }));
    instructions.push(Instruction::COPY(InstructionCopy {
        src: Value::CONSTANT(!label_type as u32),
        dst: dst.clone(),
    }));
    instructions.push(Instruction::JUMP(end.clone()));
    instructions.push(Instruction::LABEL(target));
    instructions.push(Instruction::COPY(InstructionCopy {
        src: Value::CONSTANT(label_type as u32),
        dst: dst.clone(),
    }));
    instructions.push(Instruction::LABEL(end));
    dst
}

fn gen_temp_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("tmp.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn gen_label(label_type: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!(
        "label_{}.{}",
        label_type,
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}
