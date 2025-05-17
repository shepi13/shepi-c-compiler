// Generates Three Adress Code Intermediate representation (Step between Parser AST and Assembly AST)

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::parser::{self, Expression};

#[derive(Debug)]
pub struct Program<'a> {
    pub main: Function<'a>,
}

#[derive(Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug)]
pub enum Instruction {
    RETURN(Value),
    UNARYOP(InstructionUnary),
    BINARYOP(InstructionBinary),
    COPY(InstructionCopy),
    LABEL(String),
    JUMP(InstructionJump),
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
    JUMP,
    JUMPIFZERO,
    JUMPIFNOTZERO,
}
#[derive(Debug)]
pub enum UnaryOperator {
    NEGATE,
    COMPLEMENT,
    LOGICALNOT,
}

#[derive(Debug, Clone)]
pub enum Value {
    CONSTANT(u32),
    VARIABLE(String),
}

pub fn gen_tac_ast<'a>(parser_ast: &parser::Program<'a>) -> Program<'a> {
    Program {
        main: gen_function(&parser_ast.main),
    }
}
fn gen_function<'a>(function: &parser::Function<'a>) -> Function<'a> {
    let mut instructions: Vec<Instruction> = Vec::new();
    gen_block(&function.body, &mut instructions);
    instructions.push(Instruction::RETURN(Value::CONSTANT(0)));
    Function {
        name: function.name,
        instructions,
    }
}
fn gen_block(block: &parser::Block, instructions: &mut Vec<Instruction>) {
    for block_item in block {
        match block_item {
            parser::BlockItem::STATEMENT(statement) => gen_instructions(&statement, instructions),
            parser::BlockItem::DECLARATION(decl) => {
                if let Some(value) = &decl.value {
                    let result = gen_expression(&value, instructions);
                    instructions.push(Instruction::COPY(InstructionCopy {
                        src: result,
                        dst: Value::VARIABLE(decl.name.to_string()),
                    }));
                }
            }
        }
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
            instructions.push(Instruction::JUMP(InstructionJump {
                jump_type: JumpType::JUMPIFZERO,
                condition,
                target: else_label.clone(),
            }));
            gen_instructions(&if_true, instructions);
            instructions.push(Instruction::JUMP(InstructionJump {
                jump_type: JumpType::JUMP,
                condition: Value::CONSTANT(1),
                target: end_label.clone(),
            }));
            instructions.push(Instruction::LABEL(else_label));
            if let Some(false_statement) = if_false.as_ref() {
                gen_instructions(&false_statement, instructions);
            }
            instructions.push(Instruction::LABEL(end_label));
        }
        parser::Statement::GOTO(target) => instructions.push(Instruction::JUMP(InstructionJump {
            jump_type: JumpType::JUMP,
            condition: Value::CONSTANT(1),
            target: target.to_string(),
        })),
        parser::Statement::LABEL(name) => {
            instructions.push(Instruction::LABEL(name.to_string()));
        }
        parser::Statement::COMPOUND(block) => gen_block(block, instructions),
        parser::Statement::BREAK(_)
        | parser::Statement::CONTINUE(_)
        | parser::Statement::WHILE(_)
        | parser::Statement::DOWHILE(_)
        | parser::Statement::FOR(_, _, _) => panic!("Not implemented!"),
    }
}
fn gen_expression(expression: &parser::Expression, instructions: &mut Vec<Instruction>) -> Value {
    match expression {
        parser::Expression::LITEXP(parser::Literal::INT(val)) => Value::CONSTANT(*val),
        parser::Expression::UNARY(operator) => {
            let src = match operator.as_ref() {
                parser::UnaryExpression::COMPLEMENT(expr)
                | parser::UnaryExpression::NEGATE(expr)
                | parser::UnaryExpression::LOGICALNOT(expr) => gen_expression(expr, instructions),
            };
            let dst = Value::VARIABLE(gen_temp_name());
            instructions.push(Instruction::UNARYOP(InstructionUnary {
                operator: gen_operator(operator.as_ref()),
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
            instructions.push(Instruction::JUMP(InstructionJump {
                jump_type: JumpType::JUMPIFZERO,
                condition: cond,
                target: e2_label.clone(),
            }));
            let e1 = gen_expression(&condition.if_true, instructions);
            instructions.push(Instruction::COPY(InstructionCopy {
                src: e1,
                dst: dst.clone(),
            }));
            instructions.push(Instruction::JUMP(InstructionJump {
                jump_type: JumpType::JUMP,
                condition: Value::CONSTANT(1),
                target: end_label.clone(),
            }));
            instructions.push(Instruction::LABEL(e2_label));
            let e2 = gen_expression(&condition.if_false, instructions);
            instructions.push(Instruction::COPY(InstructionCopy {
                src: e2,
                dst: dst.clone(),
            }));
            instructions.push(Instruction::LABEL(end_label));
            dst
        }
    }
}

fn gen_short_circuit(
    instructions: &mut Vec<Instruction>,
    operator: &parser::BinaryOperator,
    left: &Expression,
    right: &Expression,
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
    instructions.push(Instruction::JUMP(InstructionJump {
        jump_type: jump_type.clone(),
        condition: v1,
        target: target.clone(),
    }));
    let v2 = gen_expression(right, instructions);
    instructions.push(Instruction::JUMP(InstructionJump {
        jump_type,
        condition: v2,
        target: target.clone(),
    }));
    instructions.push(Instruction::COPY(InstructionCopy {
        src: Value::CONSTANT(!label_type as u32),
        dst: dst.clone(),
    }));
    instructions.push(Instruction::JUMP(InstructionJump {
        jump_type: JumpType::JUMP,
        condition: Value::CONSTANT(1),
        target: end.clone(),
    }));
    instructions.push(Instruction::LABEL(target));
    instructions.push(Instruction::COPY(InstructionCopy {
        src: Value::CONSTANT(label_type as u32),
        dst: dst.clone(),
    }));
    instructions.push(Instruction::LABEL(end));
    dst
}

fn gen_operator(operator: &parser::UnaryExpression) -> UnaryOperator {
    match operator {
        parser::UnaryExpression::COMPLEMENT(_) => UnaryOperator::COMPLEMENT,
        parser::UnaryExpression::NEGATE(_) => UnaryOperator::NEGATE,
        parser::UnaryExpression::LOGICALNOT(_) => UnaryOperator::LOGICALNOT,
    }
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
