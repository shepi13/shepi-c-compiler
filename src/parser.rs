use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use lazy_static::lazy_static;

use crate::lexer::TokenType;

pub type Program<'a> = Vec<FunctionDeclaration<'a>>;
pub type Block<'a> = Vec<BlockItem<'a>>;
// Statements and Declarations
#[derive(Debug)]
pub enum BlockItem<'a> {
    STATEMENT(Statement<'a>),
    DECLARATION(Declaration<'a>),
}
#[derive(Debug)]
pub enum Statement<'a> {
    RETURN(Expression),
    EXPRESSION(Expression),
    IF(Expression, Box<Statement<'a>>, Box<Option<Statement<'a>>>),
    WHILE(Loop<'a>),
    DOWHILE(Loop<'a>),
    FOR(ForInit, Loop<'a>, Option<Expression>),
    BREAK(String),
    CONTINUE(String),
    COMPOUND(Block<'a>),
    LABEL(String, Box<Statement<'a>>),
    GOTO(String),
    NULL,
}
#[derive(Debug)]
pub struct Loop<'a> {
    pub label: String,
    pub condition: Expression,
    pub body: Box<Statement<'a>>,
}
#[derive(Debug)]
pub enum ForInit {
    INITDECL(VariableDeclaration),
    INITEXP(Option<Expression>),
}
#[derive(Debug)]
pub enum Declaration<'a> {
    VARIABLE(VariableDeclaration),
    FUNCTION(FunctionDeclaration<'a>),
}
#[derive(Debug)]
pub struct VariableDeclaration {
    pub name: String,
    pub value: Option<Expression>,
}
#[derive(Debug)]
pub struct FunctionDeclaration<'a> {
    pub name: &'a str,
    pub params: Vec<String>,
    pub body: Option<Block<'a>>,
}
// Expressions
#[derive(Debug)]
pub enum Expression {
    LITEXP(Literal),
    VAR(String),
    UNARY(UnaryOperator, Box<Expression>),
    BINARY(Box<BinaryExpression>),
    ASSIGNMENT(Box<AssignmentExpression>),
    CONDITION(Box<ConditionExpression>),
    FUNCTION(String, Vec<Expression>),
}
#[derive(Debug)]
pub struct BinaryExpression {
    pub operator: BinaryOperator,
    pub left: Expression,
    pub right: Expression,
}
#[derive(Debug)]
pub struct AssignmentExpression {
    pub left: Expression,
    pub right: Expression,
}
#[derive(Debug)]

pub struct ConditionExpression {
    pub condition: Expression,
    pub if_true: Expression,
    pub if_false: Expression,
}
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    COMPLEMENT,
    NEGATE,
    LOGICALNOT,
    POSTINCREMENT,
    POSTDECREMENT,
    PREINCREMENT,
    PREDECREMENT,
}
#[derive(Debug, Clone)]
pub enum BinaryOperator {
    // Numeric
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    REMAINDER,
    // Bitwise
    LEFTSHIFT,
    RIGHTSHIFT,
    BITXOR,
    BITOR,
    BITAND,
    // Logical/Relational
    LOGICALAND,
    LOGICALOR,
    ISEQUAL,
    NOTEQUAL,
    LESSTHAN,
    LESSTHANEQUAL,
    GREATERTHAN,
    GREATERTHANEQUAL,
}
#[derive(Debug, Clone)]
pub enum Literal {
    INT(u32),
}

lazy_static! {
    static ref precedence_table: HashMap<TokenType<'static>, usize> = HashMap::from([
        (TokenType::STAR, 50),
        (TokenType::FORWARDSLASH, 50),
        (TokenType::PERCENT, 50),
        (TokenType::PLUS, 45),
        (TokenType::HYPHEN, 45),
        (TokenType::LEFTSHIFT, 40),
        (TokenType::RIGHTSHIFT, 40),
        (TokenType::LESSTHAN, 35),
        (TokenType::LESSTHANEQUAL, 35),
        (TokenType::GREATERTHAN, 35),
        (TokenType::GREATERTHANEQUAL, 35),
        (TokenType::DOUBLEEQUAL, 30),
        (TokenType::NOTEQUAL, 30),
        (TokenType::AMPERSAND, 25),
        (TokenType::CARET, 20),
        (TokenType::PIPE, 15),
        (TokenType::DOUBLEAMPERSAND, 10),
        (TokenType::DOUBLEPIPE, 5),
        (TokenType::QUESTIONMARK, 3),
        (TokenType::EQUAL, 1),
    ]);
}

fn loop_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("loop.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn parse_identifier<'a>(tokens: &mut &[TokenType<'a>]) -> &'a str {
    // Parses an identifier and advances the cursor
    let next_token = &tokens[0];
    *tokens = &tokens[1..];
    match next_token {
        TokenType::IDENTIFIER(name) => name,
        _ => panic!("Expected identifier"),
    }
}
fn parse_binop(tokens: &mut &[TokenType]) -> BinaryOperator {
    // Advance cursor, and map tokens representing binary operations to binary op type
    let next_token = &tokens[0];
    *tokens = &tokens[1..];
    match next_token {
        TokenType::PLUS => BinaryOperator::ADD,
        TokenType::HYPHEN => BinaryOperator::SUBTRACT,
        TokenType::STAR => BinaryOperator::MULTIPLY,
        TokenType::FORWARDSLASH => BinaryOperator::DIVIDE,
        TokenType::PERCENT => BinaryOperator::REMAINDER,
        TokenType::LEFTSHIFT => BinaryOperator::LEFTSHIFT,
        TokenType::RIGHTSHIFT => BinaryOperator::RIGHTSHIFT,
        TokenType::PIPE => BinaryOperator::BITOR,
        TokenType::CARET => BinaryOperator::BITXOR,
        TokenType::AMPERSAND => BinaryOperator::BITAND,
        TokenType::LESSTHAN => BinaryOperator::LESSTHAN,
        TokenType::LESSTHANEQUAL => BinaryOperator::LESSTHANEQUAL,
        TokenType::GREATERTHAN => BinaryOperator::GREATERTHAN,
        TokenType::GREATERTHANEQUAL => BinaryOperator::GREATERTHANEQUAL,
        TokenType::DOUBLEEQUAL => BinaryOperator::ISEQUAL,
        TokenType::NOTEQUAL => BinaryOperator::NOTEQUAL,
        TokenType::DOUBLEAMPERSAND => BinaryOperator::LOGICALAND,
        TokenType::DOUBLEPIPE => BinaryOperator::LOGICALOR,
        _ => panic!("Expected binary operator"),
    }
}
fn parse_argument_list(tokens: &mut &[TokenType]) -> Vec<Expression> {
    let mut args: Vec<Expression> = Vec::new();
    let mut comma = false;
    while tokens[0] != TokenType::CLOSEPAREN {
        args.push(parse_expression(tokens, 0));
        comma = try_consume(tokens, TokenType::COMMA);
    }
    assert!(!comma, "Trailing comma not allowed in C arg list");
    args
}
fn parse_post_operator(tokens: &mut &[TokenType], expression: Expression) -> Expression {
    if try_consume(tokens, TokenType::INCREMENT) {
        parse_post_operator(
            tokens,
            Expression::UNARY(UnaryOperator::POSTINCREMENT, expression.into()),
        )
    } else if try_consume(tokens, TokenType::DECREMENT) {
        parse_post_operator(
            tokens,
            Expression::UNARY(UnaryOperator::POSTDECREMENT, expression.into()),
        )
    } else {
        expression
    }
}
fn parse_factor(tokens: &mut &[TokenType]) -> Expression {
    // Parses a factor (unary value/operator) of a larger expression
    let token = &tokens[0];
    *tokens = &tokens[1..];
    match token {
        TokenType::CONSTANT(val) => Expression::LITEXP(Literal::INT(
            val.parse().expect("Failed to convert constant to int"),
        )),
        TokenType::HYPHEN => Expression::UNARY(UnaryOperator::NEGATE, parse_factor(tokens).into()),
        TokenType::TILDE => {
            Expression::UNARY(UnaryOperator::COMPLEMENT, parse_factor(tokens).into())
        }
        TokenType::EXCLAM => {
            Expression::UNARY(UnaryOperator::LOGICALNOT, parse_factor(tokens).into())
        }
        TokenType::INCREMENT => {
            Expression::UNARY(UnaryOperator::PREINCREMENT, parse_factor(tokens).into())
        }
        TokenType::DECREMENT => {
            Expression::UNARY(UnaryOperator::PREDECREMENT, parse_factor(tokens).into())
        }
        TokenType::OPENPAREN => {
            let expr = parse_expression(tokens, 0);
            expect(tokens, TokenType::CLOSEPAREN);
            parse_post_operator(tokens, expr)
        }
        TokenType::IDENTIFIER(name) => {
            if try_consume(tokens, TokenType::OPENPAREN) {
                let args = parse_argument_list(tokens);
                expect(tokens, TokenType::CLOSEPAREN);
                parse_post_operator(tokens, Expression::FUNCTION(name.to_string(), args))
            } else {
                parse_post_operator(tokens, Expression::VAR(name.to_string()))
            }
        }
        _ => panic!("Expected factor, found {}", token.to_string()),
    }
}
fn parse_expression(tokens: &mut &[TokenType], min_prec: usize) -> Expression {
    // Parses an expression using precedence climbing
    let mut left = parse_factor(tokens);
    while precedence_table.contains_key(&tokens[0]) && precedence_table[&tokens[0]] >= min_prec {
        let token_prec = precedence_table[&tokens[0]];
        if try_consume(tokens, TokenType::EQUAL) {
            let right = parse_expression(tokens, token_prec);
            left = Expression::ASSIGNMENT(AssignmentExpression { left, right }.into());
        } else if try_consume(tokens, TokenType::QUESTIONMARK) {
            let if_true = parse_expression(tokens, 0);
            expect(tokens, TokenType::COLON);
            let if_false = parse_expression(tokens, token_prec);
            left = Expression::CONDITION(
                ConditionExpression {
                    condition: left,
                    if_true,
                    if_false,
                }
                .into(),
            );
        } else {
            let operator = parse_binop(tokens);
            let right = parse_expression(tokens, token_prec + 1);
            left = Expression::BINARY(
                BinaryExpression {
                    operator,
                    left,
                    right,
                }
                .into(),
            );
        }
    }
    left
}
fn parse_optional_expression(tokens: &mut &[TokenType]) -> Option<Expression> {
    match tokens[0] {
        TokenType::CONSTANT(_)
        | TokenType::HYPHEN
        | TokenType::TILDE
        | TokenType::EXCLAM
        | TokenType::OPENPAREN
        | TokenType::IDENTIFIER(_) => Some(parse_expression(tokens, 0)),
        _ => None,
    }
}
fn parse_statement<'a>(tokens: &mut &[TokenType<'a>]) -> Statement<'a> {
    // Parses a statement
    if try_consume(tokens, TokenType::KEYWORD("return")) {
        let return_value = parse_expression(tokens, 0);
        expect(tokens, TokenType::SEMICOLON);
        Statement::RETURN(return_value)
    } else if try_consume(tokens, TokenType::SEMICOLON) {
        Statement::NULL
    } else if try_consume(tokens, TokenType::KEYWORD("if")) {
        expect(tokens, TokenType::OPENPAREN);
        let cond = parse_expression(tokens, 0);
        expect(tokens, TokenType::CLOSEPAREN);
        let then = parse_statement(tokens);
        let otherwise = if try_consume(tokens, TokenType::KEYWORD("else")) {
            Some(parse_statement(tokens))
        } else {
            None
        };
        Statement::IF(cond, then.into(), otherwise.into())
    } else if matches!(tokens[0], TokenType::IDENTIFIER(_)) && tokens[1] == TokenType::COLON {
        if let TokenType::IDENTIFIER(name) = tokens[0] {
            *tokens = &tokens[2..];
            Statement::LABEL(name.to_string(), parse_statement(tokens).into())
        } else {
            panic!("Token 0 must be identifier");
        }
    } else if try_consume(tokens, TokenType::KEYWORD("goto")) {
        let target = parse_identifier(tokens);
        expect(tokens, TokenType::SEMICOLON);
        Statement::GOTO(target.to_string())
    } else if try_consume(tokens, TokenType::OPENBRACE) {
        Statement::COMPOUND(parse_block(tokens))
    } else if try_consume(tokens, TokenType::KEYWORD("break")) {
        expect(tokens, TokenType::SEMICOLON);
        Statement::BREAK(String::new())
    } else if try_consume(tokens, TokenType::KEYWORD("continue")) {
        expect(tokens, TokenType::SEMICOLON);
        Statement::CONTINUE(String::new())
    } else if try_consume(tokens, TokenType::KEYWORD("while")) {
        expect(tokens, TokenType::OPENPAREN);
        let condition = parse_expression(tokens, 0);
        expect(tokens, TokenType::CLOSEPAREN);
        Statement::WHILE(Loop {
            label: loop_name(),
            condition,
            body: parse_statement(tokens).into(),
        })
    } else if try_consume(tokens, TokenType::KEYWORD("do")) {
        let body = parse_statement(tokens);
        expect(tokens, TokenType::KEYWORD("while"));
        expect(tokens, TokenType::OPENPAREN);
        let condition = parse_expression(tokens, 0);
        expect(tokens, TokenType::CLOSEPAREN);
        expect(tokens, TokenType::SEMICOLON);
        Statement::DOWHILE(Loop {
            label: loop_name(),
            condition,
            body: body.into(),
        })
    } else if try_consume(tokens, TokenType::KEYWORD("for")) {
        expect(tokens, TokenType::OPENPAREN);
        let init = if tokens[0] == TokenType::KEYWORD("int") {
            ForInit::INITDECL(parse_variable_declaration(tokens))
        } else {
            let init = ForInit::INITEXP(parse_optional_expression(tokens));
            expect(tokens, TokenType::SEMICOLON);
            init
        };
        let condition = match parse_optional_expression(tokens) {
            Some(expr) => expr,
            None => Expression::LITEXP(Literal::INT(1)),
        };
        expect(tokens, TokenType::SEMICOLON);
        let post_loop = parse_optional_expression(tokens);
        expect(tokens, TokenType::CLOSEPAREN);
        Statement::FOR(
            init,
            Loop {
                label: loop_name(),
                condition,
                body: parse_statement(tokens).into(),
            },
            post_loop,
        )
    } else {
        let expr = parse_expression(tokens, 0);
        expect(tokens, TokenType::SEMICOLON);
        Statement::EXPRESSION(expr)
    }
}
fn parse_variable_declaration<'a>(tokens: &mut &[TokenType<'a>]) -> VariableDeclaration {
    expect(tokens, TokenType::KEYWORD("int"));
    let name = parse_identifier(tokens);
    let value = match tokens[0] {
        TokenType::EQUAL => {
            *tokens = &tokens[1..];
            Some(parse_expression(tokens, 0))
        }
        _ => None,
    };
    expect(tokens, TokenType::SEMICOLON);
    VariableDeclaration {
        name: name.to_string(),
        value,
    }
}
fn parse_block_item<'a>(tokens: &mut &[TokenType<'a>]) -> BlockItem<'a> {
    if tokens[0] == TokenType::KEYWORD("int") {
        let decl = match tokens[2] {
            TokenType::OPENPAREN => Declaration::FUNCTION(parse_function_declaration(tokens)),
            _ => Declaration::VARIABLE(parse_variable_declaration(tokens)),
        };
        BlockItem::DECLARATION(decl)
    } else {
        BlockItem::STATEMENT(parse_statement(tokens))
    }
}
fn parse_block<'a>(tokens: &mut &[TokenType<'a>]) -> Block<'a> {
    let mut body: Block = Vec::new();
    while tokens[0] != TokenType::CLOSEBRACE {
        body.push(parse_block_item(tokens));
    }
    expect(tokens, TokenType::CLOSEBRACE);
    body
}
fn parse_param_list<'a>(tokens: &mut &[TokenType<'a>]) -> Vec<String> {
    if try_consume(tokens, TokenType::KEYWORD("void")) {
        return Vec::new();
    }
    let mut params: Vec<String> = Vec::new();
    loop {
        expect(tokens, TokenType::KEYWORD("int"));
        params.push(parse_identifier(tokens).to_string());
        if !try_consume(tokens, TokenType::COMMA) {
            break;
        }
    }
    params
}
fn parse_function_declaration<'a>(tokens: &mut &[TokenType<'a>]) -> FunctionDeclaration<'a> {
    // Parses a function
    expect(tokens, TokenType::KEYWORD("int"));
    let name = parse_identifier(tokens);
    expect(tokens, TokenType::OPENPAREN);
    let params = parse_param_list(tokens);
    expect(tokens, TokenType::CLOSEPAREN);
    if try_consume(tokens, TokenType::SEMICOLON) {
        FunctionDeclaration {
            name,
            params,
            body: None,
        }
    } else if try_consume(tokens, TokenType::OPENBRACE) {
        FunctionDeclaration {
            name,
            params,
            body: Some(parse_block(tokens)),
        }
    } else {
        panic!("Function declaration must be followed by definition or semicolon");
    }
}
pub fn parse<'a>(tokens: &mut &[TokenType<'a>]) -> Program<'a> {
    // Parses entire program
    let mut program: Program = Vec::new();
    while !tokens.is_empty() {
        program.push(parse_function_declaration(tokens));
    }
    program
}
fn expect(tokens: &mut &[TokenType], token: TokenType) {
    if token != tokens[0] {
        panic!("Syntax Error: Expected {}", token.to_string())
    }
    *tokens = &tokens[1..];
}

fn try_consume(tokens: &mut &[TokenType], token: TokenType) -> bool {
    let is_match = tokens[0] == token;
    if is_match {
        *tokens = &tokens[1..];
    }
    is_match
}
