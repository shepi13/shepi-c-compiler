use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use lazy_static::lazy_static;

use crate::lexer::TokenType;

#[derive(Debug)]
pub struct Program<'a> {
    pub main: Function<'a>,
}
#[derive(Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub body: Vec<BlockItem<'a>>,
}

// Statements and Declarations
#[derive(Debug)]
pub enum BlockItem<'a> {
    STATEMENT(Statement<'a>),
    DECLARATION(Declaration),
}
#[derive(Debug)]
pub enum Statement<'a> {
    RETURN(Expression),
    EXPRESSION(Expression),
    IF(Expression, Rc<Statement<'a>>, Rc<Option<Statement<'a>>>),
    LABEL(&'a str),
    GOTO(&'a str),
    NULL,
}
#[derive(Debug)]
pub struct Declaration {
    pub name: String,
    pub value: Option<Expression>,
}

// Expressions
#[derive(Debug)]
pub enum Expression {
    LITEXP(Literal),
    VAR(String),
    UNARY(Rc<UnaryExpression>),
    BINARY(Rc<BinaryExpression>),
    ASSIGNMENT(Rc<AssignmentExpression>),
    CONDITION(Rc<ConditionExpression>),
}
#[derive(Debug)]
pub enum UnaryExpression {
    COMPLEMENT(Expression),
    NEGATE(Expression),
    LOGICALNOT(Expression),
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

// PARSER

pub struct SymbolTable<'a> {
    variables: Vec<HashMap<&'a str, String>>,
    labels: Vec<HashSet<&'a str>>,
    gotos: Vec<HashSet<&'a str>>,
}

impl<'a> SymbolTable<'a> {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            labels: Vec::new(),
            gotos: Vec::new(),
        }
    }
    fn enter_scope(&mut self) {
        self.variables.push(HashMap::new());
        self.labels.push(HashSet::new());
        self.gotos.push(HashSet::new());
    }
    fn leave_scope(&mut self) {
        self.variables.pop();
        let labels = self.labels.pop().expect("Scope missing from symbol table");
        let gotos = self.gotos.pop().expect("Scope missing from symbol table");

        let mut set_diff = gotos.difference(&labels);
        match set_diff.next() {
            Some(label) => panic!("Tried to goto {}, but label doesn't exist", label),
            _ => (),
        };
    }
    fn declare_variable(&mut self, name: &'a str) -> String {
        let stacklen = self.variables.len() - 1;
        let table = &mut self.variables[stacklen];
        if table.contains_key(name) {
            panic!("Duplicate variable name in current scope: {}", name);
        }
        let unique_name = SymbolTable::gen_variable_name(name);
        table.insert(name, unique_name.clone());
        unique_name
    }
    fn resolve_variable(&self, name: &str) -> String {
        for table in self.variables.iter().rev() {
            if table.contains_key(name) {
                return table[name].clone();
            }
        }
        panic!("Undeclared variable: {}", name);
    }
    fn declare_label(&mut self, target: &'a str) {
        for table in &self.labels {
            if table.contains(target) {
                panic!("Duplicate label: {}", target);
            }
        }
        let stacklen = &self.labels.len() - 1;
        self.labels[stacklen].insert(target);
    }
    fn declare_goto(&mut self, target: &'a str) {
        let stacklen = &self.gotos.len() - 1;
        self.gotos[stacklen].insert(target);
    }
    
    fn gen_variable_name(name: &str) -> String {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        format!("{}.{}", name, COUNTER.fetch_add(1, Ordering::Relaxed))
    }
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
fn parse_factor(tokens: &mut &[TokenType], symbol_table: &mut SymbolTable) -> Expression {
    // Parses a factor (unary value/operator) of a larger expression
    let token = &tokens[0];
    *tokens = &tokens[1..];
    let result = match token {
        TokenType::CONSTANT(val) => Expression::LITEXP(Literal::INT(
            val.parse().expect("Failed to convert constant to int"),
        )),
        TokenType::HYPHEN => {
            Expression::UNARY(UnaryExpression::NEGATE(parse_factor(tokens, symbol_table)).into())
        }
        TokenType::TILDE => Expression::UNARY(
            UnaryExpression::COMPLEMENT(parse_factor(tokens, symbol_table)).into(),
        ),
        TokenType::EXCLAM => Expression::UNARY(
            UnaryExpression::LOGICALNOT(parse_factor(tokens, symbol_table)).into(),
        ),
        TokenType::OPENPAREN => {
            let expr = parse_expression(tokens, 0, symbol_table);
            expect(tokens, TokenType::CLOSEPAREN);
            expr
        }
        TokenType::IDENTIFIER(name) => Expression::VAR(symbol_table.resolve_variable(name)),
        _ => panic!("Expected factor, found {}", token.to_string()),
    };
    result
}
fn parse_expression(
    tokens: &mut &[TokenType],
    min_prec: usize,
    symbol_table: &mut SymbolTable,
) -> Expression {
    // Parses an expression using precedence climbing
    let mut left = parse_factor(tokens, symbol_table);
    while precedence_table.contains_key(&tokens[0]) && precedence_table[&tokens[0]] >= min_prec {
        let token_prec = precedence_table[&tokens[0]];
        if try_consume(tokens, TokenType::EQUAL) {
            match left {
                Expression::VAR(_) => {}
                _ => panic!("Left hand side of assignment must be a variable!"),
            }
            let right = parse_expression(tokens, token_prec, symbol_table);
            left = Expression::ASSIGNMENT(AssignmentExpression { left, right }.into());
        } else if try_consume(tokens, TokenType::QUESTIONMARK) {
            let if_true = parse_expression(tokens, 0, symbol_table);
            expect(tokens, TokenType::COLON);
            let if_false = parse_expression(tokens, token_prec, symbol_table);
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
            let right = parse_expression(tokens, token_prec + 1, symbol_table);
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
fn parse_statement<'a>(
    tokens: &mut &[TokenType<'a>],
    symbol_table: &mut SymbolTable<'a>,
) -> Statement<'a> {
    // Parses a statement
    if try_consume(tokens, TokenType::KEYWORD("return")) {
        let return_value = parse_expression(tokens, 0, symbol_table);
        expect(tokens, TokenType::SEMICOLON);
        Statement::RETURN(return_value)
    } else if try_consume(tokens, TokenType::SEMICOLON) {
        Statement::NULL
    } else if try_consume(tokens, TokenType::KEYWORD("if")) {
        expect(tokens, TokenType::OPENPAREN);
        let cond = parse_expression(tokens, 0, symbol_table);
        expect(tokens, TokenType::CLOSEPAREN);
        let then = parse_statement(tokens, symbol_table);
        let otherwise = if try_consume(tokens, TokenType::KEYWORD("else")) {
            Some(parse_statement(tokens, symbol_table))
        } else {
            None
        };
        Statement::IF(cond, then.into(), otherwise.into())
    } else if matches!(tokens[0], TokenType::IDENTIFIER(_)) && tokens[1] == TokenType::COLON {
        if let TokenType::IDENTIFIER(name) = tokens[0] {
            *tokens = &tokens[2..];
            symbol_table.declare_label(name);
            Statement::LABEL(name)
        } else {
            panic!("Token 0 must be identifier");
        }
    } else if try_consume(tokens, TokenType::KEYWORD("goto")) {
        let target = parse_identifier(tokens);
        expect(tokens, TokenType::SEMICOLON);
        symbol_table.declare_goto(target);
        Statement::GOTO(target)
    } else {
        let expr = parse_expression(tokens, 0, symbol_table);
        expect(tokens, TokenType::SEMICOLON);
        Statement::EXPRESSION(expr)
    }
}
fn parse_declaration<'a>(
    tokens: &mut &[TokenType<'a>],
    symbol_table: &mut SymbolTable<'a>,
) -> Declaration {
    let name = parse_identifier(tokens);
    let name = symbol_table.declare_variable(name);
    let value = match tokens[0] {
        TokenType::EQUAL => {
            *tokens = &tokens[1..];
            Some(parse_expression(tokens, 0, symbol_table))
        }
        _ => None,
    };
    expect(tokens, TokenType::SEMICOLON);
    Declaration { name, value }
}
fn parse_block_item<'a>(
    tokens: &mut &[TokenType<'a>],
    symbol_table: &mut SymbolTable<'a>,
) -> BlockItem<'a> {
    if try_consume(tokens, TokenType::KEYWORD("int")) {
        BlockItem::DECLARATION(parse_declaration(tokens, symbol_table))
    } else {
        BlockItem::STATEMENT(parse_statement(tokens, symbol_table))
    }
}
fn parse_function<'a>(
    tokens: &mut &[TokenType<'a>],
    symbol_table: &mut SymbolTable<'a>,
) -> Function<'a> {
    // Parses a function
    expect(tokens, TokenType::KEYWORD("int"));
    let name = parse_identifier(tokens);
    expect(tokens, TokenType::OPENPAREN);
    expect(tokens, TokenType::KEYWORD("void"));
    expect(tokens, TokenType::CLOSEPAREN);
    expect(tokens, TokenType::OPENBRACE);
    symbol_table.enter_scope();
    let mut body: Vec<BlockItem> = Vec::new();
    while tokens[0] != TokenType::CLOSEBRACE {
        body.push(parse_block_item(tokens, symbol_table));
    }
    symbol_table.leave_scope();
    expect(tokens, TokenType::CLOSEBRACE);
    Function { name, body }
}
pub fn parse<'a>(tokens: &mut &[TokenType<'a>], symbol_table: &mut SymbolTable<'a>) -> Program<'a> {
    // Parses entire program
    let function = parse_function(tokens, symbol_table);
    assert!(function.name == "main", "Failed to find main function!");
    assert!(tokens.is_empty());
    Program { main: function }
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
