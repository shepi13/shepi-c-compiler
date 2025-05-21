use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use lazy_static::lazy_static;

use crate::lexer::TokenType;

pub type Program = Vec<Declaration>;
pub type Block = Vec<BlockItem>;
// Statements and Declarations
#[derive(Debug)]
pub enum BlockItem {
    StatementItem(Statement),
    DeclareItem(Declaration),
}
#[derive(Debug)]
pub enum Statement {
    Return(TypedExpression),
    ExprStmt(TypedExpression),
    If(TypedExpression, Box<Statement>, Box<Option<Statement>>),
    While(Loop),
    DoWhile(Loop),
    For(ForInit, Loop, Option<TypedExpression>),
    Switch(SwitchStatement),
    Case(TypedExpression, Box<Statement>),
    Default(Box<Statement>),
    Break(String),
    Continue(String),
    Compound(Block),
    Label(String, Box<Statement>),
    Goto(String),
    Null,
}
#[derive(Debug)]
pub struct SwitchStatement {
    pub label: String,
    pub condition: TypedExpression,
    pub cases: Vec<(String, TypedExpression)>,
    pub statement: Box<Statement>,
    pub default: Option<String>,
}
#[derive(Debug)]
pub struct Loop {
    pub label: String,
    pub condition: TypedExpression,
    pub body: Box<Statement>,
}
#[derive(Debug)]
pub enum ForInit {
    Decl(VariableDeclaration),
    Expr(Option<TypedExpression>),
}
#[derive(Debug)]
pub enum Declaration {
    Variable(VariableDeclaration),
    Function(FunctionDeclaration),
}
#[derive(Debug, PartialEq, Clone)]
pub enum CType {
    Int,
    Long,
    Function(Vec<CType>, Box<CType>),
}
#[derive(Debug)]
pub struct VariableDeclaration {
    pub name: String,
    pub value: Option<TypedExpression>,
    pub ctype: CType,
    pub storage: Option<StorageClass>,
}
#[derive(Debug)]
pub struct FunctionDeclaration {
    pub name: String,
    pub ctype: CType,
    pub params: Vec<String>,
    pub body: Option<Block>,
    pub storage: Option<StorageClass>,
}
#[derive(Debug, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
}
impl StorageClass {
    fn from(token: &TokenType) -> Self {
        match token {
            TokenType::Specifier("static") => Self::Static,
            TokenType::Specifier("extern") => Self::Extern,
            _ => panic!("Invalid storage class!"),
        }
    }
}
// Expressions
#[derive(Debug)]
pub struct TypedExpression {
    pub ctype: Option<CType>,
    pub expr: Expression,
}
impl From<Expression> for TypedExpression {
    fn from(value: Expression) -> Self {
        Self {ctype: None, expr: value}
    }
}
impl From<Expression> for Box<TypedExpression> {
    fn from(value: Expression) -> Self {
        TypedExpression { ctype: None, expr: value }.into()
    }
}
#[derive(Debug)]
pub enum Expression {
    Constant(Constant),
    Variable(String),
    Unary(UnaryOperator, Box<TypedExpression>),
    Binary(Box<BinaryExpression>),
    Assignment(Box<AssignmentExpression>),
    Condition(Box<ConditionExpression>),
    FunctionCall(String, Vec<TypedExpression>),
    Cast(CType, Box<TypedExpression>),
}
#[derive(Debug)]
pub struct BinaryExpression {
    pub operator: BinaryOperator,
    pub left: TypedExpression,
    pub right: TypedExpression,
    pub is_assignment: bool,
}
#[derive(Debug)]
pub struct AssignmentExpression {
    pub left: TypedExpression,
    pub right: TypedExpression,
}
#[derive(Debug)]

pub struct ConditionExpression {
    pub condition: TypedExpression,
    pub if_true: TypedExpression,
    pub if_false: TypedExpression,
}
#[derive(Debug)]
pub enum UnaryOperator {
    Complement,
    Negate,
    LogicalNot,
    Increment(Increment),
}
#[derive(Debug)]
pub enum Increment {
    PreIncrement,
    PostIncrement,
    PreDecrement,
    PostDecrement,
}
#[derive(Debug, Clone)]
pub enum BinaryOperator {
    // Numeric
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    // Bitwise
    LeftShift,
    RightShift,
    BitXor,
    BitOr,
    BitAnd,
    // Logical/Relational
    LogicalAnd,
    LogicalOr,
    IsEqual,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
}
#[derive(Debug, Clone)]
pub enum Constant {
    Int(i32),
    Long(i64),
}

lazy_static! {
    static ref precedence_table: HashMap<TokenType<'static>, usize> = HashMap::from([
        (TokenType::Star, 50),
        (TokenType::ForwardSlash, 50),
        (TokenType::Percent, 50),
        (TokenType::Plus, 45),
        (TokenType::Hyphen, 45),
        (TokenType::LeftShift, 40),
        (TokenType::RightShift, 40),
        (TokenType::LessThan, 35),
        (TokenType::LessThanEqual, 35),
        (TokenType::GreaterThan, 35),
        (TokenType::GreaterThanEqual, 35),
        (TokenType::DoubleEqual, 30),
        (TokenType::ExclamEqual, 30),
        (TokenType::Ampersand, 25),
        (TokenType::Caret, 20),
        (TokenType::Pipe, 15),
        (TokenType::DoubleAmpersand, 10),
        (TokenType::DoublePipe, 5),
        (TokenType::QuestionMark, 3),
        (TokenType::Equal, 1),
        (TokenType::PlusEqual, 1),
        (TokenType::HyphenEqual, 1),
        (TokenType::StarEqual, 1),
        (TokenType::ForwardSlashEqual, 1),
        (TokenType::PercentEqual, 1),
        (TokenType::AmpersandEqual, 1),
        (TokenType::PipeEqual, 1),
        (TokenType::CaretEqual, 1),
        (TokenType::LeftShiftEqual, 1),
        (TokenType::RightShiftEqual, 1),
    ]);
}

fn try_consume(tokens: &mut &[TokenType], token: TokenType) -> bool {
    let is_match = tokens[0] == token;
    if is_match {
        *tokens = &tokens[1..];
    }
    is_match
}
fn expect(tokens: &mut &[TokenType], token: TokenType) {
    if token != tokens[0] {
        panic!(
            "Syntax Error: Expected `{:?}`, found `{:?}`",
            token, tokens[0]
        )
    }
    *tokens = &tokens[1..];
}

fn loop_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("loop.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}
fn switch_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("switch.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn is_type_specifier(token: &TokenType) -> bool {
    matches!(
        *token,
        TokenType::Specifier("int") | TokenType::Specifier("long")
    )
}
fn is_assignment_token(token: &TokenType) -> bool {
    use TokenType::*;
    matches!(
        token,
        PlusEqual
            | HyphenEqual
            | StarEqual
            | ForwardSlashEqual
            | PercentEqual
            | AmpersandEqual
            | PipeEqual
            | CaretEqual
            | LeftShiftEqual
            | RightShiftEqual
    )
}

fn parse_type(tokens: &mut &[TokenType]) -> CType {
    use TokenType::Specifier;
    let index = tokens.iter().position(|token| !is_type_specifier(token));
    let index = index.unwrap_or(tokens.len());
    let type_tokens = &tokens[..index];
    *tokens = &tokens[index..];

    let count_token = |token| type_tokens.iter().filter(|elem| **elem == token).count();
    let int_count = count_token(Specifier("int"));
    let long_count = count_token(Specifier("long"));
    if int_count == 1 && long_count == 0 {
        CType::Int
    } else if (int_count == 0 || int_count == 1) && long_count == 1 {
        CType::Long
    } else if (int_count == 0 || int_count == 1) && long_count == 2 {
        CType::Long // Long Long but we represent it the same as they are both 8 bits
    } else {
        panic!("Invalid type!")
    }
}

fn parse_specifiers(tokens: &mut &[TokenType]) -> (CType, Option<StorageClass>) {
    let mut types = Vec::new();
    let mut storage_classes = Vec::new();
    while matches!(tokens[0], TokenType::Specifier(_)) {
        if is_type_specifier(&tokens[0]) {
            types.push(tokens[0].clone());
        } else {
            storage_classes.push(StorageClass::from(&tokens[0]));
        }
        *tokens = &tokens[1..];
    }
    if storage_classes.len() > 1 {
        panic!("Invalid storage class!")
    }
    let c_type = parse_type(&mut &types[..]);
    let storage_class = storage_classes.pop();
    (c_type, storage_class)
}

fn parse_identifier<'a>(tokens: &mut &[TokenType<'a>]) -> &'a str {
    // Parses an identifier and advances the cursor
    let next_token = &tokens[0];
    *tokens = &tokens[1..];
    match next_token {
        TokenType::Identifier(name) => name,
        _ => panic!("Expected identifier"),
    }
}
fn parse_binop(tokens: &mut &[TokenType]) -> BinaryOperator {
    // Advance cursor, and map tokens representing binary operations to binary op type
    use TokenType::*;
    let next_token = &tokens[0];
    *tokens = &tokens[1..];
    match next_token {
        // Math or Assignment
        Plus | PlusEqual => BinaryOperator::Add,
        Hyphen | HyphenEqual => BinaryOperator::Subtract,
        Star | StarEqual => BinaryOperator::Multiply,
        ForwardSlash | ForwardSlashEqual => BinaryOperator::Divide,
        Percent | PercentEqual => BinaryOperator::Remainder,
        LeftShift | LeftShiftEqual => BinaryOperator::LeftShift,
        RightShift | RightShiftEqual => BinaryOperator::RightShift,
        Pipe | PipeEqual => BinaryOperator::BitOr,
        Caret | CaretEqual => BinaryOperator::BitXor,
        Ampersand | AmpersandEqual => BinaryOperator::BitAnd,
        // Relational
        LessThan => BinaryOperator::LessThan,
        LessThanEqual => BinaryOperator::LessThanEqual,
        GreaterThan => BinaryOperator::GreaterThan,
        GreaterThanEqual => BinaryOperator::GreaterThanEqual,
        DoubleEqual => BinaryOperator::IsEqual,
        ExclamEqual => BinaryOperator::NotEqual,
        DoubleAmpersand => BinaryOperator::LogicalAnd,
        DoublePipe => BinaryOperator::LogicalOr,
        _ => panic!("Expected binary operator"),
    }
}
fn parse_argument_list(tokens: &mut &[TokenType]) -> Vec<TypedExpression> {
    let mut args: Vec<TypedExpression> = Vec::new();
    let mut comma = false;
    while tokens[0] != TokenType::CloseParen {
        args.push(parse_expression(tokens, 0));
        comma = try_consume(tokens, TokenType::Comma);
    }
    assert!(!comma, "Trailing comma not allowed in C arg list");
    args
}
fn parse_post_operator(tokens: &mut &[TokenType], expression: TypedExpression) -> TypedExpression {
    if try_consume(tokens, TokenType::Increment) {
        parse_post_operator(
            tokens,
            Expression::Unary(
                UnaryOperator::Increment(Increment::PostIncrement),
                expression.into(),
            ).into(),
        )
    } else if try_consume(tokens, TokenType::Decrement) {
        parse_post_operator(
            tokens,
            Expression::Unary(
                UnaryOperator::Increment(Increment::PostDecrement),
                expression.into(),
            ).into(),
        )
    } else {
        expression
    }
}
fn parse_factor(tokens: &mut &[TokenType]) -> TypedExpression {
    // Parses a factor (unary value/operator) of a larger expression
    let token = &tokens[0];
    *tokens = &tokens[1..];
    match token {
        TokenType::Constant(val) => match val.parse::<i32>() {
            Ok(val) => Expression::Constant(Constant::Int(val)).into(),
            Err(_) => Expression::Constant(Constant::Long(
                val.parse().expect("Failed to convert constant to int"),
            )).into(),
        },
        TokenType::LongConstant(val) => Expression::Constant(Constant::Long(
            val.parse().expect("Failed to convert constant to long"),
        )).into(),
        TokenType::Hyphen => Expression::Unary(UnaryOperator::Negate, parse_factor(tokens).into()).into(),
        TokenType::Tilde => {
            Expression::Unary(UnaryOperator::Complement, parse_factor(tokens).into()).into()
        }
        TokenType::Exclam => {
            Expression::Unary(UnaryOperator::LogicalNot, parse_factor(tokens).into()).into()
        }
        TokenType::Increment => Expression::Unary(
            UnaryOperator::Increment(Increment::PreIncrement),
            parse_factor(tokens).into(),
        ).into(),
        TokenType::Decrement => Expression::Unary(
            UnaryOperator::Increment(Increment::PreDecrement),
            parse_factor(tokens).into(),
        ).into(),
        TokenType::OpenParen => {
            if is_type_specifier(&tokens[0]) {
                let ctype = parse_type(tokens);
                expect(tokens, TokenType::CloseParen);
                Expression::Cast(ctype, parse_factor(tokens).into()).into()
            } else {
                let expr = parse_expression(tokens, 0);
                expect(tokens, TokenType::CloseParen);
                parse_post_operator(tokens, expr)
            }
        }
        TokenType::Identifier(name) => {
            if try_consume(tokens, TokenType::OpenParen) {
                let args = parse_argument_list(tokens);
                expect(tokens, TokenType::CloseParen);
                parse_post_operator(tokens, Expression::FunctionCall(name.to_string(), args).into())
            } else {
                parse_post_operator(tokens, Expression::Variable(name.to_string()).into())
            }
        }
        _ => panic!("Expected factor, found {:?}", token),
    }
}
fn parse_expression(tokens: &mut &[TokenType], min_prec: usize) -> TypedExpression {
    // Parses an expression using precedence climbing
    let mut left = parse_factor(tokens);
    while precedence_table.contains_key(&tokens[0]) && precedence_table[&tokens[0]] >= min_prec {
        let token_prec = precedence_table[&tokens[0]];
        if try_consume(tokens, TokenType::Equal) {
            let right = parse_expression(tokens, token_prec);
            left = Expression::Assignment(AssignmentExpression { left, right }.into()).into();
        } else if try_consume(tokens, TokenType::QuestionMark) {
            let if_true = parse_expression(tokens, 0);
            expect(tokens, TokenType::Colon);
            let if_false = parse_expression(tokens, token_prec);
            left = Expression::Condition(
                ConditionExpression {
                    condition: left,
                    if_true,
                    if_false,
                }
                .into(),
            ).into();
        } else {
            let is_assignment = is_assignment_token(&tokens[0]);
            let operator = parse_binop(tokens);
            let right = if is_assignment {
                parse_expression(tokens, token_prec)
            } else {
                parse_expression(tokens, token_prec + 1)
            };
            left = Expression::Binary(
                BinaryExpression {
                    operator,
                    left,
                    right,
                    is_assignment,
                }
                .into(),
            ).into();
        }
    }
    left
}
fn parse_optional_expression(tokens: &mut &[TokenType]) -> Option<TypedExpression> {
    match tokens[0] {
        TokenType::Constant(_)
        | TokenType::Hyphen
        | TokenType::Tilde
        | TokenType::Exclam
        | TokenType::OpenParen
        | TokenType::Identifier(_) => Some(parse_expression(tokens, 0)),
        _ => None,
    }
}
fn parse_statement(tokens: &mut &[TokenType]) -> Statement {
    // Parses a statement
    if try_consume(tokens, TokenType::Keyword("return")) {
        let return_value = parse_expression(tokens, 0);
        expect(tokens, TokenType::SemiColon);
        Statement::Return(return_value)
    } else if try_consume(tokens, TokenType::SemiColon) {
        Statement::Null
    } else if try_consume(tokens, TokenType::Keyword("if")) {
        expect(tokens, TokenType::OpenParen);
        let cond = parse_expression(tokens, 0);
        expect(tokens, TokenType::CloseParen);
        let then = parse_statement(tokens);
        let otherwise = if try_consume(tokens, TokenType::Keyword("else")) {
            Some(parse_statement(tokens))
        } else {
            None
        };
        Statement::If(cond, then.into(), otherwise.into())
    } else if try_consume(tokens, TokenType::Keyword("goto")) {
        let target = parse_identifier(tokens);
        expect(tokens, TokenType::SemiColon);
        Statement::Goto(target.to_string())
    } else if try_consume(tokens, TokenType::OpenBrace) {
        Statement::Compound(parse_block(tokens))
    } else if try_consume(tokens, TokenType::Keyword("break")) {
        expect(tokens, TokenType::SemiColon);
        Statement::Break(String::new())
    } else if try_consume(tokens, TokenType::Keyword("continue")) {
        expect(tokens, TokenType::SemiColon);
        Statement::Continue(String::new())
    } else if try_consume(tokens, TokenType::Keyword("while")) {
        expect(tokens, TokenType::OpenParen);
        let condition = parse_expression(tokens, 0);
        expect(tokens, TokenType::CloseParen);
        Statement::While(Loop {
            label: loop_name(),
            condition,
            body: parse_statement(tokens).into(),
        })
    } else if try_consume(tokens, TokenType::Keyword("do")) {
        let body = parse_statement(tokens);
        expect(tokens, TokenType::Keyword("while"));
        expect(tokens, TokenType::OpenParen);
        let condition = parse_expression(tokens, 0);
        expect(tokens, TokenType::CloseParen);
        expect(tokens, TokenType::SemiColon);
        Statement::DoWhile(Loop {
            label: loop_name(),
            condition,
            body: body.into(),
        })
    } else if try_consume(tokens, TokenType::Keyword("for")) {
        expect(tokens, TokenType::OpenParen);
        let init = if matches!(tokens[0], TokenType::Specifier(_)) {
            ForInit::Decl(parse_variable_declaration(tokens))
        } else {
            let init = ForInit::Expr(parse_optional_expression(tokens));
            expect(tokens, TokenType::SemiColon);
            init
        };
        let condition = match parse_optional_expression(tokens) {
            Some(expr) => expr,
            None => Expression::Constant(Constant::Int(1)).into(),
        };
        expect(tokens, TokenType::SemiColon);
        let post_loop = parse_optional_expression(tokens);
        expect(tokens, TokenType::CloseParen);
        Statement::For(
            init,
            Loop {
                label: loop_name(),
                condition,
                body: parse_statement(tokens).into(),
            },
            post_loop,
        )
    } else if try_consume(tokens, TokenType::Keyword("switch")) {
        let label = switch_name();
        expect(tokens, TokenType::OpenParen);
        let condition = parse_expression(tokens, 0);
        expect(tokens, TokenType::CloseParen);
        let cases = Vec::new();
        let statement = parse_statement(tokens).into();
        Statement::Switch(SwitchStatement {
            label,
            condition,
            cases,
            statement,
            default: None,
        })
    } else if try_consume(tokens, TokenType::Keyword("default")) {
        expect(tokens, TokenType::Colon);
        Statement::Default(parse_statement(tokens).into())
    } else if try_consume(tokens, TokenType::Keyword("case")) {
        let matcher = parse_expression(tokens, 0);
        expect(tokens, TokenType::Colon);
        let statement = parse_statement(tokens).into();
        Statement::Case(matcher, statement)
    } else if matches!(tokens[0], TokenType::Identifier(_)) && tokens[1] == TokenType::Colon {
        if let TokenType::Identifier(name) = tokens[0] {
            *tokens = &tokens[2..];
            Statement::Label(name.to_string(), parse_statement(tokens).into())
        } else {
            panic!("Token 0 must be identifier");
        }
    } else {
        let expr = parse_expression(tokens, 0);
        expect(tokens, TokenType::SemiColon);
        Statement::ExprStmt(expr)
    }
}
fn parse_variable_declaration(tokens: &mut &[TokenType]) -> VariableDeclaration {
    let decl = parse_declaration(tokens);
    match decl {
        Declaration::Variable(var_decl) => var_decl,
        _ => panic!("Expected variable declaration"),
    }
}

fn parse_declaration(tokens: &mut &[TokenType]) -> Declaration {
    let (ctype, storage) = parse_specifiers(tokens);
    let name = parse_identifier(tokens);
    if try_consume(tokens, TokenType::OpenParen) {
        let (params, param_types) = parse_param_list(tokens);
        expect(tokens, TokenType::CloseParen);
        let body = if try_consume(tokens, TokenType::SemiColon) {
            None
        } else if try_consume(tokens, TokenType::OpenBrace) {
            Some(parse_block(tokens))
        } else {
            panic!("Function declaration must be followed by definition or semicolon");
        };
        Declaration::Function(FunctionDeclaration {
            name: name.to_string(),
            params,
            body,
            storage,
            ctype: CType::Function(param_types, ctype.into()),
        })
    } else {
        let value = match tokens[0] {
            TokenType::Equal => {
                *tokens = &tokens[1..];
                Some(parse_expression(tokens, 0))
            }
            _ => None,
        };
        expect(tokens, TokenType::SemiColon);
        Declaration::Variable(VariableDeclaration {
            name: name.to_string(),
            value,
            ctype,
            storage,
        })
    }
}

fn parse_block_item(tokens: &mut &[TokenType]) -> BlockItem {
    if matches!(tokens[0], TokenType::Specifier(_)) {
        BlockItem::DeclareItem(parse_declaration(tokens))
    } else {
        BlockItem::StatementItem(parse_statement(tokens))
    }
}
fn parse_block(tokens: &mut &[TokenType]) -> Block {
    let mut body: Block = Vec::new();
    while tokens[0] != TokenType::CloseBrace {
        body.push(parse_block_item(tokens));
    }
    expect(tokens, TokenType::CloseBrace);
    body
}
fn parse_param_list(tokens: &mut &[TokenType]) -> (Vec<String>, Vec<CType>) {
    if try_consume(tokens, TokenType::Keyword("void")) {
        return (Vec::new(), Vec::new());
    }
    let mut params = Vec::new();
    let mut param_types = Vec::new();
    loop {
        param_types.push(parse_type(tokens));
        params.push(parse_identifier(tokens).to_string());
        if !try_consume(tokens, TokenType::Comma) {
            break;
        }
    }
    (params, param_types)
}
pub fn parse(tokens: &mut &[TokenType]) -> Program {
    // Parses entire program
    let mut program: Program = Vec::new();
    while !tokens.is_empty() {
        program.push(parse_declaration(tokens));
    }
    program
}
