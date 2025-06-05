mod declarators;

use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use declarators::{
    parse_abstract_declarator, parse_declarator, process_abstract_declarator, process_declarator,
};
use lazy_static::lazy_static;

use crate::parse::lexer::Tokens;

use super::{lexer, parse_tree::VariableInitializer};
use lexer::Token;

use super::parse_tree::{
    AssignmentExpression, BinaryExpression, BinaryOperator, Block, BlockItem, CType,
    ConditionExpression, Constant, Declaration, Expression, ForInit, FunctionDeclaration,
    IncrementType, Loop, Program, Statement, StorageClass, SwitchStatement, TypedExpression,
    UnaryOperator, VariableDeclaration,
};

lazy_static! {
    static ref precedence_table: HashMap<Token<'static>, usize> = HashMap::from([
        (Token::Star, 50),
        (Token::ForwardSlash, 50),
        (Token::Percent, 50),
        (Token::Plus, 45),
        (Token::Hyphen, 45),
        (Token::LeftShift, 40),
        (Token::RightShift, 40),
        (Token::LessThan, 35),
        (Token::LessThanEqual, 35),
        (Token::GreaterThan, 35),
        (Token::GreaterThanEqual, 35),
        (Token::DoubleEqual, 30),
        (Token::ExclamEqual, 30),
        (Token::Ampersand, 25),
        (Token::Caret, 20),
        (Token::Pipe, 15),
        (Token::DoubleAmpersand, 10),
        (Token::DoublePipe, 5),
        (Token::QuestionMark, 3),
        (Token::Equal, 1),
        (Token::PlusEqual, 1),
        (Token::HyphenEqual, 1),
        (Token::StarEqual, 1),
        (Token::ForwardSlashEqual, 1),
        (Token::PercentEqual, 1),
        (Token::AmpersandEqual, 1),
        (Token::PipeEqual, 1),
        (Token::CaretEqual, 1),
        (Token::LeftShiftEqual, 1),
        (Token::RightShiftEqual, 1),
    ]);
}

fn loop_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("loop.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}
fn switch_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("switch.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn is_assignment_token(token: &Token) -> bool {
    use lexer::Token::*;
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

fn parse_type(tokens: &[Token]) -> CType {
    use Token::Specifier;
    // Count specifier tokens
    let count_token = |token| tokens.iter().filter(|elem| **elem == token).count();
    let signed_count = count_token(Specifier("signed"));
    let unsigned_count = count_token(Specifier("unsigned"));
    let int_count = count_token(Specifier("int"));
    let long_count = count_token(Specifier("long"));
    let double_count = count_token(Specifier("double"));

    let is_signed = match (signed_count, unsigned_count) {
        (0, 0) | (1, 0) => true,
        (0, 1) => false,
        _ => panic!("Cannot specify signed or unsigned more than once!"),
    };
    assert!(!tokens.is_empty(), "Must specify type!");
    assert!(int_count <= 1, "Repeated int keyword!");

    // Doubles must be either double or long double
    if double_count == 1 && [0, 1].contains(&long_count) {
        assert!(unsigned_count == 0, "Double cannot be unsigned!");
        assert!(signed_count == 0, "Double cannot be signed!");
        assert!(int_count == 0, "Mixed int/double declaration!");
        return CType::Double;
    }
    assert!(double_count == 0, "Invalid double declaration");

    match (long_count, is_signed) {
        (0, false) => CType::UnsignedInt,
        (0, true) => CType::Int,
        (1 | 2, false) => CType::UnsignedLong,
        (1 | 2, true) => CType::Long,
        _ => panic!("Too many long specifiers"),
    }
}

fn parse_specifiers(tokens: &mut Tokens) -> (CType, Option<StorageClass>) {
    let mut types = Vec::new();
    let mut storage_classes = Vec::new();
    while matches!(tokens.peek(), Token::Specifier(_)) {
        if tokens.peek().is_type_specifier() {
            types.push(tokens[0]);
        } else {
            storage_classes.push(StorageClass::from(&tokens[0]));
        }
        tokens.consume();
    }
    assert!(storage_classes.len() <= 1, "Can only have one storage class!");
    (parse_type(&mut &types[..]), storage_classes.pop())
}

fn parse_identifier(tokens: &mut Tokens) -> String {
    // Parses an identifier and advances the cursor
    let next_token = tokens.consume();
    match next_token {
        Token::Identifier(name) => name.to_string(),
        _ => panic!("Expected identifier"),
    }
}
fn parse_binop(tokens: &mut Tokens) -> BinaryOperator {
    // Advance cursor, and map tokens representing binary operations to binary op type
    use Token::*;
    let next_token = tokens.consume();
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
fn parse_argument_list(tokens: &mut Tokens) -> Vec<TypedExpression> {
    let mut args: Vec<TypedExpression> = Vec::new();
    let mut comma = false;
    while tokens[0] != Token::CloseParen {
        args.push(parse_expression(tokens, 0));
        comma = tokens.try_consume(Token::Comma);
    }
    assert!(!comma, "Trailing comma not allowed in C arg list");
    args
}
fn parse_post_operator(tokens: &mut Tokens, expression: TypedExpression) -> TypedExpression {
    if tokens.try_consume(Token::Increment) {
        parse_post_operator(
            tokens,
            Expression::Unary(
                UnaryOperator::Increment(IncrementType::PostIncrement),
                expression.into(),
            )
            .into(),
        )
    } else if tokens.try_consume(Token::Decrement) {
        parse_post_operator(
            tokens,
            Expression::Unary(
                UnaryOperator::Increment(IncrementType::PostDecrement),
                expression.into(),
            )
            .into(),
        )
    } else if tokens.try_consume(Token::OpenBracket) {
        let sub_expr = parse_expression(tokens, 0);
        tokens.expect(Token::CloseBracket);
        parse_post_operator(
            tokens,
            Expression::Subscript(expression.into(), sub_expr.into()).into(),
        )
    } else {
        expression
    }
}

fn parse_constant(token: &Token) -> Constant {
    match token {
        Token::Constant(val) => match val.parse::<i32>() {
            Ok(val) => Constant::Int(val.into()),
            Err(_) => Constant::Long(val.parse().expect("Failed to convert constant to int")),
        },
        Token::ConstantLong(val) => {
            Constant::Long(val.parse().expect("Failed to convert constant to long"))
        }
        Token::Unsigned(val) => match val.parse::<u32>() {
            Ok(val) => Constant::UnsignedInt(val.into()),
            Err(_) => Constant::UnsignedLong(
                val.parse().expect("Failed to convert unsigned constant to int"),
            ),
        },
        Token::UnsignedLong(val) => Constant::UnsignedLong(
            val.parse().expect("Failed to convert unsigned constant to long"),
        ),
        Token::Double(val) => Constant::Double(val.parse().expect("Failed to convert to double!")),
        _ => panic!("Expected constant!"),
    }
}

fn parse_factor(tokens: &mut Tokens) -> TypedExpression {
    // Parses a factor (unary value/operator) of a larger expression
    let token = tokens.consume();
    let expr = match token {
        Token::Constant(_)
        | Token::ConstantLong(_)
        | Token::Unsigned(_)
        | Token::UnsignedLong(_)
        | Token::Double(_) => Expression::Constant(parse_constant(&token)),
        Token::Hyphen => Expression::Unary(UnaryOperator::Negate, parse_factor(tokens).into()),
        Token::Tilde => Expression::Unary(UnaryOperator::Complement, parse_factor(tokens).into()),
        Token::Exclam => Expression::Unary(UnaryOperator::LogicalNot, parse_factor(tokens).into()),
        Token::Ampersand => Expression::AddrOf(parse_factor(tokens).into()),
        Token::Star => Expression::Dereference(parse_factor(tokens).into()),
        Token::Increment => Expression::Unary(
            UnaryOperator::Increment(IncrementType::PreIncrement),
            parse_factor(tokens).into(),
        ),
        Token::Decrement => Expression::Unary(
            UnaryOperator::Increment(IncrementType::PreDecrement),
            parse_factor(tokens).into(),
        ),
        Token::OpenParen => {
            if tokens.peek().is_type_specifier() {
                let type_tokens = tokens.consume_type_specifiers();
                let base_type = parse_type(type_tokens);
                let abstract_decl = parse_abstract_declarator(tokens);
                let ctype = process_abstract_declarator(abstract_decl, base_type);
                tokens.expect(Token::CloseParen);
                Expression::Cast(ctype, parse_factor(tokens).into())
            } else {
                let expr = parse_expression(tokens, 0);
                tokens.expect(Token::CloseParen);
                expr.expr
            }
        }
        Token::Identifier(name) => {
            let name = name.to_string();
            if tokens.try_consume(Token::OpenParen) {
                let args = parse_argument_list(tokens);
                tokens.expect(Token::CloseParen);
                Expression::FunctionCall(name, args)
            } else {
                Expression::Variable(name)
            }
        }
        _ => panic!("Expected factor, found {:?}", token),
    };
    parse_post_operator(tokens, expr.into())
}
fn parse_expression(tokens: &mut Tokens, min_prec: usize) -> TypedExpression {
    // Parses an expression using precedence climbing
    let mut left = parse_factor(tokens);
    while precedence_table.contains_key(&tokens[0]) && precedence_table[&tokens[0]] >= min_prec {
        let token_prec = precedence_table[&tokens[0]];
        if tokens.try_consume(Token::Equal) {
            let right = parse_expression(tokens, token_prec);
            left = Expression::Assignment(AssignmentExpression { left, right }.into()).into();
        } else if tokens.try_consume(Token::QuestionMark) {
            let if_true = parse_expression(tokens, 0);
            tokens.expect(Token::Colon);
            let if_false = parse_expression(tokens, token_prec);
            left = Expression::Condition(
                ConditionExpression { condition: left, if_true, if_false }.into(),
            )
            .into();
        } else {
            let is_assignment = is_assignment_token(&tokens[0]);
            let operator = parse_binop(tokens);
            let right = if is_assignment {
                parse_expression(tokens, token_prec)
            } else {
                parse_expression(tokens, token_prec + 1)
            };
            left = Expression::Binary(
                BinaryExpression { operator, left, right, is_assignment }.into(),
            )
            .into();
        }
    }
    left
}
fn parse_optional_expression(tokens: &mut Tokens) -> Option<TypedExpression> {
    use Token::*;
    match tokens[0] {
        Constant(_) | Hyphen | Tilde | Exclam | OpenParen | Identifier(_) => {
            Some(parse_expression(tokens, 0))
        }
        _ => None,
    }
}
fn parse_statement(tokens: &mut Tokens) -> Statement {
    // Parses a statement
    if tokens.try_consume(Token::Keyword("return")) {
        let return_value = parse_expression(tokens, 0);
        tokens.expect(Token::SemiColon);
        Statement::Return(return_value)
    } else if tokens.try_consume(Token::SemiColon) {
        Statement::Null
    } else if tokens.try_consume(Token::Keyword("if")) {
        tokens.expect(Token::OpenParen);
        let cond = parse_expression(tokens, 0);
        tokens.expect(Token::CloseParen);
        let then = parse_statement(tokens);
        let otherwise = if tokens.try_consume(Token::Keyword("else")) {
            Some(parse_statement(tokens))
        } else {
            None
        };
        Statement::If(cond, then.into(), otherwise.into())
    } else if tokens.try_consume(Token::Keyword("goto")) {
        let target = parse_identifier(tokens).to_string();
        tokens.expect(Token::SemiColon);
        Statement::Goto(target)
    } else if tokens.try_consume(Token::OpenBrace) {
        Statement::Compound(parse_block(tokens))
    } else if tokens.try_consume(Token::Keyword("break")) {
        tokens.expect(Token::SemiColon);
        Statement::Break(String::new())
    } else if tokens.try_consume(Token::Keyword("continue")) {
        tokens.expect(Token::SemiColon);
        Statement::Continue(String::new())
    } else if tokens.try_consume(Token::Keyword("while")) {
        tokens.expect(Token::OpenParen);
        let condition = parse_expression(tokens, 0);
        tokens.expect(Token::CloseParen);
        Statement::While(Loop {
            label: loop_name(),
            condition,
            body: parse_statement(tokens).into(),
        })
    } else if tokens.try_consume(Token::Keyword("do")) {
        let body = parse_statement(tokens);
        tokens.expect(Token::Keyword("while"));
        tokens.expect(Token::OpenParen);
        let condition = parse_expression(tokens, 0);
        tokens.expect(Token::CloseParen);
        tokens.expect(Token::SemiColon);
        Statement::DoWhile(Loop {
            label: loop_name(),
            condition,
            body: body.into(),
        })
    } else if tokens.try_consume(Token::Keyword("for")) {
        tokens.expect(Token::OpenParen);
        let init = if matches!(tokens[0], Token::Specifier(_)) {
            ForInit::Decl(parse_variable_declaration(tokens))
        } else {
            let init = ForInit::Expr(parse_optional_expression(tokens));
            tokens.expect(Token::SemiColon);
            init
        };
        let condition = match parse_optional_expression(tokens) {
            Some(expr) => expr,
            None => Expression::Constant(Constant::Int(1)).into(),
        };
        tokens.expect(Token::SemiColon);
        let post_loop = parse_optional_expression(tokens);
        tokens.expect(Token::CloseParen);
        Statement::For(
            init,
            Loop {
                label: loop_name(),
                condition,
                body: parse_statement(tokens).into(),
            },
            post_loop,
        )
    } else if tokens.try_consume(Token::Keyword("switch")) {
        let label = switch_name();
        tokens.expect(Token::OpenParen);
        let condition = parse_expression(tokens, 0);
        tokens.expect(Token::CloseParen);
        let cases = Vec::new();
        let statement = parse_statement(tokens).into();
        Statement::Switch(SwitchStatement {
            label,
            condition,
            cases,
            statement,
            default: None,
        })
    } else if tokens.try_consume(Token::Keyword("default")) {
        tokens.expect(Token::Colon);
        Statement::Default(parse_statement(tokens).into())
    } else if tokens.try_consume(Token::Keyword("case")) {
        let matcher = parse_expression(tokens, 0);
        tokens.expect(Token::Colon);
        let statement = parse_statement(tokens).into();
        Statement::Case(matcher, statement)
    } else if matches!(tokens[0], Token::Identifier(_)) && tokens[1] == Token::Colon {
        let Token::Identifier(name) = tokens.consume() else { panic!("Unreachable!") };
        let label_name = name.to_string();
        tokens.expect(Token::Colon);
        Statement::Label(label_name, parse_statement(tokens).into())
    } else {
        let expr = parse_expression(tokens, 0);
        tokens.expect(Token::SemiColon);
        Statement::ExprStmt(expr)
    }
}
fn parse_variable_declaration(tokens: &mut Tokens) -> VariableDeclaration {
    let decl = parse_declaration(tokens);
    match decl {
        Declaration::Variable(var_decl) => var_decl,
        _ => panic!("Expected variable declaration"),
    }
}

fn parse_declaration(tokens: &mut Tokens) -> Declaration {
    let (ctype, storage) = parse_specifiers(tokens);
    let declarator = parse_declarator(tokens);
    let decl_result = process_declarator(declarator, ctype);
    if matches!(decl_result.ctype, CType::Function(_, _)) {
        let body = if tokens.try_consume(Token::OpenBrace) {
            Some(parse_block(tokens))
        } else {
            tokens.expect(Token::SemiColon);
            None
        };
        Declaration::Function(FunctionDeclaration {
            name: decl_result.name,
            ctype: decl_result.ctype,
            params: decl_result.params,
            storage,
            body,
        })
    } else {
        let init =
            if tokens.try_consume(Token::Equal) { Some(parse_initializer(tokens)) } else { None };
        tokens.expect(Token::SemiColon);
        Declaration::Variable(VariableDeclaration {
            name: decl_result.name,
            ctype: decl_result.ctype,
            init,
            storage,
        })
    }
}

fn parse_initializer(tokens: &mut Tokens) -> VariableInitializer {
    if tokens.try_consume(Token::OpenBrace) {
        assert!(tokens.peek() != Token::CloseBrace, "Empty initializer invalid pre C23");
        let mut initializers = Vec::new();
        while !tokens.try_consume(Token::CloseBrace) {
            initializers.push(parse_initializer(tokens));
            if !tokens.try_consume(Token::Comma) {
                tokens.expect(Token::CloseBrace);
                break;
            }
        }
        VariableInitializer::CompoundInit(initializers)
    } else {
        VariableInitializer::SingleElem(parse_expression(tokens, 0))
    }
}

fn parse_block_item(tokens: &mut Tokens) -> BlockItem {
    if matches!(tokens.peek(), Token::Specifier(_)) {
        BlockItem::DeclareItem(parse_declaration(tokens))
    } else {
        BlockItem::StatementItem(parse_statement(tokens))
    }
}
fn parse_block(tokens: &mut Tokens) -> Block {
    let mut body: Block = Vec::new();
    while tokens.peek() != Token::CloseBrace {
        body.push(parse_block_item(tokens));
    }
    tokens.expect(Token::CloseBrace);
    body
}
pub fn parse(tokens: &mut Tokens) -> Program {
    // Parses entire program
    let mut program: Program = Vec::new();
    while !tokens.is_empty() {
        program.push(parse_declaration(tokens));
    }
    program
}
