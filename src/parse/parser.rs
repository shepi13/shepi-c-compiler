use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::declarators::{
    parse_abstract_declarator, parse_declarator, process_abstract_declarator, process_declarator,
};
use lazy_static::lazy_static;

use crate::{
    helpers::{
        error::Error,
        lib::{unescape_char, unescape_string},
    },
    validate::ctype::CType,
};

use super::lexer::{Token, Tokens};
use super::parse_tree::{
    AssignmentExpression, BinaryExpression, BinaryOperator, Block, BlockItem, ConditionExpression,
    Constant, Declaration, Expression, ForInit, FunctionDeclaration, IncrementType, Location, Loop,
    MemberDeclaration, Program, Statement, StorageClass, TypedExpression, UnaryOperator,
    VariableDeclaration, VariableInitializer,
};

type Result<T> = std::result::Result<T, Error>;

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

pub fn parse_type(tokens: &[Token]) -> std::result::Result<CType, &'static str> {
    use Token::Specifier;
    let assert = |cond, msg: &'static str| if cond { Ok(()) } else { Err(msg) };
    // Handle struct/union, which can't appear with other type specifiers and must have an identifier
    if matches!(tokens[0], Token::Specifier("struct" | "union")) {
        assert(tokens.len() == 2, "Struct must have identifier and no other type specifiers!")?;
        let Token::Identifier(identifier) = tokens[1] else {
            return Err("Failed to parse name of struct/union");
        };
        return if tokens[0] == Token::Specifier("struct") {
            Ok(CType::Structure(identifier.to_string()))
        } else {
            Ok(CType::Union(identifier.to_string()))
        };
    }
    // Count specifier tokens
    let count_token = |token| tokens.iter().filter(|elem| **elem == token).count();
    let void_count = count_token(Specifier("void"));
    let signed_count = count_token(Specifier("signed"));
    let unsigned_count = count_token(Specifier("unsigned"));
    let int_count = count_token(Specifier("int"));
    let long_count = count_token(Specifier("long"));
    let double_count = count_token(Specifier("double"));
    let char_count = count_token(Specifier("char"));

    let is_signed = match (signed_count, unsigned_count) {
        (0, 0) | (1, 0) => true,
        (0, 1) => false,
        _ => return Err("Cannot specify signed or unsigned more than once!"),
    };
    assert(!tokens.is_empty(), "Must specify type!")?;
    assert(int_count <= 1, "Repeated int keyword!")?;

    // Void type
    if void_count > 0 {
        assert(tokens.len() == 1, "Void type cannot be combined with other specifiers!")?;
        return Ok(CType::Void);
    }

    // Double and long double declaration and checks
    if double_count == 1 && [0, 1].contains(&long_count) {
        assert(unsigned_count == 0, "Double cannot be unsigned!")?;
        assert(signed_count == 0, "Double cannot be signed!")?;
        assert(int_count == 0, "Mixed int/double declaration!")?;
        assert(char_count == 0, "Mixed char/double declaration!")?;
        return Ok(CType::Double);
    }
    assert(double_count == 0, "Invalid double declaration")?;
    // Char declaration and checks
    if char_count == 1 {
        assert(long_count == 0, "Char cannot be long!")?;
        assert(int_count == 0, "Mixed int/char declaration!")?;
        if signed_count == 1 {
            return Ok(CType::SignedChar);
        } else if unsigned_count == 1 {
            return Ok(CType::UnsignedChar);
        } else {
            return Ok(CType::Char);
        }
    }
    assert(char_count == 0, "Invalid char declaration!")?;

    match (long_count, is_signed) {
        (0, false) => Ok(CType::UnsignedInt),
        (0, true) => Ok(CType::Int),
        (1 | 2, false) => Ok(CType::UnsignedLong),
        (1 | 2, true) => Ok(CType::Long),
        _ => Err("Too many long specifiers"),
    }
}

fn parse_specifiers(tokens: &mut Tokens) -> Result<(CType, Option<StorageClass>)> {
    let mut types = Vec::new();
    let mut storage_classes = Vec::new();
    while matches!(tokens.peek(), Token::Specifier(_)) {
        if tokens.peek().is_type_specifier() {
            types.push(tokens[0]);
            // Struct/Union have to have identifier for type parsing
            if matches!(tokens.peek(), Token::Specifier("struct" | "union")) {
                types.push(tokens[1]);
                tokens.consume();
            }
        } else {
            storage_classes.push(StorageClass::from(&tokens[0]));
        }
        tokens.consume();
    }
    tokens.assert(storage_classes.len() <= 1, "Can only have one storage class!")?;
    let ctype = parse_type(&types[..]).map_err(|err| tokens.parse_error(err))?;
    Ok((ctype, storage_classes.pop()))
}

pub fn parse_identifier(tokens: &mut Tokens) -> Result<String> {
    // Parses an identifier and advances the cursor
    let next_token = tokens.consume();
    match next_token {
        Token::Identifier(name) => Ok(name.to_string()),
        _ => {
            tokens.rewind();
            Err(tokens.parse_error("Expected identifier"))
        }
    }
}
fn parse_binop(tokens: &mut Tokens) -> Result<BinaryOperator> {
    // Advance cursor, and map tokens representing binary operations to binary op type
    use Token::*;
    let next_token = tokens.consume();
    match next_token {
        // Math or Assignment
        Plus | PlusEqual => Ok(BinaryOperator::Add),
        Hyphen | HyphenEqual => Ok(BinaryOperator::Subtract),
        Star | StarEqual => Ok(BinaryOperator::Multiply),
        ForwardSlash | ForwardSlashEqual => Ok(BinaryOperator::Divide),
        Percent | PercentEqual => Ok(BinaryOperator::Remainder),
        LeftShift | LeftShiftEqual => Ok(BinaryOperator::LeftShift),
        RightShift | RightShiftEqual => Ok(BinaryOperator::RightShift),
        Pipe | PipeEqual => Ok(BinaryOperator::BitOr),
        Caret | CaretEqual => Ok(BinaryOperator::BitXor),
        Ampersand | AmpersandEqual => Ok(BinaryOperator::BitAnd),
        // Relational
        LessThan => Ok(BinaryOperator::LessThan),
        LessThanEqual => Ok(BinaryOperator::LessThanEqual),
        GreaterThan => Ok(BinaryOperator::GreaterThan),
        GreaterThanEqual => Ok(BinaryOperator::GreaterThanEqual),
        DoubleEqual => Ok(BinaryOperator::IsEqual),
        ExclamEqual => Ok(BinaryOperator::NotEqual),
        DoubleAmpersand => Ok(BinaryOperator::LogicalAnd),
        DoublePipe => Ok(BinaryOperator::LogicalOr),
        _ => Err(tokens.parse_error("Expected binary operator!")),
    }
}

fn parse_argument_list(tokens: &mut Tokens) -> Result<Vec<TypedExpression>> {
    let mut args: Vec<TypedExpression> = Vec::new();
    let mut comma = false;
    while tokens[0] != Token::CloseParen {
        args.push(parse_expression(tokens, 0)?);
        comma = tokens.try_consume(Token::Comma);
    }
    tokens.assert(!comma, "Trailing comma not allowed in C arg list")?;
    Ok(args)
}
fn parse_post_operator(
    tokens: &mut Tokens,
    expression: TypedExpression,
) -> Result<TypedExpression> {
    if tokens.try_consume(Token::Dot) {
        let tag = parse_identifier(tokens)?;
        parse_post_operator(tokens, Expression::DotAccess(expression.into(), tag).into())
    } else if tokens.try_consume(Token::Arrow) {
        let tag = parse_identifier(tokens)?;
        parse_post_operator(tokens, Expression::Arrow(expression.into(), tag).into())
    } else if tokens.try_consume(Token::Increment) {
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
        let sub_expr = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::CloseBracket)?;
        parse_post_operator(
            tokens,
            Expression::Subscript(expression.into(), sub_expr.into()).into(),
        )
    } else {
        Ok(expression)
    }
}

pub fn parse_constant(tokens: &mut Tokens) -> Result<Constant> {
    let result = try_parse_constant(tokens.consume());
    match result {
        Ok(constant) => Ok(constant),
        Err(_) => Err(tokens.parse_error("Failed to parse constant!")),
    }
}
fn try_parse_constant(token: Token) -> std::result::Result<Constant, Box<dyn std::error::Error>> {
    match token {
        Token::Character(data) => match unescape_char(data) {
            Ok(val) => Ok(Constant::Int(val as i64)),
            Err(message) => Err(message.into()),
        },
        Token::Constant(val) => match val.parse::<i32>() {
            Ok(val) => Ok(Constant::Int(val.into())),
            Err(_) => Ok(Constant::Long(val.parse()?)),
        },
        Token::ConstantLong(val) => Ok(Constant::Long(val.parse()?)),
        Token::Unsigned(val) => match val.parse::<u32>() {
            Ok(val) => Ok(Constant::UInt(val.into())),
            Err(_) => Ok(Constant::ULong(val.parse()?)),
        },
        Token::UnsignedLong(val) => Ok(Constant::ULong(val.parse()?)),
        Token::Double(val) => Ok(Constant::Double(val.parse()?)),
        _ => Err("Unexpected token!".into()),
    }
}

fn parse_type_name(tokens: &mut Tokens) -> Result<CType> {
    let type_tokens = tokens.consume_type_specifiers();
    let base_type = parse_type(type_tokens).map_err(|err| tokens.parse_error(err))?;
    let abstract_decl = parse_abstract_declarator(tokens)?;
    process_abstract_declarator(abstract_decl, base_type)
}

fn parse_cast_expression(tokens: &mut Tokens) -> Result<TypedExpression> {
    if tokens[0] == Token::OpenParen && tokens[1].is_type_specifier() {
        tokens.expect_token(Token::OpenParen)?;
        let ctype = parse_type_name(tokens)?;
        tokens.expect_token(Token::CloseParen)?;
        Ok(Expression::Cast(ctype, parse_cast_expression(tokens)?.into()).into())
    } else {
        parse_factor(tokens)
    }
}

fn parse_factor(tokens: &mut Tokens) -> Result<TypedExpression> {
    // Parses a factor (unary value/operator) of a larger expression
    let token = tokens.consume();
    let expr = match token {
        Token::Constant(_)
        | Token::ConstantLong(_)
        | Token::Unsigned(_)
        | Token::UnsignedLong(_)
        | Token::Character(_)
        | Token::Double(_) => {
            tokens.rewind();
            Expression::Constant(parse_constant(tokens)?)
        }
        Token::String(_) => {
            tokens.rewind();
            let mut string_data = String::new();
            while matches!(tokens.peek(), Token::String(_)) {
                let Token::String(data) = tokens.consume() else { panic!("Is string") };
                match unescape_string(data) {
                    Ok(data) => string_data.push_str(&data),
                    Err(message) => {
                        tokens.rewind();
                        return Err(tokens.parse_error(&message));
                    }
                }
            }
            Expression::StringLiteral(string_data)
        }
        Token::Keyword("sizeof") => {
            if tokens[0] == Token::OpenParen && tokens[1].is_type_specifier() {
                tokens.expect_token(Token::OpenParen)?;
                let ctype = parse_type_name(tokens)?;
                tokens.expect_token(Token::CloseParen)?;
                Expression::SizeOfT(ctype)
            } else {
                Expression::SizeOf(parse_factor(tokens)?.into())
            }
        }
        Token::Hyphen => {
            Expression::Unary(UnaryOperator::Negate, parse_cast_expression(tokens)?.into())
        }
        Token::Tilde => {
            Expression::Unary(UnaryOperator::Complement, parse_cast_expression(tokens)?.into())
        }
        Token::Exclam => {
            Expression::Unary(UnaryOperator::LogicalNot, parse_cast_expression(tokens)?.into())
        }
        Token::Ampersand => Expression::AddrOf(parse_cast_expression(tokens)?.into()),
        Token::Star => Expression::Dereference(parse_cast_expression(tokens)?.into()),
        Token::Increment => Expression::Unary(
            UnaryOperator::Increment(IncrementType::PreIncrement),
            parse_cast_expression(tokens)?.into(),
        ),
        Token::Decrement => Expression::Unary(
            UnaryOperator::Increment(IncrementType::PreDecrement),
            parse_cast_expression(tokens)?.into(),
        ),
        Token::OpenParen => {
            let expr = parse_expression(tokens, 0)?;
            tokens.expect_token(Token::CloseParen)?;
            expr.expr
        }
        Token::Identifier(name) => {
            let name = name.to_string();
            if tokens.try_consume(Token::OpenParen) {
                let args = parse_argument_list(tokens)?;
                tokens.expect_token(Token::CloseParen)?;
                Expression::FunctionCall(name, args)
            } else {
                Expression::Variable(name)
            }
        }
        _ => {
            return Err(tokens.parse_error("Expected factor!"));
        }
    };
    parse_post_operator(tokens, expr.into())
}
fn parse_expression(tokens: &mut Tokens, min_prec: usize) -> Result<TypedExpression> {
    // Parses an expression using precedence climbing
    let mut left = parse_cast_expression(tokens)?;
    while precedence_table.contains_key(&tokens[0]) && precedence_table[&tokens[0]] >= min_prec {
        let token_prec = precedence_table[&tokens[0]];
        if tokens.try_consume(Token::Equal) {
            let right = parse_expression(tokens, token_prec)?;
            left = Expression::Assignment(AssignmentExpression { left, right }.into()).into();
        } else if tokens.try_consume(Token::QuestionMark) {
            let if_true = parse_expression(tokens, 0)?;
            tokens.expect_token(Token::Colon)?;
            let if_false = parse_expression(tokens, token_prec)?;
            left = Expression::Condition(
                ConditionExpression { condition: left, if_true, if_false }.into(),
            )
            .into();
        } else {
            let is_assignment = tokens[0].is_assignment_token();
            let operator = parse_binop(tokens)?;
            let right = if is_assignment {
                parse_expression(tokens, token_prec)?
            } else {
                parse_expression(tokens, token_prec + 1)?
            };
            left = Expression::Binary(
                BinaryExpression { operator, left, right, is_assignment }.into(),
            )
            .into();
        }
    }
    Ok(left)
}
fn parse_optional_expression(tokens: &mut Tokens) -> Result<Option<TypedExpression>> {
    use Token::*;
    match tokens[0] {
        Constant(_) | Hyphen | Tilde | Exclam | OpenParen | Identifier(_) => {
            Ok(Some(parse_expression(tokens, 0)?))
        }
        _ => Ok(None),
    }
}
fn parse_statement(tokens: &mut Tokens) -> Result<Statement> {
    // Parses a statement
    if tokens.try_consume(Token::Keyword("return")) {
        if tokens.try_consume(Token::SemiColon) {
            Ok(Statement::Return(None))
        } else {
            let return_value = parse_expression(tokens, 0)?;
            tokens.expect_token(Token::SemiColon)?;
            Ok(Statement::Return(Some(return_value)))
        }
    } else if tokens.try_consume(Token::SemiColon) {
        Ok(Statement::Null)
    } else if tokens.try_consume(Token::Keyword("if")) {
        tokens.expect_token(Token::OpenParen)?;
        let cond = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::CloseParen)?;
        let then = parse_statement(tokens)?;
        let otherwise = if tokens.try_consume(Token::Keyword("else")) {
            Some(parse_statement(tokens)?)
        } else {
            None
        };
        Ok(Statement::If(cond, then.into(), otherwise.into()))
    } else if tokens.try_consume(Token::Keyword("goto")) {
        let target = parse_identifier(tokens)?.to_string();
        tokens.expect_token(Token::SemiColon)?;
        Ok(Statement::Goto(target))
    } else if tokens.try_consume(Token::OpenBrace) {
        Ok(Statement::Compound(parse_block(tokens)?))
    } else if tokens.try_consume(Token::Keyword("break")) {
        tokens.expect_token(Token::SemiColon)?;
        Ok(Statement::Break(String::new()))
    } else if tokens.try_consume(Token::Keyword("continue")) {
        tokens.expect_token(Token::SemiColon)?;
        Ok(Statement::Continue(String::new()))
    } else if tokens.try_consume(Token::Keyword("while")) {
        tokens.expect_token(Token::OpenParen)?;
        let condition = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::CloseParen)?;
        Ok(Statement::While(Loop {
            label: loop_name(),
            condition,
            body: parse_statement(tokens)?.into(),
        }))
    } else if tokens.try_consume(Token::Keyword("do")) {
        let body = parse_statement(tokens)?;
        tokens.expect_token(Token::Keyword("while"))?;
        tokens.expect_token(Token::OpenParen)?;
        let condition = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::CloseParen)?;
        tokens.expect_token(Token::SemiColon)?;
        Ok(Statement::DoWhile(Loop {
            label: loop_name(),
            condition,
            body: body.into(),
        }))
    } else if tokens.try_consume(Token::Keyword("for")) {
        tokens.expect_token(Token::OpenParen)?;
        let init = if matches!(tokens[0], Token::Specifier(_)) {
            ForInit::Decl(parse_variable_declaration(tokens)?)
        } else {
            let init = ForInit::Expr(parse_optional_expression(tokens)?);
            tokens.expect_token(Token::SemiColon)?;
            init
        };
        let condition = match parse_optional_expression(tokens)? {
            Some(expr) => expr,
            None => Expression::Constant(Constant::Int(1)).into(),
        };
        tokens.expect_token(Token::SemiColon)?;
        let post_loop = parse_optional_expression(tokens)?;
        tokens.expect_token(Token::CloseParen)?;
        Ok(Statement::For(
            init.into(),
            Loop {
                label: loop_name(),
                condition,
                body: parse_statement(tokens)?.into(),
            },
            post_loop,
        ))
    } else if tokens.try_consume(Token::Keyword("switch")) {
        let label = switch_name();
        tokens.expect_token(Token::OpenParen)?;
        let condition = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::CloseParen)?;
        let cases = Vec::new();
        let statement = parse_statement(tokens)?.into();
        Ok(Statement::Switch {
            label,
            condition,
            cases,
            statement,
            default: None,
        })
    } else if tokens.try_consume(Token::Keyword("default")) {
        tokens.expect_token(Token::Colon)?;
        Ok(Statement::Default(parse_statement(tokens)?.into()))
    } else if tokens.try_consume(Token::Keyword("case")) {
        let matcher = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::Colon)?;
        let statement = parse_statement(tokens)?.into();
        Ok(Statement::Case(matcher, statement))
    } else if matches!(tokens[0], Token::Identifier(_)) && tokens[1] == Token::Colon {
        let Token::Identifier(name) = tokens.consume() else { panic!("Unreachable!") };
        let label_name = name.to_string();
        tokens.expect_token(Token::Colon)?;
        Ok(Statement::Label(label_name, parse_statement(tokens)?.into()))
    } else {
        let expr = parse_expression(tokens, 0)?;
        tokens.expect_token(Token::SemiColon)?;
        Ok(Statement::ExprStmt(expr))
    }
}
fn parse_variable_declaration(tokens: &mut Tokens) -> Result<VariableDeclaration> {
    let decl = parse_declaration(tokens)?;
    match decl {
        Declaration::Variable(var_decl) => Ok(var_decl),
        _ => Err(tokens.parse_error("Expected variable declaration")),
    }
}

fn parse_members(tokens: &mut Tokens) -> Result<Vec<MemberDeclaration>> {
    let mut members = Vec::new();
    if tokens.try_consume(Token::OpenBrace) {
        while !tokens.try_consume(Token::CloseBrace) {
            let type_tokens = tokens.consume_type_specifiers();
            let ctype = parse_type(type_tokens).or_else(|err| Err(tokens.parse_error(err)))?;
            let declarator = parse_declarator(tokens)?;
            let decl_result = process_declarator(declarator, ctype)?;
            tokens.assert(
                !matches!(decl_result.ctype, CType::Function(_, _)),
                "Function struct members not supported!",
            )?;
            members.push(MemberDeclaration {
                name: decl_result.name,
                ctype: decl_result.ctype,
            });
            tokens.expect_token(Token::SemiColon)?;
        }
        tokens.assert(!members.is_empty(), "Struct/Union initializer cannot be empty!")?;
    }
    Ok(members)
}

fn parse_declaration(tokens: &mut Tokens) -> Result<Declaration> {
    let start_loc = tokens.location();
    let (ctype, storage) = parse_specifiers(tokens)?;
    if tokens.peek() == Token::SemiColon || tokens.peek() == Token::OpenBrace {
        if let CType::Structure(tag) = ctype {
            let members = parse_members(tokens)?;
            tokens.expect_token(Token::SemiColon)?;
            return Ok(Declaration::Struct { tag, members });
        } else if let CType::Union(tag) = ctype {
            let members = parse_members(tokens)?;
            tokens.expect_token(Token::SemiColon)?;
            return Ok(Declaration::Union { tag, members });
        }
    }
    let declarator = parse_declarator(tokens)?;
    let decl_result = process_declarator(declarator, ctype)?;
    match decl_result.ctype {
        CType::Function(_, _) => {
            let body = if tokens.try_consume(Token::OpenBrace) {
                Some(parse_block(tokens)?)
            } else {
                tokens.expect_token(Token::SemiColon)?;
                None
            };
            Ok(Declaration::Function(FunctionDeclaration {
                name: decl_result.name,
                ctype: decl_result.ctype,
                params: decl_result.params,
                storage,
                body,
                location: Location { start_loc, end_loc: tokens.location() },
            }))
        }
        _ => {
            let init = if tokens.try_consume(Token::Equal) {
                Some(parse_initializer(tokens)?)
            } else {
                None
            };
            tokens.expect_token(Token::SemiColon)?;
            Ok(Declaration::Variable(VariableDeclaration {
                name: decl_result.name,
                ctype: decl_result.ctype,
                init,
                storage,
                location: Location { start_loc, end_loc: tokens.location() },
            }))
        }
    }
}

fn parse_initializer(tokens: &mut Tokens) -> Result<VariableInitializer> {
    if tokens.try_consume(Token::OpenBrace) {
        tokens.assert(tokens.peek() != Token::CloseBrace, "Empty initializer!")?;
        let mut initializers = Vec::new();
        while !tokens.try_consume(Token::CloseBrace) {
            initializers.push(parse_initializer(tokens)?);
            if !tokens.try_consume(Token::Comma) {
                tokens.expect_token(Token::CloseBrace)?;
                break;
            }
        }
        Ok(VariableInitializer::CompoundInit(initializers))
    } else {
        Ok(VariableInitializer::SingleElem(parse_expression(tokens, 0)?))
    }
}

fn parse_block_item(tokens: &mut Tokens) -> Result<BlockItem> {
    if matches!(tokens.peek(), Token::Specifier(_)) {
        Ok(BlockItem::DeclareItem(parse_declaration(tokens)?))
    } else {
        let start_loc = tokens.location();
        let stmt = parse_statement(tokens)?;
        let end_loc = tokens.location();
        Ok(BlockItem::StatementItem(stmt, Location { start_loc, end_loc }))
    }
}
fn parse_block(tokens: &mut Tokens) -> Result<Block> {
    let mut body: Block = Vec::new();
    while tokens.peek() != Token::CloseBrace {
        body.push(parse_block_item(tokens)?);
    }
    tokens.expect_token(Token::CloseBrace)?;
    Ok(body)
}
pub fn parse(tokens: &mut Tokens) -> Result<Program> {
    // Parses entire program
    let mut program: Program = Vec::new();
    while !tokens.is_empty() {
        program.push(parse_declaration(tokens)?);
    }
    Ok(program)
}
