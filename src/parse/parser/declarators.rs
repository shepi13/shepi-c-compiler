use crate::parse::{
    lexer::{Token, Tokens},
    parse_tree::CType,
};

use super::{parse_constant, parse_identifier, parse_type};

#[derive(Debug)]
pub enum Declarator {
    Identifier(String),
    Pointer(Box<Declarator>),
    Function(Vec<ParamInfo>, Box<Declarator>),
    Array(Box<Declarator>, u64),
}
#[derive(Debug)]
pub struct ParamInfo {
    pub ctype: CType,
    pub declarator: Box<Declarator>,
}

fn parse_simple_declarator(tokens: &mut Tokens) -> Declarator {
    if tokens.try_consume(Token::OpenParen) {
        let declarator = parse_declarator(tokens);
        tokens.expect(Token::CloseParen);
        declarator
    } else {
        Declarator::Identifier(parse_identifier(tokens).to_string())
    }
}

fn parse_direct_declarator(tokens: &mut Tokens) -> Declarator {
    let declarator = parse_simple_declarator(tokens);
    if tokens.peek() == Token::OpenBracket {
        parse_array_declarator(tokens, declarator)
    } else if tokens.try_consume(Token::OpenParen) {
        let param_info = parse_param_list(tokens)
            .into_iter()
            .map(|(ctype, declarator)| ParamInfo { ctype, declarator: declarator.into() })
            .collect();
        tokens.expect(Token::CloseParen);
        Declarator::Function(param_info, declarator.into())
    } else {
        declarator
    }
}

fn parse_array_declarator(tokens: &mut Tokens, mut declarator: Declarator) -> Declarator {
    while tokens.try_consume(Token::OpenBracket) {
        let subscript = parse_constant(&tokens.consume());
        assert!(subscript.is_integer(), "Array indices must be integers");
        let subscript = subscript.int_value() as u64;
        assert!(subscript > 0, "Array must have size > 0");
        declarator = Declarator::Array(declarator.into(), subscript);
        tokens.expect(Token::CloseBracket);
    }
    declarator
}

fn parse_param_list(tokens: &mut Tokens) -> Vec<(CType, Declarator)> {
    if tokens.try_consume(Token::Keyword("void")) {
        return Vec::new();
    }
    let mut params = Vec::new();
    loop {
        params.push(parse_param(tokens));
        if !tokens.try_consume(Token::Comma) {
            break;
        }
    }
    params
}
fn parse_param(tokens: &mut Tokens) -> (CType, Declarator) {
    let type_tokens = tokens.consume_type_specifiers();
    (parse_type(type_tokens), parse_declarator(tokens))
}

pub fn parse_declarator(tokens: &mut Tokens) -> Declarator {
    if tokens.try_consume(Token::Star) {
        Declarator::Pointer(parse_declarator(tokens).into())
    } else {
        parse_direct_declarator(tokens)
    }
}

#[derive(Debug)]
pub struct DeclaratorResult {
    pub name: String,
    pub ctype: CType,
    pub params: Vec<String>,
}
pub fn process_declarator(decl: Declarator, base_type: CType) -> DeclaratorResult {
    match decl {
        Declarator::Identifier(name) => DeclaratorResult {
            name,
            ctype: base_type,
            params: Vec::new(),
        },
        Declarator::Pointer(inner) => process_declarator(*inner, CType::Pointer(base_type.into())),
        Declarator::Array(inner, size) => {
            let derived_t = CType::Array(base_type.into(), size);
            process_declarator(*inner, derived_t)
        }
        Declarator::Function(params, inner) => match *inner {
            Declarator::Identifier(name) => {
                let mut param_types = Vec::new();
                let mut param_names = Vec::new();
                for param in params {
                    let result = process_declarator(*param.declarator, param.ctype);
                    assert!(
                        !matches!(result.ctype, CType::Function(_, _)),
                        "Function pointers not allowed in parameter",
                    );
                    param_names.push(result.name);
                    param_types.push(result.ctype);
                }
                DeclaratorResult {
                    name,
                    ctype: CType::Function(param_types, base_type.into()),
                    params: param_names,
                }
            }
            _ => panic!("Cannot apply additional type derivations to function!"),
        },
    }
}

#[derive(Debug)]
pub enum AbstractDeclarator {
    Base,
    Pointer(Box<AbstractDeclarator>),
    Array(Box<AbstractDeclarator>, u64),
}

pub fn parse_simple_abstract_declarator(tokens: &mut Tokens) -> AbstractDeclarator {
    if tokens.try_consume(Token::OpenParen) {
        let declarator = parse_abstract_declarator(tokens);
        tokens.expect(Token::CloseParen);
        declarator
    } else {
        AbstractDeclarator::Base
    }
}

fn parse_direct_abstract_declarator(tokens: &mut Tokens) -> AbstractDeclarator {
    let declarator = parse_simple_abstract_declarator(tokens);
    if tokens.peek() == Token::OpenBracket {
        parse_abstract_array_declarator(tokens, declarator)
    } else {
        declarator
    }
}

pub fn parse_abstract_declarator(tokens: &mut Tokens) -> AbstractDeclarator {
    if tokens.try_consume(Token::Star) {
        AbstractDeclarator::Pointer(parse_abstract_declarator(tokens).into())
    } else {
        parse_direct_abstract_declarator(tokens)
    }
}

fn parse_abstract_array_declarator(
    tokens: &mut Tokens,
    mut decl: AbstractDeclarator,
) -> AbstractDeclarator {
    while tokens.try_consume(Token::OpenBracket) {
        let subscript = parse_constant(&tokens.consume());
        assert!(subscript.is_integer(), "Array indices must be integers");
        let subscript = subscript.int_value() as u64;
        assert!(subscript > 0, "Array must have size > 0");
        decl = AbstractDeclarator::Array(decl.into(), subscript);
        tokens.expect(Token::CloseBracket);
    }
    decl
}

pub fn process_abstract_declarator(decl: AbstractDeclarator, base_type: CType) -> CType {
    match decl {
        AbstractDeclarator::Base => base_type,
        AbstractDeclarator::Pointer(inner) => {
            process_abstract_declarator(*inner, CType::Pointer(base_type.into()))
        }
        AbstractDeclarator::Array(inner, size) => {
            let derived_t = CType::Array(base_type.into(), size);
            process_abstract_declarator(*inner, derived_t)
        }
    }
}
