use crate::parse::{lexer::TokenType, parse_tree::CType};

use super::{expect, parse_identifier, parse_type, try_consume};

#[derive(Debug)]
pub enum Declarator {
    Identifier(String),
    Pointer(Box<Declarator>),
    Function(Vec<ParamInfo>, Box<Declarator>),
}
#[derive(Debug)]
pub struct ParamInfo {
    pub ctype: CType,
    pub declarator: Box<Declarator>,
}

fn parse_simple_declarator(tokens: &mut &[TokenType]) -> Declarator {
    if try_consume(tokens, TokenType::OpenParen) {
        let declarator = parse_declarator(tokens);
        expect(tokens, TokenType::CloseParen);
        declarator
    } else {
        Declarator::Identifier(parse_identifier(tokens).to_string())
    }
}

fn parse_direct_declarator(tokens: &mut &[TokenType]) -> Declarator {
    let declarator = parse_simple_declarator(tokens);
    if try_consume(tokens, TokenType::OpenParen) {
        let param_info = parse_param_list(tokens)
            .into_iter()
            .map(|(ctype, declarator)| ParamInfo { ctype, declarator: declarator.into() })
            .collect();
        expect(tokens, TokenType::CloseParen);
        Declarator::Function(param_info, declarator.into())
    } else {
        declarator
    }
}
fn parse_param_list(tokens: &mut &[TokenType]) -> Vec<(CType, Declarator)> {
    if try_consume(tokens, TokenType::Keyword("void")) {
        return Vec::new();
    }
    let mut params = Vec::new();
    loop {
        params.push(parse_param(tokens));
        if !try_consume(tokens, TokenType::Comma) {
            break;
        }
    }
    params
}
fn parse_param(tokens: &mut &[TokenType]) -> (CType, Declarator) {
    (parse_type(tokens), parse_declarator(tokens))
}

pub fn parse_declarator(tokens: &mut &[TokenType]) -> Declarator {
    if try_consume(tokens, TokenType::Star) {
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
    Pointer(Box<AbstractDeclarator>),
    Base,
}

pub fn parse_abstract_declarator(tokens: &mut &[TokenType]) -> AbstractDeclarator {
    if try_consume(tokens, TokenType::Star) {
        AbstractDeclarator::Pointer(parse_abstract_declarator(tokens).into())
    } else if try_consume(tokens, TokenType::CloseParen) {
        AbstractDeclarator::Base
    } else {
        parse_direct_abstract_declarator(tokens)
    }
}

fn parse_direct_abstract_declarator(tokens: &mut &[TokenType]) -> AbstractDeclarator {
    expect(tokens, TokenType::OpenParen);
    let result = parse_abstract_declarator(tokens);
    expect(tokens, TokenType::CloseParen);
    result
}

pub fn process_abstract_declarator(decl: AbstractDeclarator, base_type: CType) -> CType {
    match decl {
        AbstractDeclarator::Base => base_type,
        AbstractDeclarator::Pointer(inner) => {
            CType::Pointer(process_abstract_declarator(*inner, base_type).into())
        }
    }
}
