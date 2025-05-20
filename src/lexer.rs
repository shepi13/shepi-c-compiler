use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum TokenType<'a> {
    SPECIFIER(&'a str),
    KEYWORD(&'a str),
    IDENTIFIER(&'a str),
    CONSTANT(&'a str),
    // Symbols
    OPENPAREN,
    CLOSEPAREN,
    OPENBRACE,
    CLOSEBRACE,
    OPENBRACKET,
    CLOSEBRACKET,
    SEMICOLON,
    // Unary Operators
    TILDE,
    HYPHEN,
    DECREMENT,
    INCREMENT,
    // Binary Operators
    PLUS,
    STAR,
    FORWARDSLASH,
    PERCENT,
    LEFTSHIFT,
    RIGHTSHIFT,
    CARET,
    PIPE,
    AMPERSAND,
    // Relational
    EXCLAM,
    DOUBLEAMPERSAND,
    DOUBLEPIPE,
    DOUBLEEQUAL,
    NOTEQUAL,
    LESSTHAN,
    GREATERTHAN,
    LESSTHANEQUAL,
    GREATERTHANEQUAL,
    // Assignment
    EQUAL,
    PLUSEQUAL,
    HYPHENEQUAL,
    STAREQUAL,
    FORWARDSLASHEQUAL,
    PERCENTEQUAL,
    AMPERSANDEQUAL,
    PIPEEQUAL,
    CARETEQUAL,
    LEFTSHIFTEQUAL,
    RIGHTSHIFTEQUAL,
    //Conditional,
    QUESTIONMARK,
    COLON,
    // Other
    COMMA,
}

impl<'a> TokenType<'a> {
    fn from(data: &'a str, token_type: &'a TokenType) -> TokenType<'a> {
        match token_type {
            TokenType::KEYWORD(_) => TokenType::KEYWORD(data),
            TokenType::IDENTIFIER(_) => TokenType::IDENTIFIER(data),
            TokenType::CONSTANT(_) => TokenType::CONSTANT(data),
            TokenType::SPECIFIER(_) => TokenType::SPECIFIER(data),
            _ => token_type.clone(),
        }
    }
}

lazy_static! {
    static ref token_regexes: Vec<(TokenType<'static>, Regex)> = {
        let mut regexes: Vec<(TokenType, Regex)> = Vec::new();
        // Keywords
        regexes.push((TokenType::SPECIFIER(""), Regex::new(&[
            r"(^int\b)",
            r"(^static\b)",
            r"(^extern\b)",
        ].join("|")).unwrap()));
        regexes.push((TokenType::KEYWORD(""), Regex::new(&[
            r"(^void\b)",
            r"(^return\b)",
            r"(^if\b)",
            r"(^else\b)",
            r"(^goto\b)",
            r"(^do\b)",
            r"(^for\b)",
            r"(^while\b)",
            r"(^break\b)",
            r"(^continue\b)",
            r"(^switch\b)",
            r"(^case\b)",
            r"(^default\b)",
        ].join("|")).unwrap()));
        // Identifiers and Constants
        regexes.push((
            TokenType::IDENTIFIER(""),
            Regex::new(r"^[a-zA-Z_]\w*\b").unwrap(),
        ));
        regexes.push((
            TokenType::CONSTANT(""),
            Regex::new(r"^[0-9]+\b").unwrap(),
        ));
        // 3 Character Symbols
        regexes.push((TokenType::LEFTSHIFTEQUAL, Regex::new(r"^<<=").unwrap()));
        regexes.push((TokenType::RIGHTSHIFTEQUAL, Regex::new(r"^>>=").unwrap()));
        // 2 Character Symbols
        regexes.push((TokenType::DECREMENT, Regex::new(r"^--").unwrap()));
        regexes.push((TokenType::INCREMENT, Regex::new(r"^\+\+").unwrap()));
        regexes.push((TokenType::LEFTSHIFT, Regex::new(r"^<<").unwrap()));
        regexes.push((TokenType::RIGHTSHIFT, Regex::new(r"^>>").unwrap()));
        regexes.push((TokenType::DOUBLEAMPERSAND, Regex::new(r"^\&\&").unwrap()));
        regexes.push((TokenType::DOUBLEEQUAL, Regex::new(r"^==").unwrap()));
        regexes.push((TokenType::DOUBLEPIPE, Regex::new(r"^\|\|").unwrap()));
        regexes.push((TokenType::NOTEQUAL, Regex::new(r"^\!=").unwrap()));
        regexes.push((TokenType::GREATERTHANEQUAL, Regex::new(r"^>=").unwrap()));
        regexes.push((TokenType::LESSTHANEQUAL, Regex::new(r"^<=").unwrap()));
        regexes.push((TokenType::PLUSEQUAL, Regex::new(r"^\+\=").unwrap()));
        regexes.push((TokenType::HYPHENEQUAL, Regex::new(r"^\-\=").unwrap()));
        regexes.push((TokenType::STAREQUAL, Regex::new(r"^\*\=").unwrap()));
        regexes.push((TokenType::FORWARDSLASHEQUAL, Regex::new(r"^\/\=").unwrap()));
        regexes.push((TokenType::PERCENTEQUAL, Regex::new(r"^\%\=").unwrap()));
        regexes.push((TokenType::AMPERSANDEQUAL, Regex::new(r"^\&\=").unwrap()));
        regexes.push((TokenType::CARETEQUAL, Regex::new(r"^\^\=").unwrap()));
        regexes.push((TokenType::PIPEEQUAL, Regex::new(r"^\|\=").unwrap()));
        // Single Character Symbols
        regexes.push((TokenType::HYPHEN, Regex::new(r"^-").unwrap()));
        regexes.push((TokenType::TILDE, Regex::new(r"^~").unwrap()));
        regexes.push((TokenType::SEMICOLON, Regex::new(r"^;").unwrap()));
        regexes.push((TokenType::PLUS, Regex::new(r"^\+").unwrap()));
        regexes.push((TokenType::STAR, Regex::new(r"^\*").unwrap()));
        regexes.push((TokenType::FORWARDSLASH, Regex::new(r"^\/").unwrap()));
        regexes.push((TokenType::PERCENT, Regex::new(r"^\%").unwrap()));
        regexes.push((TokenType::PIPE, Regex::new(r"^\|").unwrap()));
        regexes.push((TokenType::CARET, Regex::new(r"^\^").unwrap()));
        regexes.push((TokenType::AMPERSAND, Regex::new(r"^\&").unwrap()));
        regexes.push((TokenType::EXCLAM, Regex::new(r"^\!").unwrap()));
        regexes.push((TokenType::GREATERTHAN, Regex::new(r"^>").unwrap()));
        regexes.push((TokenType::LESSTHAN, Regex::new(r"^<").unwrap()));
        regexes.push((TokenType::EQUAL, Regex::new(r"^=").unwrap()));
        regexes.push((TokenType::QUESTIONMARK, Regex::new(r"^\?").unwrap()));
        regexes.push((TokenType::COLON, Regex::new(r"^\:").unwrap()));
        regexes.push((TokenType::COMMA, Regex::new(r"^\,").unwrap()));
        // Brackets
        regexes.push((TokenType::OPENPAREN, Regex::new(r"^\(").unwrap()));
        regexes.push((TokenType::CLOSEPAREN, Regex::new(r"^\)").unwrap()));
        regexes.push((TokenType::OPENBRACE, Regex::new(r"^\{").unwrap()));
        regexes.push((TokenType::CLOSEBRACE, Regex::new(r"^\}").unwrap()));
        regexes.push((TokenType::OPENBRACKET, Regex::new(r"^\[").unwrap()));
        regexes.push((TokenType::CLOSEBRACKET, Regex::new(r"^\]").unwrap()));

        regexes
    };
}

pub fn parse<'a>(mut data: &'a str) -> Vec<TokenType<'a>> {
    let mut tokens: Vec<TokenType> = Vec::new();
    while let Some(first_char) = data.chars().nth(0) {
        if first_char.is_whitespace() {
            data = &data[1..];
            continue;
        }
        let mut success = false;
        for (token, regex) in token_regexes.iter() {
            if let Some(found) = regex.find(data) {
                let result = found.as_str();
                data = &data[result.len()..];
                tokens.push(TokenType::from(result, token));
                success = true;
                break;
            }
        }
        if !success {
            panic!("Failed to match token '{}'", first_char);
        }
    }
    tokens
}
