use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum TokenType<'a> {
    Specifier(&'a str),
    Keyword(&'a str),
    Identifier(&'a str),
    Constant(&'a str),
    ConstantLong(&'a str),
    Unsigned(&'a str),
    UnsignedLong(&'a str),
    Double(&'a str),
    // Symbols
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    SemiColon,
    // Unary Operators
    Tilde,
    Hyphen,
    Decrement,
    Increment,
    // Binary Operators
    Plus,
    Star,
    ForwardSlash,
    Percent,
    LeftShift,
    RightShift,
    Caret,
    Pipe,
    Ampersand,
    // Relational
    Exclam,
    DoubleAmpersand,
    DoublePipe,
    DoubleEqual,
    ExclamEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    // Assignment
    Equal,
    PlusEqual,
    HyphenEqual,
    StarEqual,
    ForwardSlashEqual,
    PercentEqual,
    AmpersandEqual,
    PipeEqual,
    CaretEqual,
    LeftShiftEqual,
    RightShiftEqual,
    //Conditional,
    QuestionMark,
    Colon,
    // Other
    Comma,
}

impl<'a> TokenType<'a> {
    fn from(data: &'a str, token_type: &'a TokenType) -> TokenType<'a> {
        match token_type {
            TokenType::Keyword(_) => TokenType::Keyword(data),
            TokenType::Identifier(_) => TokenType::Identifier(data),
            TokenType::Constant(_) => TokenType::Constant(data),
            TokenType::ConstantLong(_) => TokenType::ConstantLong(data),
            TokenType::Specifier(_) => TokenType::Specifier(data),
            TokenType::Unsigned(_) => TokenType::Unsigned(data),
            TokenType::UnsignedLong(_) => TokenType::UnsignedLong(data),
            TokenType::Double(_) => TokenType::Double(data),
            _ => token_type.clone(),
        }
    }
}

lazy_static! {
    static ref token_regexes: Vec<(TokenType<'static>, Regex)> = {
        let mut regexes: Vec<(TokenType, Regex)> = Vec::new();
        // Keywords
        regexes.push((TokenType::Specifier(""), Regex::new(&[
            r"^int\b",
            r"^long\b",
            r"^static\b",
            r"^extern\b",
            r"^signed\b",
            r"^unsigned\b",
            r"^double\b",
        ].join("|")).unwrap()));
        regexes.push((TokenType::Keyword(""), Regex::new(&[
            r"^void\b",
            r"^return\b",
            r"^if\b",
            r"^else\b",
            r"^goto\b",
            r"^do\b",
            r"^for\b",
            r"^while\b",
            r"^break\b",
            r"^continue\b",
            r"^switch\b",
            r"^case\b",
            r"^default\b",
        ].join("|")).unwrap()));
        // Identifiers and Constants
        regexes.push((
            TokenType::Identifier(""),
            Regex::new(r"(^[a-zA-Z_]\w*\b)").unwrap(),
        ));
        regexes.push((
            TokenType::Unsigned(""),
            Regex::new(r"^((?<val>[0-9]+)[uU])[^\w.]").unwrap(),
        ));
        regexes.push((
            TokenType::ConstantLong(""),
            Regex::new(r"^((?<val>[0-9]+)[lL])[^\w.]").unwrap(),
        ));
        regexes.push((
            TokenType::UnsignedLong(""),
            Regex::new(r"^((?<val>[0-9]+)(?:[lL][uU]|[uU][lL]))[^\w.]").unwrap(),
        ));
        regexes.push((
            TokenType::Double(""),
            Regex::new(r"^(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)[^\w.]").unwrap(),
        ));
        regexes.push((
            TokenType::Constant(""),
            Regex::new(r"^(?<val>[0-9]+)[^\w.]").unwrap(),
        ));
        // 3 Character Symbols
        regexes.push((TokenType::LeftShiftEqual, Regex::new(r"^<<=").unwrap()));
        regexes.push((TokenType::RightShiftEqual, Regex::new(r"^>>=").unwrap()));
        // 2 Character Symbols
        regexes.push((TokenType::Decrement, Regex::new(r"^--").unwrap()));
        regexes.push((TokenType::Increment, Regex::new(r"^\+\+").unwrap()));
        regexes.push((TokenType::LeftShift, Regex::new(r"^<<").unwrap()));
        regexes.push((TokenType::RightShift, Regex::new(r"^>>").unwrap()));
        regexes.push((TokenType::DoubleAmpersand, Regex::new(r"^\&\&").unwrap()));
        regexes.push((TokenType::DoubleEqual, Regex::new(r"^==").unwrap()));
        regexes.push((TokenType::DoublePipe, Regex::new(r"^\|\|").unwrap()));
        regexes.push((TokenType::ExclamEqual, Regex::new(r"^\!=").unwrap()));
        regexes.push((TokenType::GreaterThanEqual, Regex::new(r"^>=").unwrap()));
        regexes.push((TokenType::LessThanEqual, Regex::new(r"^<=").unwrap()));
        regexes.push((TokenType::PlusEqual, Regex::new(r"^\+\=").unwrap()));
        regexes.push((TokenType::HyphenEqual, Regex::new(r"^\-\=").unwrap()));
        regexes.push((TokenType::StarEqual, Regex::new(r"^\*\=").unwrap()));
        regexes.push((TokenType::ForwardSlashEqual, Regex::new(r"^\/\=").unwrap()));
        regexes.push((TokenType::PercentEqual, Regex::new(r"^\%\=").unwrap()));
        regexes.push((TokenType::AmpersandEqual, Regex::new(r"^\&\=").unwrap()));
        regexes.push((TokenType::CaretEqual, Regex::new(r"^\^\=").unwrap()));
        regexes.push((TokenType::PipeEqual, Regex::new(r"^\|\=").unwrap()));
        // Single Character Symbols
        regexes.push((TokenType::Hyphen, Regex::new(r"^-").unwrap()));
        regexes.push((TokenType::Tilde, Regex::new(r"^~").unwrap()));
        regexes.push((TokenType::SemiColon, Regex::new(r"^;").unwrap()));
        regexes.push((TokenType::Plus, Regex::new(r"^\+").unwrap()));
        regexes.push((TokenType::Star, Regex::new(r"^\*").unwrap()));
        regexes.push((TokenType::ForwardSlash, Regex::new(r"^\/").unwrap()));
        regexes.push((TokenType::Percent, Regex::new(r"^\%").unwrap()));
        regexes.push((TokenType::Pipe, Regex::new(r"^\|").unwrap()));
        regexes.push((TokenType::Caret, Regex::new(r"^\^").unwrap()));
        regexes.push((TokenType::Ampersand, Regex::new(r"^\&").unwrap()));
        regexes.push((TokenType::Exclam, Regex::new(r"^\!").unwrap()));
        regexes.push((TokenType::GreaterThan, Regex::new(r"^>").unwrap()));
        regexes.push((TokenType::LessThan, Regex::new(r"^<").unwrap()));
        regexes.push((TokenType::Equal, Regex::new(r"^=").unwrap()));
        regexes.push((TokenType::QuestionMark, Regex::new(r"^\?").unwrap()));
        regexes.push((TokenType::Colon, Regex::new(r"^\:").unwrap()));
        regexes.push((TokenType::Comma, Regex::new(r"^\,").unwrap()));
        // Brackets
        regexes.push((TokenType::OpenParen, Regex::new(r"^\(").unwrap()));
        regexes.push((TokenType::CloseParen, Regex::new(r"^\)").unwrap()));
        regexes.push((TokenType::OpenBrace, Regex::new(r"^\{").unwrap()));
        regexes.push((TokenType::CloseBrace, Regex::new(r"^\}").unwrap()));
        regexes.push((TokenType::OpenBracket, Regex::new(r"^\[").unwrap()));
        regexes.push((TokenType::CloseBracket, Regex::new(r"^\]").unwrap()));

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
        if let Some((captures, token)) = token_regexes
            .iter()
            .find_map(|token_re| token_re.1.captures(data).map(|m| (m, &token_re.0)))
        {
            // Use the first capture group instead of 0 so we can exclude characters (fake lookahead). Defaults to 0 if no match
            let fullmatch =
                captures.get(1).map(|m| m.as_str()).unwrap_or(captures.get(0).unwrap().as_str());
            // Use val capture group to capture data, defaults to full match if it doesn't exist
            let datamatch = captures.name("val").map(|m| m.as_str()).unwrap_or(fullmatch);
            // Consume data and push token
            data = &data[fullmatch.len()..];
            tokens.push(TokenType::from(datamatch, token));
        } else {
            panic!("Failed to match token '{}'", first_char);
        }
    }
    tokens
}
