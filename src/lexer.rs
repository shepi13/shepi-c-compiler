use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum TokenType<'a> {
    Specifier(&'a str),
    Keyword(&'a str),
    Identifier(&'a str),
    Constant(&'a str),
    LongConstant(&'a str),
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
            TokenType::LongConstant(_) => TokenType::LongConstant(&data[..data.len() - 1]),
            TokenType::Specifier(_) => TokenType::Specifier(data),
            _ => token_type.clone(),
        }
    }
}

lazy_static! {
    static ref token_regexes: Vec<(TokenType<'static>, Regex)> = {
        let mut regexes: Vec<(TokenType, Regex)> = Vec::new();
        // Keywords
        regexes.push((TokenType::Specifier(""), Regex::new(&[
            r"(^int\b)",
            r"(^long\b)",
            r"(^static\b)",
            r"(^extern\b)",
        ].join("|")).unwrap()));
        regexes.push((TokenType::Keyword(""), Regex::new(&[
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
            TokenType::Identifier(""),
            Regex::new(r"^[a-zA-Z_]\w*\b").unwrap(),
        ));
        regexes.push((
            TokenType::LongConstant(""),
            Regex::new(r"^[0-9]+[lL]\b").unwrap(),
        ));
        regexes.push((
            TokenType::Constant(""),
            Regex::new(r"^[0-9]+\b").unwrap(),
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
        if let Some((result, token)) = token_regexes
            .iter()
            .find_map(|token_re| token_re.1.find(data).map(|m| (m.as_str(), &token_re.0)))
        {
            data = &data[result.len()..];
            tokens.push(TokenType::from(result, token));
        } else {
            panic!("Failed to match token '{}'", first_char);
        }
    }
    tokens
}
