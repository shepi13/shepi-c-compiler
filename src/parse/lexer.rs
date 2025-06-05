use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum Token<'a> {
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

impl<'a> Token<'a> {
    pub fn is_type_specifier(&self) -> bool {
        use Token::Specifier;
        matches!(
            self,
            Specifier("int")
                | Specifier("long")
                | Specifier("signed")
                | Specifier("unsigned")
                | Specifier("double")
        )
    }
    fn from(data: &'a str, token_type: Token<'a>) -> Self {
        use Token::*;
        let mut token_type = token_type;
        match &mut token_type {
            Keyword(val) | Identifier(val) | Constant(val) | ConstantLong(val) | Specifier(val)
            | Unsigned(val) | UnsignedLong(val) | Double(val) => {
                *val = data;
            }
            _ => (),
        }
        token_type
    }
}

lazy_static! {
    static ref token_regexes: Vec<(Token<'static>, Regex)> = vec![
        // Type specifiers
        (Token::Specifier(""), Regex::new(&[
            r"^int\b",
            r"^long\b",
            r"^static\b",
            r"^extern\b",
            r"^signed\b",
            r"^unsigned\b",
            r"^double\b",
        ].join("|")).unwrap()),
        // Other keywords
        (Token::Keyword(""), Regex::new(&[
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
        ].join("|")).unwrap()),
        // Identifiers and Constants
        (Token::Identifier(""), Regex::new(r"(^[a-zA-Z_]\w*\b)").unwrap()),
        (Token::Constant(""), Regex::new(r"^(?<val>[0-9]+)[^\w.]").unwrap()),
        (Token::Unsigned(""), Regex::new(r"^((?<val>[0-9]+)[uU])[^\w.]").unwrap()),
        (Token::ConstantLong(""), Regex::new(r"^((?<val>[0-9]+)[lL])[^\w.]").unwrap()),
        (Token::UnsignedLong(""), Regex::new(r"^((?<val>[0-9]+)(?:[lL][uU]|[uU][lL]))[^\w.]").unwrap()),
        (
            Token::Double(""),
            Regex::new(r"^(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)[^\w.]").unwrap(),
        ),
        // 3 Character Symbols
        (Token::RightShiftEqual, Regex::new(r"^>>=").unwrap()),
        (Token::LeftShiftEqual, Regex::new(r"^<<=").unwrap()),
        // 2 Character Symbols
        (Token::Decrement, Regex::new(r"^--").unwrap()),
        (Token::Increment, Regex::new(r"^\+\+").unwrap()),
        (Token::LeftShift, Regex::new(r"^<<").unwrap()),
        (Token::RightShift, Regex::new(r"^>>").unwrap()),
        (Token::DoubleAmpersand, Regex::new(r"^\&\&").unwrap()),
        (Token::DoubleEqual, Regex::new(r"^==").unwrap()),
        (Token::DoublePipe, Regex::new(r"^\|\|").unwrap()),
        (Token::ExclamEqual, Regex::new(r"^\!=").unwrap()),
        (Token::GreaterThanEqual, Regex::new(r"^>=").unwrap()),
        (Token::LessThanEqual, Regex::new(r"^<=").unwrap()),
        (Token::PlusEqual, Regex::new(r"^\+\=").unwrap()),
        (Token::HyphenEqual, Regex::new(r"^\-\=").unwrap()),
        (Token::StarEqual, Regex::new(r"^\*\=").unwrap()),
        (Token::ForwardSlashEqual, Regex::new(r"^\/\=").unwrap()),
        (Token::PercentEqual, Regex::new(r"^\%\=").unwrap()),
        (Token::AmpersandEqual, Regex::new(r"^\&\=").unwrap()),
        (Token::CaretEqual, Regex::new(r"^\^\=").unwrap()),
        (Token::PipeEqual, Regex::new(r"^\|\=").unwrap()),
        // Single Character Symbols
        (Token::Hyphen, Regex::new(r"^-").unwrap()),
        (Token::Tilde, Regex::new(r"^~").unwrap()),
        (Token::SemiColon, Regex::new(r"^;").unwrap()),
        (Token::Plus, Regex::new(r"^\+").unwrap()),
        (Token::Star, Regex::new(r"^\*").unwrap()),
        (Token::ForwardSlash, Regex::new(r"^\/").unwrap()),
        (Token::Percent, Regex::new(r"^\%").unwrap()),
        (Token::Pipe, Regex::new(r"^\|").unwrap()),
        (Token::Caret, Regex::new(r"^\^").unwrap()),
        (Token::Ampersand, Regex::new(r"^\&").unwrap()),
        (Token::Exclam, Regex::new(r"^\!").unwrap()),
        (Token::GreaterThan, Regex::new(r"^>").unwrap()),
        (Token::LessThan, Regex::new(r"^<").unwrap()),
        (Token::Equal, Regex::new(r"^=").unwrap()),
        (Token::QuestionMark, Regex::new(r"^\?").unwrap()),
        (Token::Colon, Regex::new(r"^\:").unwrap()),
        (Token::Comma, Regex::new(r"^\,").unwrap()),
        // Brackets
        (Token::OpenParen, Regex::new(r"^\(").unwrap()),
        (Token::CloseParen, Regex::new(r"^\)").unwrap()),
        (Token::OpenBrace, Regex::new(r"^\{").unwrap()),
        (Token::CloseBrace, Regex::new(r"^\}").unwrap()),
        (Token::OpenBracket, Regex::new(r"^\[").unwrap()),
        (Token::CloseBracket, Regex::new(r"^\]").unwrap()),
    ];
}

#[derive(Debug)]
pub struct Tokens<'a> {
    tokens: Vec<Token<'a>>,
    locations: Vec<(usize, usize)>,
    source: Vec<&'a str>,
    current_token: usize,
}
impl<'a> Tokens<'a> {
    fn from(tokens: Vec<Token<'a>>, source: &'a str, token_locations: Vec<(usize, usize)>) -> Self {
        let source = source.lines().collect();
        Self {
            tokens,
            source,
            locations: Vec::new(),
            current_token: 0,
        }
    }
    pub fn current_line(&self) -> &str {
        let line_number = self.locations[self.current_token].0;
        &self.source[line_number]
    }
    pub fn peek(&self) -> Token {
        self.tokens[self.current_token]
    }
    pub fn consume(&mut self) -> Token {
        let result = self.tokens[self.current_token];
        self.current_token += 1;
        result
    }
    pub fn try_consume(&mut self, token: Token) -> bool {
        self.peek() == token && {
            self.consume();
            true
        }
    }
    pub fn expect(&mut self, token: Token) {
        let success = self.try_consume(token);
        assert!(success, "Syntax Error: Expected `{:?}`, found `{:?}`", token, self.peek())
    }
    pub fn is_empty(&self) -> bool {
        self.current_token >= self.tokens.len()
    }
    pub fn consume_type_specifiers(&mut self) -> &[Token] {
        let tokens = &self.tokens[self.current_token..];
        let position =
            tokens.iter().position(|token| !token.is_type_specifier()).unwrap_or(tokens.len());
        self.current_token += position;
        &tokens[..position]
    }
}
impl<'a> std::ops::Index<usize> for Tokens<'a> {
    type Output = Token<'a>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.tokens[self.current_token + index]
    }
}

pub fn parse(mut data: &str) -> Tokens {
    let mut tokens: Vec<Token<'_>> = Vec::new();
    let mut locations: Vec<(usize, usize)> = Vec::new();
    let source = data;
    let mut source_location = (0, 0); // (line_number, col)
    while let Some(first_char) = data.chars().next() {
        if first_char.is_whitespace() {
            data = &data[1..];
            if first_char == '\n' {
                source_location = (source_location.0 + 1, 0);
            }
        } else if let Some((captures, token)) = token_regexes
            .iter()
            .find_map(|token_re| token_re.1.captures(data).map(|m| (m, &token_re.0)))
        {
            // Use the first capture group instead of 0 so we can exclude characters (fake lookahead). Defaults to 0 if no match
            let fullmatch =
                captures.get(1).map(|m| m.as_str()).unwrap_or(captures.get(0).unwrap().as_str());
            // Use val capture group to capture data, defaults to full match if it doesn't exist
            let datamatch = captures.name("val").map(|m| m.as_str()).unwrap_or(fullmatch);
            // Consume data and push token
            locations.push(source_location);
            tokens.push(Token::from(datamatch, *token));
            data = &data[fullmatch.len()..];
            source_location = (source_location.0, source_location.1 + fullmatch.len());
        } else {
            panic!("Failed to match token '{}'", first_char);
        }
    }
    Tokens::from(tokens, source, locations)
}
