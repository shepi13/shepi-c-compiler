use super::lexer::Token;

pub type Program = Vec<Declaration>;
pub type Block = Vec<BlockItem>;

#[derive(Debug, Clone, Copy)]
pub struct Location {
    pub start_loc: (usize, usize),
    pub end_loc: (usize, usize),
}
// Statements and Declarations
#[derive(Debug, Clone)]
pub enum BlockItem {
    StatementItem(Statement, Location),
    DeclareItem(Declaration),
}
#[derive(Debug, Clone)]
pub enum Statement {
    Return(TypedExpression),
    ExprStmt(TypedExpression),
    If(TypedExpression, Box<Statement>, Box<Option<Statement>>),
    While(Loop),
    DoWhile(Loop),
    For(Box<ForInit>, Loop, Option<TypedExpression>),
    Switch(Box<SwitchStatement>),
    Case(TypedExpression, Box<Statement>),
    Default(Box<Statement>),
    Break(String),
    Continue(String),
    Compound(Block),
    Label(String, Box<Statement>),
    Goto(String),
    Null,
}
#[derive(Debug, Clone)]
pub struct SwitchStatement {
    pub label: String,
    pub condition: TypedExpression,
    pub cases: Vec<(String, TypedExpression)>,
    pub statement: Box<Statement>,
    pub default: Option<String>,
}
#[derive(Debug, Clone)]
pub struct Loop {
    pub label: String,
    pub condition: TypedExpression,
    pub body: Box<Statement>,
}
#[derive(Debug, Clone)]
pub enum ForInit {
    Decl(VariableDeclaration),
    Expr(Option<TypedExpression>),
}
#[derive(Debug, Clone)]
pub enum Declaration {
    Variable(VariableDeclaration),
    Function(FunctionDeclaration),
}
#[derive(Debug, Clone)]
pub enum VariableInitializer {
    SingleElem(TypedExpression),
    CompoundInit(Vec<VariableInitializer>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CType {
    Char,
    SignedChar,
    UnsignedChar,
    Int,
    Long,
    UnsignedInt,
    UnsignedLong,
    Double,
    Function(Vec<CType>, Box<CType>),
    Pointer(Box<CType>),
    Array(Box<CType>, u64),
}

impl CType {
    pub fn size(&self) -> u64 {
        match self {
            CType::Char | CType::UnsignedChar | CType::SignedChar => 1,
            CType::Int | CType::UnsignedInt => 4,
            CType::Long | CType::UnsignedLong => 8,
            CType::Double => 8,
            CType::Function(_, _) => panic!("Not a variable or constant!"),
            CType::Pointer(_) => 8,
            CType::Array(elem_t, elem_c) => elem_c * elem_t.size(),
        }
    }
    pub fn is_signed(&self) -> bool {
        match self {
            CType::UnsignedInt | CType::UnsignedLong => false,
            CType::Int | CType::Long => true,
            _ => panic!("Not an integer type!"),
        }
    }
    pub fn is_char(&self) -> bool {
        matches!(self, Self::Char | Self::SignedChar | Self::UnsignedChar)
    }
    pub fn is_int(&self) -> bool {
        match self {
            Self::Int
            | Self::Long
            | Self::UnsignedInt
            | Self::UnsignedLong
            | Self::Char
            | Self::UnsignedChar
            | Self::SignedChar => true,
            Self::Double | Self::Function(_, _) | Self::Pointer(_) | Self::Array(_, _) => false,
        }
    }
    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_))
    }
    pub fn is_arithmetic(&self) -> bool {
        self.is_int() || *self == Self::Double
    }
}

#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub name: String,
    pub init: Option<VariableInitializer>,
    pub ctype: CType,
    pub storage: Option<StorageClass>,
    pub location: Location,
}
#[derive(Debug, Clone)]
pub struct FunctionDeclaration {
    pub name: String,
    pub ctype: CType,
    pub params: Vec<String>,
    pub body: Option<Block>,
    pub storage: Option<StorageClass>,
    pub location: Location,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
}
impl StorageClass {
    pub fn from(token: &Token) -> Self {
        match token {
            Token::Specifier("static") => Self::Static,
            Token::Specifier("extern") => Self::Extern,
            _ => panic!("Invalid storage class!"),
        }
    }
}
// Expressions
#[derive(Debug, Clone)]
pub struct TypedExpression {
    pub ctype: Option<CType>,
    pub expr: Expression,
}
impl From<Expression> for TypedExpression {
    fn from(value: Expression) -> Self {
        Self { ctype: None, expr: value }
    }
}
impl From<Expression> for Box<TypedExpression> {
    fn from(value: Expression) -> Self {
        TypedExpression { ctype: None, expr: value }.into()
    }
}
impl TypedExpression {
    pub fn is_lvalue(&self) -> bool {
        match self.expr {
            Expression::Dereference(_)
            | Expression::Subscript(_, _)
            | Expression::StringLiteral(_) => true,
            Expression::Variable(_) => !matches!(self.ctype, Some(CType::Array(_, _))),
            _ => false,
        }
    }
}
#[derive(Debug, Clone)]
pub enum Expression {
    Constant(Constant),
    StringLiteral(String),
    Variable(String),
    Unary(UnaryOperator, Box<TypedExpression>),
    Binary(Box<BinaryExpression>),
    Assignment(Box<AssignmentExpression>),
    Condition(Box<ConditionExpression>),
    FunctionCall(String, Vec<TypedExpression>),
    Cast(CType, Box<TypedExpression>),
    Dereference(Box<TypedExpression>),
    AddrOf(Box<TypedExpression>),
    Subscript(Box<TypedExpression>, Box<TypedExpression>),
}
#[derive(Debug, Clone)]
pub struct BinaryExpression {
    pub operator: BinaryOperator,
    pub left: TypedExpression,
    pub right: TypedExpression,
    pub is_assignment: bool,
}
#[derive(Debug, Clone)]
pub struct AssignmentExpression {
    pub left: TypedExpression,
    pub right: TypedExpression,
}
#[derive(Debug, Clone)]

pub struct ConditionExpression {
    pub condition: TypedExpression,
    pub if_true: TypedExpression,
    pub if_false: TypedExpression,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Complement,
    Negate,
    LogicalNot,
    Increment(IncrementType),
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IncrementType {
    PreIncrement,
    PostIncrement,
    PreDecrement,
    PostDecrement,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constant {
    Char(i64),
    Int(i64), // Limited to i32, but we'll store it as i64 for convenient conversions
    Long(i64),
    UChar(u64),
    UInt(u64),
    ULong(u64),
    Double(f64),
}

impl Constant {
    pub fn int_value(&self) -> i128 {
        match self {
            Self::Int(val) | Self::Long(val) | Self::Char(val) => *val as i128,
            Self::UInt(val) | Self::ULong(val) | Self::UChar(val) => *val as i128,
            Self::Double(val) => *val as i128,
        }
    }
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Long(_) | Self::UInt(_) | Self::ULong(_))
    }
}
