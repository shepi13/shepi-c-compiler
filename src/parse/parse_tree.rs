use super::lexer::TokenType;

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
    UnsignedInt,
    UnsignedLong,
    Double,
    Function(Vec<CType>, Box<CType>),
}

impl CType {
    pub fn size(&self) -> usize {
        match self {
            CType::Int | CType::UnsignedInt => 4,
            CType::Long | CType::UnsignedLong => 8,
            CType::Double => 8,
            CType::Function(_, _) => panic!("Not a variable or constant!"),
        }
    }
    pub fn is_signed(&self) -> bool {
        match self {
            CType::UnsignedInt | CType::UnsignedLong => false,
            CType::Int | CType::Long => true,
            CType::Double => panic!("Not an integer type!"),
            CType::Function(_, _) => panic!("Not an integer type!"),
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
            Self::Int | Self::Long | Self::UnsignedInt | Self::UnsignedLong => true,
            Self::Double | Self::Function(_, _) => false,
        }
    }
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
    pub fn from(token: &TokenType) -> Self {
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
        Self { ctype: None, expr: value }
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
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Complement,
    Negate,
    LogicalNot,
    Increment(Increment),
}
#[derive(Debug, Clone)]
pub enum Increment {
    PreIncrement,
    PostIncrement,
    PreDecrement,
    PostDecrement,
}
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i64), // Limited to i32, but we'll store it as i64 for convenient conversions
    Long(i64),
    UnsignedInt(u64),
    UnsignedLong(u64),
    Double(f64),
}

impl Constant {
    pub fn int_value(&self) -> i128 {
        match self {
            Self::Int(val) | Self::Long(val) => *val as i128,
            Self::UnsignedInt(val) | Self::UnsignedLong(val) => *val as i128,
            Self::Double(val) => *val as i128,
        }
    }
}
