use crate::{
    parse::parse_tree,
    validate::ctype::{CType, Initializer},
};

pub type Program = Vec<TopLevelDecl>;
#[derive(Debug, Clone)]
pub enum TopLevelDecl {
    Function { name: String, params: Vec<Value>, instructions: Vec<Instruction>, global: bool },
    StaticDecl { identifier: String, global: bool, ctype: CType, initializer: Vec<Initializer> },
    StaticConstant { identifier: String, ctype: CType, initializer: Initializer },
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Return(Option<Value>),
    SignExtend(Value, Value),
    ZeroExtend(Value, Value),
    Truncate(Value, Value),
    DoubleToInt(Value, Value),
    DoubleToUInt(Value, Value),
    IntToDouble(Value, Value),
    UIntToDouble(Value, Value),
    UnaryOp { operator: UnaryOperator, src: Value, dst: Value },
    BinaryOp { operator: parse_tree::BinaryOperator, src1: Value, src2: Value, dst: Value },
    Copy(Value, Value),
    GetAddress(Value, Value),
    Load(Value, Value),
    Store(Value, Value),
    Label(String),
    Jump(String),
    JumpCond { jump_type: JumpType, condition: Value, target: String },
    Function(String, Vec<Value>, Option<Value>),
    AddPtr(Value, Value, u64, Value),
    CopyToOffset(Value, String, u64),
}
#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    Complement,
    Negate,
    LogicalNot,
}
impl From<parse_tree::UnaryOperator> for UnaryOperator {
    fn from(operator: parse_tree::UnaryOperator) -> Self {
        match operator {
            parse_tree::UnaryOperator::Complement => Self::Complement,
            parse_tree::UnaryOperator::LogicalNot => Self::LogicalNot,
            parse_tree::UnaryOperator::Negate => Self::Negate,
            _ => panic!("Invalid TAC operator"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum JumpType {
    JumpIfZero,
    JumpIfNotZero,
}

#[derive(Debug, Clone)]
pub enum Value {
    ConstValue(parse_tree::Constant),
    Variable(String),
}
