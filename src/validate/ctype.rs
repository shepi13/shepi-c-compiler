//! Module for CType type and implementations, along with symbol table and other static
//! variable initialization data structures.

use crate::parse::parse_tree::{Expression, TypedExpression};
use std::collections::HashMap;

// CType implementation

/// Supported CTypes
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
    /// Size of type in bytes
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
    /// For integers, returns whether or not the type is signed
    ///     - panics for non-int types
    pub fn is_signed(&self) -> bool {
        match self {
            CType::UnsignedInt | CType::UnsignedLong | CType::UnsignedChar => false,
            CType::Int | CType::Long | CType::Char | CType::SignedChar => true,
            _ => panic!("Not an integer type!"),
        }
    }
    /// Checks if a type is a character (single byte)
    pub fn is_char(&self) -> bool {
        matches!(self, Self::Char | Self::SignedChar | Self::UnsignedChar)
    }
    /// Checks if a type is an integer type
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
    /// Checks if a type is a pointer
    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_))
    }
    /// Checks if a type is numeric
    pub fn is_arithmetic(&self) -> bool {
        self.is_int() || *self == Self::Double
    }
}

/// Trait for types that can be lvalues
pub trait IsLValue {
    /// Should return whether or not the caller is an lvalue.
    fn is_lvalue(&self) -> bool;
}

impl IsLValue for TypedExpression {
    fn is_lvalue(&self) -> bool {
        match self.expr {
            Expression::Dereference(_)
            | Expression::Subscript(_, _)
            | Expression::StringLiteral(_) => true,
            Expression::Variable(_) => !matches!(self.ctype, Some(CType::Array(_, _))),
            _ => false,
        }
    }
}

/// Main symbol table for the compiler
#[derive(Debug, Clone)]
pub struct TypeTable {
    pub symbols: Symbols,
    pub current_function: Option<String>,
}

/// A map from symbols to their type and any stored attributes
pub type Symbols = HashMap<String, Symbol>;
#[derive(Debug, Clone)]
pub struct Symbol {
    pub ctype: CType,
    pub attrs: SymbolAttr,
}

#[derive(Debug, PartialEq, Clone)]
pub enum SymbolAttr {
    Function { defined: bool, global: bool },
    Static { init: StaticInitializer, global: bool },
    Constant(Initializer),
    Local,
}

/// Static Variable Initializers
///   - None is uninitialized
///   - Tentative is zero initialized
#[derive(Debug, PartialEq, Clone)]
pub enum StaticInitializer {
    Tentative,
    Initialized(Vec<Initializer>),
    None,
}
/// Initializers by value for all CTypes
#[derive(Debug, PartialEq, Clone)]
pub enum Initializer {
    Char(i64),
    Int(i64), // Limited to i32, but we store as i64 for matching and do our own conversion
    Long(i64),
    UChar(u64),
    UInt(u64),
    ULong(u64),
    Double(f64),
    ZeroInit(u64),
    StringInit { data: String, null_terminated: bool },
    PointerInit(String),
}
impl Initializer {
    pub fn int_value(&self) -> i128 {
        match self {
            Self::Int(val) | Self::Long(val) | Self::Char(val) => *val as i128,
            Self::UInt(val) | Self::ULong(val) | Self::UChar(val) => *val as i128,
            Self::Double(val) => *val as i128,
            Self::ZeroInit(_) => 0,
            Self::StringInit { data: _, null_terminated: _ } | Self::PointerInit(_) => {
                panic!("Non-numerical initializer!")
            }
        }
    }
}
