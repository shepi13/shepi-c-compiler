//! Module for CType type and implementations, along with symbol table and other static
//! variable initialization data structures.

use crate::{
    helpers::error::Error,
    parse::parse_tree::{Expression, TypedExpression},
    validate::type_check::eval_constant_expr,
};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

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
    VarArgs,
}

impl CType {
    /// Size of type in bytes
    pub fn size(&self) -> u64 {
        match self {
            CType::Char | CType::UnsignedChar | CType::SignedChar => 1,
            CType::Int | CType::UnsignedInt => 4,
            CType::Long | CType::UnsignedLong => 8,
            CType::Double => 8,
            CType::Function(_, _) | CType::VarArgs => panic!("Not a variable or constant!"),
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
            Self::Double
            | Self::Function(_, _)
            | Self::Pointer(_)
            | Self::Array(_, _)
            | Self::VarArgs => false,
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

/// Generate a unique name for a constant string symbol table entry
pub fn string_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("string.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
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

type Result<T> = std::result::Result<T, Error>;

// Typed expressions

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

impl Expression {
    pub fn set_type(self, ctype: &CType) -> Result<TypedExpression> {
        let typed_expr = TypedExpression::from(self);
        typed_expr.set_type(ctype)
    }
}

impl TypedExpression {
    pub fn set_type(mut self, ctype: &CType) -> Result<Self> {
        self.ctype = Some(ctype.clone());
        Ok(self)
    }
    pub fn get_type(&self) -> CType {
        self.ctype.clone().expect("Undefined type!")
    }
    pub fn is_null_ptr(&self) -> Result<bool> {
        let result = match eval_constant_expr(self, self.ctype.as_ref().expect("Has a type")) {
            Ok(val) => val.is_integer() && val.int_value() == 0,
            Err(_) => false,
        };
        Ok(result)
    }
    pub fn convert_to(self, ctype: &CType) -> Result<Self> {
        if self.get_type() == *ctype {
            Ok(self)
        } else {
            Expression::Cast(ctype.clone(), self.into()).set_type(ctype)
        }
    }
    pub fn promote_char(self) -> Result<Self> {
        if self.get_type().is_char() { self.convert_to(&CType::Int) } else { Ok(self) }
    }
    pub fn convert_by_assignment(self, ctype: &CType) -> Result<Self> {
        if self.get_type() == *ctype {
            Ok(self)
        } else if (self.get_type().is_arithmetic() && ctype.is_arithmetic())
            || (self.is_null_ptr()? && ctype.is_pointer())
        {
            self.convert_to(ctype)
        } else {
            Err(Error::new("Invalid types", "Failed to convert by assignment!"))
        }
    }
}
pub fn get_common_pointer_type(left: &TypedExpression, right: &TypedExpression) -> Result<CType> {
    let left_type = left.get_type();
    let right_type = right.get_type();
    if left_type == right_type || right.is_null_ptr()? {
        Ok(left_type)
    } else if left.is_null_ptr()? {
        Ok(right_type)
    } else {
        Err(Error::new("Invalid types", "Pointers have incompatible types!"))
    }
}
pub fn get_common_type(left: &TypedExpression, right: &TypedExpression) -> Result<CType> {
    let left_type = left.get_type();
    let right_type = right.get_type();
    // Promote character types to int
    let left_type = if left_type.is_char() { CType::Int } else { left_type };
    let right_type = if right_type.is_char() { CType::Int } else { right_type };
    if left_type.is_pointer() || right_type.is_pointer() {
        get_common_pointer_type(left, right)
    } else if left_type == CType::Double || right_type == CType::Double {
        Ok(CType::Double)
    } else if left_type == right_type {
        Ok(left_type)
    } else if left_type.size() == right_type.size() {
        if left_type.is_signed() { Ok(right_type) } else { Ok(left_type) }
    } else if left_type.size() > right_type.size() {
        Ok(left_type)
    } else {
        Ok(right_type)
    }
}
