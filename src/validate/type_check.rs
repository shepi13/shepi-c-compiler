use std::{
    collections::{HashMap, HashSet},
    iter::zip,
};

use crate::parse::parse_tree::{
    AssignmentExpression, BinaryExpression, BinaryOperator, Block, BlockItem, CType,
    ConditionExpression, Constant, Declaration, Expression, ForInit, FunctionDeclaration, Program,
    Statement, StorageClass, TypedExpression, UnaryOperator, VariableDeclaration,
    VariableInitializer,
};

#[derive(Debug, Clone)]
struct TypeTable {
    symbols: Symbols,
    current_function: Option<String>,
}

pub type Symbols = HashMap<String, Symbol>;
#[derive(Debug, Clone)]
pub struct Symbol {
    pub ctype: CType,
    pub attrs: SymbolAttr,
}

impl Symbol {
    pub fn get_function_attrs(&self) -> &FunctionAttributes {
        match &self.attrs {
            SymbolAttr::Function(attrs) => attrs,
            _ => panic!("Not a function!"),
        }
    }
    pub fn get_static_attrs(&self) -> &StaticAttributes {
        match &self.attrs {
            SymbolAttr::Static(attrs) => attrs,
            _ => panic!("Not a static variable: {:#?}", self),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum SymbolAttr {
    Function(FunctionAttributes),
    Static(StaticAttributes),
    Local,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionAttributes {
    pub defined: bool,
    pub global: bool,
}
#[derive(Debug, PartialEq, Clone)]
pub struct StaticAttributes {
    pub init: StaticInitializer,
    pub global: bool,
}
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StaticInitializer {
    Tentative,
    Initialized(Initializer),
    None,
}
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Initializer {
    Int(i64), // Limited to i32, but we store as i64 for matching and do our own conversion
    Long(i64),
    UnsignedInt(u64),
    UnsignedLong(u64),
    Double(f64),
}
impl Initializer {
    pub fn int_value(&self) -> i128 {
        match self {
            Self::Int(val) | Self::Long(val) => *val as i128,
            Self::UnsignedInt(val) | Self::UnsignedLong(val) => *val as i128,
            Self::Double(val) => *val as i128,
        }
    }
}

#[derive(Debug)]
pub struct TypedProgram {
    pub program: Program,
    pub symbols: Symbols,
}

pub fn eval_constant_expr(expr: &TypedExpression, ctype: &CType) -> Result<Constant, ()> {
    match &expr.expr {
        Expression::Constant(constant) => eval_constant(constant, ctype),
        Expression::Unary(UnaryOperator::Negate, constexpr) => eval_constant_expr(constexpr, ctype),
        _ => Err(()),
    }
}

fn eval_constant(constant: &Constant, ctype: &CType) -> Result<Constant, ()> {
    match ctype {
        CType::Int => Ok(Constant::Int((constant.int_value() & 0xFFFFFFFF) as i64)),
        CType::Long => Ok(Constant::Long((constant.int_value() & 0xFFFFFFFFFFFFFFFF) as i64)),
        CType::UnsignedInt => Ok(Constant::UnsignedInt((constant.int_value() as u64) & 0xFFFFFFFF)),
        CType::UnsignedLong => Ok(Constant::UnsignedLong(constant.int_value() as u64)),
        CType::Double => match constant {
            Constant::Double(val) => Ok(Constant::Double(*val)),
            _ => Ok(Constant::Double(constant.int_value() as f64)),
        },
        CType::Pointer(ctype) => match eval_constant(constant, ctype) {
            Ok(constant) if constant.int_value() == 0 => Ok(constant),
            _ => Err(()),
        },
        _ => Err(()),
    }
}

// Set/Get Type

fn set_type(mut expression: TypedExpression, ctype: &CType) -> TypedExpression {
    expression.ctype = Some(ctype.clone());
    expression
}
pub fn get_type(expression: &TypedExpression) -> CType {
    expression.ctype.clone().expect("Undefined type!")
}
pub fn is_null_ptr(expr: &TypedExpression) -> bool {
    let val = eval_constant_expr(expr, expr.ctype.as_ref().expect("Has a type"));
    match val {
        Ok(Constant::Double(_)) | Err(()) => false,
        _ => val.expect("constexpr").int_value() == 0,
    }
}
pub fn get_common_pointer_type(left: &TypedExpression, right: &TypedExpression) -> CType {
    let left_type = get_type(left);
    let right_type = get_type(right);
    if left_type == right_type || is_null_ptr(right) {
        left_type
    } else if is_null_ptr(left) {
        right_type
    } else {
        panic!("Expressions have incompatible types!")
    }
}
pub fn get_common_type(left: &TypedExpression, right: &TypedExpression) -> CType {
    let left_type = get_type(left);
    let right_type = get_type(right);
    if left_type.is_pointer() || right_type.is_pointer() {
        get_common_pointer_type(left, right)
    } else if left_type == CType::Double || right_type == CType::Double {
        CType::Double
    } else if left_type == right_type {
        left_type
    } else if left_type.size() == right_type.size() {
        if left_type.is_signed() { right_type } else { left_type }
    } else if left_type.size() > right_type.size() {
        left_type
    } else {
        right_type
    }
}
fn convert_to(expression: TypedExpression, ctype: &CType) -> TypedExpression {
    if get_type(&expression) == *ctype {
        expression
    } else {
        set_type(Expression::Cast(ctype.clone(), expression.into()).into(), ctype)
    }
}
fn convert_by_assignment(expr: TypedExpression, ctype: &CType) -> TypedExpression {
    if get_type(&expr) == *ctype {
        expr
    } else if (get_type(&expr).is_arithmetic() && ctype.is_arithmetic())
        || (is_null_ptr(&expr) && ctype.is_pointer())
    {
        convert_to(expr, ctype)
    } else {
        panic!("Cannot convert type for assignment!")
    }
}

// Tree traversal functions
pub fn type_check_program(program: Program) -> TypedProgram {
    let mut table = TypeTable {
        symbols: HashMap::new(),
        current_function: None,
    };
    TypedProgram {
        program: program
            .into_iter()
            .map(|decl| type_check_declaration(decl, &mut table, true))
            .collect(),
        symbols: table.symbols,
    }
}

fn type_check_declaration(decl: Declaration, table: &mut TypeTable, global: bool) -> Declaration {
    match decl {
        Declaration::Function(func) => Declaration::Function(type_check_function(func, table)),
        Declaration::Variable(mut var) => {
            if global {
                type_check_var_filescope(&var, table);
            } else {
                var = type_check_var_declaration(var, table);
            }
            Declaration::Variable(var)
        }
    }
}

fn type_check_block(block: Block, table: &mut TypeTable) -> Block {
    block.into_iter().map(|item| type_check_block_item(item, table)).collect()
}
fn type_check_block_item(block_item: BlockItem, table: &mut TypeTable) -> BlockItem {
    match block_item {
        BlockItem::StatementItem(statement) => {
            BlockItem::StatementItem(type_check_statement(statement, table))
        }
        BlockItem::DeclareItem(decl) => {
            BlockItem::DeclareItem(type_check_declaration(decl, table, false))
        }
    }
}
fn type_check_statement(statement: Statement, table: &mut TypeTable) -> Statement {
    use Statement::*;
    match statement {
        Return(expr) => {
            let expr = type_check_and_convert(expr, table);
            let cur_func =
                table.current_function.as_ref().expect("Return must be inside function!");
            let cur_func_type = &table.symbols[cur_func].ctype;
            if let CType::Function(_, return_type) = cur_func_type {
                let expr = convert_by_assignment(expr, return_type);
                Return(expr)
            } else {
                panic!("Failed to match function type!")
            }
        }
        ExprStmt(expr) => ExprStmt(type_check_and_convert(expr, table)),
        Label(name, statement) => Label(name, type_check_statement(*statement, table).into()),
        If(expr, if_true, if_false) => {
            let expr = type_check_and_convert(expr, table);
            let if_true = type_check_statement(*if_true, table);
            let if_false = if_false.map(|statement| type_check_statement(statement, table));
            If(expr, if_true.into(), if_false.into())
        }
        While(mut loop_data) => {
            loop_data.condition = type_check_and_convert(loop_data.condition, table);
            loop_data.body = type_check_statement(*loop_data.body, table).into();
            While(loop_data)
        }
        DoWhile(mut loop_data) => {
            loop_data.condition = type_check_and_convert(loop_data.condition, table);
            loop_data.body = type_check_statement(*loop_data.body, table).into();
            DoWhile(loop_data)
        }
        Compound(block) => Compound(type_check_block(block, table)),
        Switch(mut switch) => {
            switch.condition = type_check_and_convert(switch.condition, table);
            let cond_type = get_type(&switch.condition);
            let mut new_cases = Vec::new();
            let mut case_vals = HashSet::new();
            for case in switch.cases {
                let constexpr =
                    eval_constant_expr(&case.1, &cond_type).expect("Must be constexpr!");
                let case_typed = type_check_and_convert(case.1, table);
                assert!(get_type(&case_typed).is_int(), "Switch case must be integer type!");
                assert!(cond_type.is_int(), "Switch condition must be integer type!");
                assert!(!case_vals.contains(&constexpr.int_value()), "Duplicate case!");
                case_vals.insert(constexpr.int_value());
                new_cases.push((case.0, Expression::Constant(constexpr).into()));
            }
            switch.cases = new_cases;
            switch.statement = type_check_statement(*switch.statement, table).into();
            Switch(switch)
        }
        For(init, mut loop_data, post) => {
            let init = match init {
                ForInit::Decl(decl) => ForInit::Decl(type_check_var_declaration(decl, table)),
                ForInit::Expr(Some(expr)) => {
                    ForInit::Expr(Some(type_check_and_convert(expr, table)))
                }
                _ => ForInit::Expr(None),
            };
            loop_data.condition = type_check_and_convert(loop_data.condition, table);
            let post = post.map(|post_expr| type_check_and_convert(post_expr, table));
            loop_data.body = type_check_statement(*loop_data.body, table).into();
            For(init, loop_data, post)
        }
        Case(_, _) | Default(_) => {
            panic!("Should be re-written into label in switch resolve pass!")
        }
        Break(_) | Continue(_) | Goto(_) | Null => statement,
    }
}

fn type_check_and_convert(expr: TypedExpression, table: &mut TypeTable) -> TypedExpression {
    let typed_expr = type_check_expression(expr, table);
    match get_type(&typed_expr) {
        CType::Array(elem_t, _) => {
            let addr_expr = Expression::AddrOf(typed_expr.into());
            set_type(addr_expr.into(), &CType::Pointer(elem_t))
        }
        _ => typed_expr,
    }
}

fn type_check_expression(expr: TypedExpression, table: &mut TypeTable) -> TypedExpression {
    match expr.expr {
        Expression::Variable(name) => {
            let var_type = &table.symbols[&name].ctype;
            assert!(!matches!(var_type, CType::Function(_, _)), "Function name used as variable!");
            set_type(Expression::Variable(name).into(), var_type)
        }
        Expression::Constant(constant) => match constant {
            Constant::Int(_) => set_type(Expression::Constant(constant).into(), &CType::Int),
            Constant::Long(_) => set_type(Expression::Constant(constant).into(), &CType::Long),
            Constant::UnsignedInt(_) => {
                set_type(Expression::Constant(constant).into(), &CType::UnsignedInt)
            }
            Constant::UnsignedLong(_) => {
                set_type(Expression::Constant(constant).into(), &CType::UnsignedLong)
            }
            Constant::Double(_) => set_type(Expression::Constant(constant).into(), &CType::Double),
        },
        Expression::Cast(new_type, inner) => {
            let typed_inner = type_check_and_convert(*inner, table);
            let old_type = get_type(&typed_inner);
            if new_type == CType::Double && old_type.is_pointer()
                || old_type == CType::Double && new_type.is_pointer()
            {
                panic!("Cannot convert double to pointer or pointer to double!");
            }
            assert!(!matches!(new_type, CType::Array(_, _)), "Cannot cast to array type");
            let cast_expr = Expression::Cast(new_type.clone(), typed_inner.into());
            set_type(cast_expr.into(), &new_type)
        }
        Expression::Unary(op, inner) => {
            let typed_inner = type_check_and_convert(*inner, table);
            let is_lvalue = typed_inner.is_lvalue();
            let new_type = get_type(&typed_inner);
            let unary_expr = Expression::Unary(op.clone(), typed_inner.into()).into();
            match op {
                UnaryOperator::LogicalNot => return set_type(unary_expr, &CType::Int),
                UnaryOperator::Complement => {
                    assert!(new_type.is_int(), "Invalid operand for operator `~`")
                }
                UnaryOperator::Negate => {
                    assert!(!new_type.is_pointer(), "Pointer is invalid operand for operator `-`")
                }
                UnaryOperator::Increment(_) => {
                    assert!(is_lvalue, "Increment target must be lvalue!");
                }
            }
            set_type(unary_expr, &new_type)
        }
        Expression::Binary(binary) => type_check_binary_expr(*binary, table),
        Expression::Assignment(assign) => {
            let left = type_check_and_convert(assign.left, table);
            let right = type_check_and_convert(assign.right, table);
            assert!(left.is_lvalue(), "Assignment target must be lvalue!");
            let left_type = &get_type(&left);
            let right = convert_by_assignment(right, left_type);
            let assign = Expression::Assignment(AssignmentExpression { left, right }.into());
            set_type(assign.into(), left_type)
        }
        Expression::Condition(cond) => {
            let condition = type_check_and_convert(cond.condition, table);
            let if_true = type_check_and_convert(cond.if_true, table);
            let if_false = type_check_and_convert(cond.if_false, table);
            let common_type = get_common_type(&if_true, &if_false);
            let if_true = convert_to(if_true, &common_type);
            let if_false = convert_to(if_false, &common_type);
            let condition_expr =
                Expression::Condition(ConditionExpression { condition, if_true, if_false }.into());
            set_type(condition_expr.into(), &common_type)
        }
        Expression::FunctionCall(name, args) => {
            let expected_type = table.symbols[&name].ctype.clone();
            match expected_type {
                CType::Function(arg_types, ret_type) => {
                    assert!(arg_types.len() == args.len(), "Incorrect number of arguments!");
                    let mut converted_args = Vec::new();
                    for (arg, arg_type) in zip(args, arg_types) {
                        let typed_arg = type_check_and_convert(arg, table);
                        converted_args.push(convert_by_assignment(typed_arg, &arg_type));
                    }
                    let call_expr = Expression::FunctionCall(name, converted_args);
                    set_type(call_expr.into(), &ret_type)
                }
                _ => panic!("Variable used as function!"),
            }
        }
        Expression::Dereference(inner) => {
            let typed_inner = type_check_and_convert(*inner, table);
            match get_type(&typed_inner) {
                CType::Pointer(reference_t) => {
                    let deref_expr = Expression::Dereference(typed_inner.into());
                    set_type(deref_expr.into(), &reference_t)
                }
                _ => panic!("Invalid dreference!"),
            }
        }
        Expression::AddrOf(inner) => {
            assert!(inner.is_lvalue(), "Invalid operand to unary `&`, not lvalue!");
            let typed_inner = type_check_expression(*inner, table);
            let reference_t = get_type(&typed_inner).into();
            let addr_expr = Expression::AddrOf(typed_inner.into());
            set_type(addr_expr.into(), &CType::Pointer(reference_t))
        }
        Expression::Subscript(left, right) => {
            let mut left = type_check_and_convert(*left, table);
            let mut right = type_check_and_convert(*right, table);
            let left_t = get_type(&left);
            let right_t = get_type(&right);
            if let CType::Pointer(ptr_t) = &left_t {
                assert!(right_t.is_int(), "Subscript must have integer and pointer operands");
                right = convert_to(right, &CType::Long);
                set_type(Expression::Subscript(left.into(), right.into()).into(), ptr_t)
            } else if let CType::Pointer(ptr_t) = &right_t {
                assert!(left_t.is_int(), "Subscript must have integer and pointer operands");
                left = convert_to(left, &CType::Long);
                set_type(Expression::Subscript(left.into(), right.into()).into(), ptr_t)
            } else {
                panic!("Subscript must have integer and pointer operands!")
            }
        }
    }
}

fn type_check_binary_expr(binary: BinaryExpression, table: &mut TypeTable) -> TypedExpression {
    use BinaryOperator::*;
    use Expression::Binary;
    let left = type_check_and_convert(binary.left, table);
    let right = type_check_and_convert(binary.right, table);
    match binary.operator {
        LogicalAnd | LogicalOr => {
            let binexpr = BinaryExpression { left, right, ..binary };
            set_type(Binary(binexpr.into()).into(), &CType::Int)
        }
        LessThan | GreaterThan | LessThanEqual | GreaterThanEqual | IsEqual | NotEqual => {
            let common_type = get_common_type(&left, &right);
            if common_type.is_pointer() && !matches!(binary.operator, IsEqual | NotEqual) {
                assert!(
                    !is_null_ptr(&left) && !is_null_ptr(&right),
                    "Cannot compare null with relational operators in C"
                );
                assert!(get_type(&left) == get_type(&right), "Pointer comparison type mismatch");
            }
            let left = convert_to(left, &common_type);
            let right = convert_to(right, &common_type);
            let binexpr = BinaryExpression { left, right, ..binary };
            set_type(Binary(binexpr.into()).into(), &CType::Int)
        }
        LeftShift | RightShift => {
            let common_type = get_common_type(&left, &right);
            let result_t = get_type(&left);
            assert!(common_type.is_int(), "Operands for bitshift must be integer types!");
            let binexpr = BinaryExpression { left, right, ..binary };
            set_type(Binary(binexpr.into()).into(), &result_t)
        }
        Multiply | Divide => {
            let common_type = get_common_type(&left, &right);
            assert!(!common_type.is_pointer(), "Cannot multiply/divide pointers!");
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        Remainder => {
            let common_type = get_common_type(&left, &right);
            assert!(common_type.is_int(), "Operands for remainder must be integer types!");
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        BitAnd | BitOr | BitXor => {
            let common_type = get_common_type(&left, &right);
            assert!(common_type.is_int(), "Operands for bit operators must be integer types!");
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        Add => {
            let left_t = get_type(&left);
            let right_t = get_type(&right);
            if left_t.is_arithmetic() && right_t.is_arithmetic() {
                type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
            } else if left_t.is_pointer() && right_t.is_int() {
                assert!(!binary.is_assignment || left.is_lvalue(), "Can only assign to lvalue!");
                let right = convert_to(right, &CType::Long);
                let binexpr = BinaryExpression { left, right, ..binary };
                set_type(Binary(binexpr.into()).into(), &left_t)
            } else if right_t.is_pointer() && left_t.is_int() {
                assert!(!binary.is_assignment, "Adding int to pointer gives pointer, not int");
                let left = convert_to(left, &CType::Long);
                let binexpr = BinaryExpression { left, right, ..binary };
                set_type(Binary(binexpr.into()).into(), &right_t)
            } else {
                panic!("Invalid operands for addition")
            }
        }
        Subtract => {
            let left_t = get_type(&left);
            let right_t = get_type(&right);
            if left_t.is_arithmetic() && right_t.is_arithmetic() {
                type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
            } else if left_t.is_pointer() && right_t.is_int() {
                let right = convert_to(right, &CType::Long);
                let binexpr = BinaryExpression { left, right, ..binary };
                set_type(Binary(binexpr.into()).into(), &left_t)
            } else if left_t.is_pointer() && left_t == right_t {
                let binexpr = BinaryExpression { left, right, ..binary };
                set_type(Binary(binexpr.into()).into(), &CType::Long)
            } else {
                panic!("Invalid operands for subtraction")
            }
        }
    }
}

fn type_check_arithmetic_binexpr(
    left: TypedExpression,
    right: TypedExpression,
    operator: BinaryOperator,
    is_assignment: bool,
) -> TypedExpression {
    let common_type = get_common_type(&left, &right);
    let binexpr = if is_assignment {
        assert!(left.is_lvalue(), "Can only assign to lvalue!");
        let right = convert_to(right, &common_type);
        BinaryExpression { left, right, operator, is_assignment }
    } else {
        let left = convert_to(left, &common_type);
        let right = convert_to(right, &common_type);
        BinaryExpression { left, right, operator, is_assignment }
    };
    set_type(Expression::Binary(binexpr.into()).into(), &common_type)
}

fn type_check_function(
    function: FunctionDeclaration,
    table: &mut TypeTable,
) -> FunctionDeclaration {
    table.current_function = Some(function.name.clone());
    let function = type_check_array_params(function);
    let ctype = &function.ctype;
    let mut defined = function.body.is_some();
    let mut global = function.storage != Some(StorageClass::Static);
    if let Some(symbol) = table.symbols.get(&function.name) {
        let attrs = symbol.get_function_attrs();
        assert!(symbol.ctype == *ctype, "Incompatible function declarations!");
        assert!(!attrs.defined || !defined, "Function redefinition!");
        assert!(!attrs.global || global, "static definition cannot follow non-static");
        global = attrs.global;
        defined = attrs.defined;
    }
    let attrs = SymbolAttr::Function(FunctionAttributes { defined, global });
    table.symbols.insert(function.name.clone(), Symbol { ctype: ctype.clone(), attrs });
    if let Some(body) = function.body {
        let CType::Function(param_types, _) = ctype else { panic!("Not a function!") };
        for (param, param_type) in zip(&function.params, param_types) {
            let attrs = SymbolAttr::Local;
            table.symbols.insert(param.clone(), Symbol { ctype: param_type.clone(), attrs });
        }
        let body = Some(type_check_block(body, table));
        FunctionDeclaration { body, ..function }
    } else {
        function
    }
}

fn type_check_array_params(mut function: FunctionDeclaration) -> FunctionDeclaration {
    let CType::Function(param_types, return_t) = function.ctype else { panic!("Not a function!") };
    assert!(!matches!(*return_t, CType::Array(_, _)), "Function cannot return array");
    let adjusted_params = param_types
        .into_iter()
        .map(|param_t| match param_t {
            CType::Array(elem_t, _) => CType::Pointer(elem_t),
            _ => param_t,
        })
        .collect();
    function.ctype = CType::Function(adjusted_params, return_t);
    function
}

fn parse_static_initializer(
    init: &Option<VariableInitializer>,
    ctype: &CType,
) -> StaticInitializer {
    let init = init.as_ref().map(VariableInitializer::get_single_init_ref);
    let init_val = init.as_ref().map(|expr| eval_constant_expr(expr, ctype));
    let init_val = init_val.unwrap_or(Ok(Constant::Int(0))).expect("Must be constexpr!");
    use Constant::*;
    match init_val {
        Int(val) | Long(val) => gen_initializer(val as i128, ctype),
        UnsignedInt(val) | UnsignedLong(val) => gen_initializer(val as i128, ctype),
        Double(val) => gen_initializer(val, ctype),
    }
}

fn gen_initializer<T>(val: T, ctype: &CType) -> StaticInitializer
where
    T: num_traits::AsPrimitive<i64>,
    T: num_traits::AsPrimitive<u64>,
    T: num_traits::AsPrimitive<f64>,
{
    match ctype {
        CType::Int => StaticInitializer::Initialized(Initializer::Int(val.as_())),
        CType::Long => StaticInitializer::Initialized(Initializer::Long(val.as_())),
        CType::UnsignedInt => StaticInitializer::Initialized(Initializer::UnsignedInt(val.as_())),
        CType::UnsignedLong => StaticInitializer::Initialized(Initializer::UnsignedLong(val.as_())),
        CType::Double => StaticInitializer::Initialized(Initializer::Double(val.as_())),
        CType::Pointer(_) if num_traits::AsPrimitive::<i64>::as_(val) == 0 => {
            StaticInitializer::Initialized(Initializer::UnsignedLong(0))
        }
        CType::Function(_, _) | CType::Pointer(_) | CType::Array(_, _) => panic!("Not a variable!"),
    }
}

fn type_check_var_declaration(
    var: VariableDeclaration,
    table: &mut TypeTable,
) -> VariableDeclaration {
    match var.storage {
        Some(StorageClass::Extern) => {
            assert!(var.init.is_none(), "Local extern variable cannot have initializer!");
            if let Some(symbol) = table.symbols.get(&var.name) {
                assert!(symbol.ctype == var.ctype, "Declaration types don't match!");
            } else {
                let init = StaticInitializer::None;
                let attrs = SymbolAttr::Static(StaticAttributes { init, global: true });
                table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
            }
            var
        }
        Some(StorageClass::Static) => {
            let init = parse_static_initializer(&var.init, &var.ctype);
            let attrs = SymbolAttr::Static(StaticAttributes { init, global: false });
            table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
            var
        }
        None => {
            let attrs = SymbolAttr::Local;
            table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
            let init_expr = var.init.map(VariableInitializer::get_single_init);
            let init = init_expr.map(|expr| type_check_and_convert(expr, table));
            let init = init.map(|expr| convert_by_assignment(expr, &var.ctype));
            let init = init.map(VariableInitializer::SingleElem);
            VariableDeclaration { init, ..var }
        }
    }
}

fn type_check_var_filescope(var: &VariableDeclaration, table: &mut TypeTable) {
    use StaticInitializer::{Initialized, None as NoInitializer, Tentative};
    let mut init_value = match (&var.init, var.storage) {
        (Some(_), _) => parse_static_initializer(&var.init, &var.ctype),
        (None, Some(StorageClass::Extern)) => NoInitializer,
        _ => Tentative,
    };
    let mut global = var.storage != Some(StorageClass::Static);
    if let Some(symbol) = table.symbols.get(&var.name) {
        let attrs = symbol.get_static_attrs();
        assert!(symbol.ctype == var.ctype, "Declaration types don't match!");
        if var.storage == Some(StorageClass::Extern) {
            global = attrs.global;
        } else if attrs.global != global {
            panic!("Conflicting linkage!");
        }
        if matches!(attrs.init, Initialized(_)) {
            assert!(!matches!(init_value, Initialized(_)), "Conflicting static initializers!");
            init_value = attrs.init;
        } else if attrs.init == Tentative && !matches!(init_value, Initialized(_)) {
            init_value = Tentative
        }
    }
    let attrs = SymbolAttr::Static(StaticAttributes { init: init_value, global });
    table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
}
