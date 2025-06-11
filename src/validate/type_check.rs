use std::{
    collections::{HashMap, HashSet},
    iter::zip,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    parse::parse_tree::{
        AssignmentExpression, BinaryExpression, BinaryOperator, Block, BlockItem,
        ConditionExpression, Constant, Declaration, Expression, ForInit, FunctionDeclaration,
        Program, Statement, StorageClass, TypedExpression, UnaryOperator, VariableDeclaration,
        VariableInitializer,
    },
    validate::{
        ctype::{
            CType, Initializer, IsLValue, StaticInitializer, Symbol, SymbolAttr, Symbols, TypeTable,
        },
        semantics::{AddLocation, Error, assert_or_err},
    },
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct TypedProgram {
    pub program: Program,
    pub symbols: Symbols,
}

pub fn eval_constant_expr(expr: &TypedExpression, ctype: &CType) -> Result<Constant> {
    match &expr.expr {
        Expression::Constant(constant) => eval_constant(constant, ctype),
        Expression::Unary(UnaryOperator::Negate, constexpr) => eval_constant_expr(constexpr, ctype),
        _ => Err(Error::new("Failed to parse constant expression!")),
    }
}

fn eval_constant(constant: &Constant, ctype: &CType) -> Result<Constant> {
    match ctype {
        CType::Char | CType::SignedChar => Ok(Constant::Char((constant.int_value() & 0xFF) as i64)),
        CType::Int => Ok(Constant::Int((constant.int_value() & 0xFFFFFFFF) as i64)),
        CType::Long => Ok(Constant::Long((constant.int_value() & 0xFFFFFFFFFFFFFFFF) as i64)),
        CType::UnsignedChar => Ok(Constant::UChar((constant.int_value() & 0xFF) as u64)),
        CType::UnsignedInt => Ok(Constant::UInt((constant.int_value() as u64) & 0xFFFFFFFF)),
        CType::UnsignedLong => Ok(Constant::ULong(constant.int_value() as u64)),
        CType::Double => match constant {
            Constant::Double(val) => Ok(Constant::Double(*val)),
            _ => Ok(Constant::Double(constant.int_value() as f64)),
        },
        CType::Pointer(ctype) | CType::Array(ctype, _) => match eval_constant(constant, ctype) {
            Ok(constant) if constant.int_value() == 0 => Ok(constant),
            _ => Err(Error::new("Failed to parse constant expression!")),
        },
        _ => Err(Error::new("Failed to parse constant expression!")),
    }
}

fn string_name() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("string.{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

// Set/Get Type

fn set_type(mut expression: TypedExpression, ctype: &CType) -> Result<TypedExpression> {
    expression.ctype = Some(ctype.clone());
    Ok(expression)
}
pub fn get_type(expression: &TypedExpression) -> CType {
    expression.ctype.clone().expect("Undefined type!")
}
pub fn is_null_ptr(expr: &TypedExpression) -> Result<bool> {
    let result = match eval_constant_expr(expr, expr.ctype.as_ref().expect("Has a type")) {
        Ok(val) => val.is_integer() && val.int_value() == 0,
        Err(_) => false,
    };
    Ok(result)
}
pub fn get_common_pointer_type(left: &TypedExpression, right: &TypedExpression) -> Result<CType> {
    let left_type = get_type(left);
    let right_type = get_type(right);
    if left_type == right_type || is_null_ptr(right)? {
        Ok(left_type)
    } else if is_null_ptr(left)? {
        Ok(right_type)
    } else {
        Err(Error::new("Expressions have incompatible types!"))
    }
}
pub fn get_common_type(left: &TypedExpression, right: &TypedExpression) -> Result<CType> {
    let left_type = get_type(left);
    let right_type = get_type(right);
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
fn convert_to(expression: TypedExpression, ctype: &CType) -> Result<TypedExpression> {
    if get_type(&expression) == *ctype {
        Ok(expression)
    } else {
        set_type(Expression::Cast(ctype.clone(), expression.into()).into(), ctype)
    }
}
fn promote_char(expression: TypedExpression) -> Result<TypedExpression> {
    let ctype = get_type(&expression);
    if ctype.is_char() { convert_to(expression, &CType::Int) } else { Ok(expression) }
}
fn convert_by_assignment(expr: TypedExpression, ctype: &CType) -> Result<TypedExpression> {
    if get_type(&expr) == *ctype {
        Ok(expr)
    } else if (get_type(&expr).is_arithmetic() && ctype.is_arithmetic())
        || (is_null_ptr(&expr)? && ctype.is_pointer())
    {
        convert_to(expr, ctype)
    } else {
        Err(Error::new("Cannot convert type for assignment!"))
    }
}

// Tree traversal functions
pub fn type_check_program(program: Program) -> Result<TypedProgram> {
    let mut table = TypeTable {
        symbols: HashMap::new(),
        current_function: None,
    };
    Ok(TypedProgram {
        program: program
            .into_iter()
            .map(|decl| type_check_declaration(decl, &mut table, true))
            .collect::<Result<Vec<Declaration>>>()?,
        symbols: table.symbols,
    })
}

fn type_check_declaration(
    decl: Declaration,
    table: &mut TypeTable,
    global: bool,
) -> Result<Declaration> {
    match decl {
        Declaration::Function(func) => {
            let location = func.location;
            Ok(Declaration::Function(type_check_function(func, table).add_location(location)?))
        }
        Declaration::Variable(mut var) => {
            let location = var.location;
            if global {
                type_check_var_filescope(&var, table).add_location(location)?;
            } else {
                var = type_check_var_declaration(var, table).add_location(location)?;
            }
            Ok(Declaration::Variable(var))
        }
    }
}

fn type_check_block(block: Block, table: &mut TypeTable) -> Result<Block> {
    block.into_iter().map(|item| type_check_block_item(item, table)).collect()
}
fn type_check_block_item(block_item: BlockItem, table: &mut TypeTable) -> Result<BlockItem> {
    match block_item {
        BlockItem::StatementItem(statement, location) => {
            let stmt = type_check_statement(statement, table).add_location(location)?;
            Ok(BlockItem::StatementItem(stmt, location))
        }
        BlockItem::DeclareItem(decl) => {
            Ok(BlockItem::DeclareItem(type_check_declaration(decl, table, false)?))
        }
    }
}
fn type_check_statement(statement: Statement, table: &mut TypeTable) -> Result<Statement> {
    use Statement::*;
    let result = match statement {
        Return(expr) => {
            let expr = type_check_and_convert(expr, table)?;
            let cur_func = table
                .current_function
                .as_ref()
                .ok_or(Error::new("Return must be inside function!"))?;
            let cur_func_type = &table.symbols[cur_func].ctype;
            let CType::Function(_, return_type) = cur_func_type else { panic!("Match failed!") };
            Return(convert_by_assignment(expr, return_type)?)
        }
        ExprStmt(expr) => ExprStmt(type_check_and_convert(expr, table)?),
        Label(name, statement) => Label(name, type_check_statement(*statement, table)?.into()),
        If(expr, if_true, if_false) => {
            let expr = type_check_and_convert(expr, table)?;
            let if_true = type_check_statement(*if_true, table)?;
            let if_false =
                if_false.map(|statement| type_check_statement(statement, table)).transpose()?;
            If(expr, if_true.into(), if_false.into())
        }
        While(mut loop_data) => {
            loop_data.condition = type_check_and_convert(loop_data.condition, table)?;
            loop_data.body = type_check_statement(*loop_data.body, table)?.into();
            While(loop_data)
        }
        DoWhile(mut loop_data) => {
            loop_data.condition = type_check_and_convert(loop_data.condition, table)?;
            loop_data.body = type_check_statement(*loop_data.body, table)?.into();
            DoWhile(loop_data)
        }
        Compound(block) => Compound(type_check_block(block, table)?),
        Switch {
            label,
            condition,
            cases,
            statement,
            default,
        } => {
            let condition = type_check_and_convert(condition, table)?;
            let condition = promote_char(condition)?;
            assert_or_err(get_type(&condition).is_int(), "Can only switch on integer types!")?;
            let cond_type = get_type(&condition);
            let mut new_cases = Vec::new();
            let mut case_vals = HashSet::new();
            for case in cases {
                let constexpr =
                    eval_constant_expr(&case.1, &cond_type).expect("Must be constexpr!");
                let case_typed = type_check_and_convert(case.1, table)?;
                assert_or_err(get_type(&case_typed).is_int(), "Switch case must be integer type!")?;
                assert_or_err(cond_type.is_int(), "Switch condition must be integer type!")?;
                assert_or_err(!case_vals.contains(&constexpr.int_value()), "Duplicate case!")?;
                case_vals.insert(constexpr.int_value());
                new_cases.push((case.0, Expression::Constant(constexpr).into()));
            }
            let statement = type_check_statement(*statement, table)?.into();
            Switch {
                label,
                condition,
                cases: new_cases,
                statement,
                default,
            }
        }
        For(init, mut loop_data, post) => {
            let init = match *init {
                ForInit::Decl(decl) => ForInit::Decl(type_check_var_declaration(decl, table)?),
                ForInit::Expr(Some(expr)) => {
                    ForInit::Expr(Some(type_check_and_convert(expr, table)?))
                }
                _ => ForInit::Expr(None),
            };
            loop_data.condition = type_check_and_convert(loop_data.condition, table)?;
            let post =
                post.map(|post_expr| type_check_and_convert(post_expr, table)).transpose()?;
            loop_data.body = type_check_statement(*loop_data.body, table)?.into();
            For(init.into(), loop_data, post)
        }
        Case(_, _) | Default(_) => {
            panic!("Should be re-written into label in switch resolve pass!")
        }
        Break(_) | Continue(_) | Goto(_) | Null => statement,
    };
    Ok(result)
}

fn type_check_and_convert(expr: TypedExpression, table: &mut TypeTable) -> Result<TypedExpression> {
    let typed_expr = type_check_expression(expr, table)?;
    match get_type(&typed_expr) {
        CType::Array(elem_t, _) => {
            let addr_expr = Expression::AddrOf(typed_expr.into());
            set_type(addr_expr.into(), &CType::Pointer(elem_t))
        }
        _ => Ok(typed_expr),
    }
}

fn type_check_expression(expr: TypedExpression, table: &mut TypeTable) -> Result<TypedExpression> {
    match expr.expr {
        Expression::Variable(name) => {
            let var_type = &table.symbols[&name].ctype;
            assert_or_err(
                !matches!(var_type, CType::Function(_, _)),
                "Function name used as variable!",
            )?;
            set_type(Expression::Variable(name).into(), var_type)
        }
        Expression::Constant(constant) => match constant {
            Constant::Int(_) => set_type(Expression::Constant(constant).into(), &CType::Int),
            Constant::Long(_) => set_type(Expression::Constant(constant).into(), &CType::Long),
            Constant::UInt(_) => {
                set_type(Expression::Constant(constant).into(), &CType::UnsignedInt)
            }
            Constant::ULong(_) => {
                set_type(Expression::Constant(constant).into(), &CType::UnsignedLong)
            }
            Constant::Double(_) => set_type(Expression::Constant(constant).into(), &CType::Double),
            Constant::Char(_) | Constant::UChar(_) => todo!("Implement char types!"),
        },
        Expression::Cast(new_type, inner) => {
            let typed_inner = type_check_and_convert(*inner, table)?;
            let old_type = get_type(&typed_inner);
            assert_or_err(
                new_type != CType::Double || !old_type.is_pointer(),
                "Cannot convert pointer to double!",
            )?;
            assert_or_err(
                old_type != CType::Double || !new_type.is_pointer(),
                "Cannot convert double to pointer!",
            )?;
            assert_or_err(!matches!(new_type, CType::Array(_, _)), "Cannot cast to array type")?;
            let cast_expr = Expression::Cast(new_type.clone(), typed_inner.into());
            set_type(cast_expr.into(), &new_type)
        }
        Expression::Unary(op, inner) => {
            let typed_inner = type_check_and_convert(*inner, table)?;
            let is_lvalue = typed_inner.is_lvalue();
            let new_type = get_type(&typed_inner);
            match op {
                UnaryOperator::LogicalNot => {
                    set_type(Expression::Unary(op, typed_inner.into()).into(), &CType::Int)
                }
                UnaryOperator::Complement => {
                    assert_or_err(new_type.is_int(), "Invalid operand for operator `~`")?;
                    let typed_inner = promote_char(typed_inner)?;
                    let result_t = get_type(&typed_inner);
                    set_type(Expression::Unary(op, typed_inner.into()).into(), &result_t)
                }
                UnaryOperator::Negate => {
                    assert_or_err(!new_type.is_pointer(), "Cannot negate pointer!")?;
                    let typed_inner = promote_char(typed_inner)?;
                    let result_t = get_type(&typed_inner);
                    set_type(Expression::Unary(op, typed_inner.into()).into(), &result_t)
                }
                UnaryOperator::Increment(_) => {
                    assert_or_err(is_lvalue, "Increment target must be lvalue!")?;
                    set_type(Expression::Unary(op, typed_inner.into()).into(), &new_type)
                }
            }
        }
        Expression::Binary(binary) => type_check_binary_expr(*binary, table),
        Expression::Assignment(assign) => {
            let left = type_check_and_convert(assign.left, table)?;
            let right = type_check_and_convert(assign.right, table)?;
            assert_or_err(left.is_lvalue(), "Assignment target must be lvalue!")?;
            let left_type = &get_type(&left);
            let right = convert_by_assignment(right, left_type)?;
            let assign = Expression::Assignment(AssignmentExpression { left, right }.into());
            set_type(assign.into(), left_type)
        }
        Expression::Condition(cond) => {
            let condition = type_check_and_convert(cond.condition, table)?;
            let if_true = type_check_and_convert(cond.if_true, table)?;
            let if_false = type_check_and_convert(cond.if_false, table)?;
            let common_type = get_common_type(&if_true, &if_false)?;
            let if_true = convert_to(if_true, &common_type)?;
            let if_false = convert_to(if_false, &common_type)?;
            let condition_expr =
                Expression::Condition(ConditionExpression { condition, if_true, if_false }.into());
            set_type(condition_expr.into(), &common_type)
        }
        Expression::FunctionCall(name, args) => {
            let expected_type = table.symbols[&name].ctype.clone();
            match expected_type {
                CType::Function(arg_types, ret_type) => {
                    assert_or_err(arg_types.len() == args.len(), "Incorrect number of arguments!")?;
                    let mut converted_args = Vec::new();
                    for (arg, arg_type) in zip(args, arg_types) {
                        let typed_arg = type_check_and_convert(arg, table)?;
                        converted_args.push(convert_by_assignment(typed_arg, &arg_type)?);
                    }
                    let call_expr = Expression::FunctionCall(name, converted_args);
                    set_type(call_expr.into(), &ret_type)
                }
                _ => Err(Error::new("Variable used as function!")),
            }
        }
        Expression::Dereference(inner) => {
            let typed_inner = type_check_and_convert(*inner, table)?;
            match get_type(&typed_inner) {
                CType::Pointer(reference_t) => {
                    let deref_expr = Expression::Dereference(typed_inner.into());
                    set_type(deref_expr.into(), &reference_t)
                }
                _ => Err(Error::new("Invalid dreference!")),
            }
        }
        Expression::AddrOf(inner) => {
            assert_or_err(inner.is_lvalue(), "Invalid operand to unary `&`, expected lvalue!")?;
            let typed_inner = type_check_expression(*inner, table)?;
            let reference_t = get_type(&typed_inner).into();
            let addr_expr = Expression::AddrOf(typed_inner.into());
            set_type(addr_expr.into(), &CType::Pointer(reference_t))
        }
        Expression::Subscript(left, right) => {
            let mut left = type_check_and_convert(*left, table)?;
            let mut right = type_check_and_convert(*right, table)?;
            let left_t = get_type(&left);
            let right_t = get_type(&right);
            if let CType::Pointer(ptr_t) = &left_t {
                assert_or_err(right_t.is_int(), "Subscript takes integer and pointer operands")?;
                right = convert_to(right, &CType::Long)?;
                set_type(Expression::Subscript(left.into(), right.into()).into(), ptr_t)
            } else if let CType::Pointer(ptr_t) = &right_t {
                assert_or_err(left_t.is_int(), "Subscript takes integer and pointer operands")?;
                left = convert_to(left, &CType::Long)?;
                // Swap left and right so left is always the pointer
                set_type(Expression::Subscript(right.into(), left.into()).into(), ptr_t)
            } else {
                Err(Error::new("Subscript takes integer and pointer operands!"))
            }
        }
        Expression::StringLiteral(ref string_data) => {
            let string_len = string_data.len() as u64 + 1;
            set_type(expr.expr.into(), &CType::Array(CType::Char.into(), string_len))
        }
    }
}

fn type_check_binary_expr(
    binary: BinaryExpression,
    table: &mut TypeTable,
) -> Result<TypedExpression> {
    use BinaryOperator::*;
    use Expression::Binary;
    let left = type_check_and_convert(binary.left, table)?;
    let right = type_check_and_convert(binary.right, table)?;
    match binary.operator {
        LogicalAnd | LogicalOr => {
            let binexpr = BinaryExpression { left, right, ..binary };
            set_type(Binary(binexpr.into()).into(), &CType::Int)
        }
        LessThan | GreaterThan | LessThanEqual | GreaterThanEqual | IsEqual | NotEqual => {
            let common_type = get_common_type(&left, &right)?;
            if common_type.is_pointer() && !matches!(binary.operator, IsEqual | NotEqual) {
                assert_or_err(
                    !is_null_ptr(&left)? && !is_null_ptr(&right)?,
                    "Cannot compare null with relational operators in C",
                )?;
                assert_or_err(
                    get_type(&left) == get_type(&right),
                    "Pointer comparison type mismatch",
                )?;
            }
            let left = convert_to(left, &common_type)?;
            let right = convert_to(right, &common_type)?;
            let binexpr = BinaryExpression { left, right, ..binary };
            set_type(Binary(binexpr.into()).into(), &CType::Int)
        }
        LeftShift | RightShift => {
            let common_type = get_common_type(&left, &right)?;
            let result_t = get_type(&left);
            assert_or_err(common_type.is_int(), "Operands for bitshift must be integer types!")?;
            let binexpr = BinaryExpression { left, right, ..binary };
            set_type(Binary(binexpr.into()).into(), &result_t)
        }
        Multiply | Divide => {
            let common_type = get_common_type(&left, &right)?;
            assert_or_err(!common_type.is_pointer(), "Cannot multiply/divide pointers!")?;
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        Remainder => {
            let common_type = get_common_type(&left, &right)?;
            assert_or_err(common_type.is_int(), "Operands for remainder must be integer types!")?;
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        BitAnd | BitOr | BitXor => {
            let common_type = get_common_type(&left, &right)?;
            assert_or_err(
                common_type.is_int(),
                "Operands for bit operators must be integer types!",
            )?;
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        Add => {
            let left_t = get_type(&left);
            let right_t = get_type(&right);
            if left_t.is_arithmetic() && right_t.is_arithmetic() {
                type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
            } else if left_t.is_pointer() && right_t.is_int() {
                assert_or_err(
                    !binary.is_assignment || left.is_lvalue(),
                    "Can only assign to lvalue!",
                )?;
                let right = convert_to(right, &CType::Long)?;
                let binexpr = BinaryExpression { left, right, ..binary };
                set_type(Binary(binexpr.into()).into(), &left_t)
            } else if right_t.is_pointer() && left_t.is_int() {
                // If right is pointer, we swap left/right, as tac expects the ptr to be on the left
                assert_or_err(
                    !binary.is_assignment,
                    "Adding int to pointer gives pointer, not int",
                )?;
                let left = convert_to(left, &CType::Long)?;
                let binexpr = BinaryExpression { left: right, right: left, ..binary };
                set_type(Binary(binexpr.into()).into(), &right_t)
            } else {
                Err(Error::new("Invalid operands for addition"))
            }
        }
        Subtract => {
            let left_t = get_type(&left);
            let right_t = get_type(&right);
            if left_t.is_arithmetic() && right_t.is_arithmetic() {
                type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
            } else if left_t.is_pointer() && right_t.is_int() {
                assert_or_err(
                    !binary.is_assignment || left.is_lvalue(),
                    "Can only assign to lvalue",
                )?;
                // Subtraction with pointer/int is addition with negated value
                let right = convert_to(right, &CType::Long)?;
                let rightexpr = Expression::Unary(UnaryOperator::Negate, right.into());
                let right = set_type(rightexpr.into(), &CType::Long)?;
                let binexpr = BinaryExpression { left, right, operator: Add, ..binary };
                set_type(Binary(binexpr.into()).into(), &left_t)
            } else if left_t.is_pointer() && left_t == right_t {
                assert_or_err(
                    !binary.is_assignment,
                    "Subtracting pointers gives int, not pointer!",
                )?;
                let binexpr = BinaryExpression { left, right, ..binary };
                set_type(Binary(binexpr.into()).into(), &CType::Long)
            } else {
                Err(Error::new("Invalid operands for subtraction"))
            }
        }
    }
}

fn type_check_arithmetic_binexpr(
    left: TypedExpression,
    right: TypedExpression,
    operator: BinaryOperator,
    is_assignment: bool,
) -> Result<TypedExpression> {
    let common_type = get_common_type(&left, &right)?;
    let binexpr = if is_assignment {
        assert_or_err(left.is_lvalue(), "Can only assign to lvalue!")?;
        let right = convert_to(right, &common_type)?;
        BinaryExpression { left, right, operator, is_assignment }
    } else {
        let left = convert_to(left, &common_type)?;
        let right = convert_to(right, &common_type)?;
        BinaryExpression { left, right, operator, is_assignment }
    };
    set_type(Expression::Binary(binexpr.into()).into(), &common_type)
}

fn type_check_function(
    function: FunctionDeclaration,
    table: &mut TypeTable,
) -> Result<FunctionDeclaration> {
    let function = type_check_array_params(function)?;
    let ctype = &function.ctype;
    let mut defined = function.body.is_some();
    let mut global = function.storage != Some(StorageClass::Static);
    if let Some(symbol) = table.symbols.get(&function.name) {
        let SymbolAttr::Function { defined: prev_def, global: prev_global } = symbol.attrs else {
            return Err(Error::new("Expected function!"));
        };
        assert_or_err(symbol.ctype == *ctype, "Incompatible function declarations!")?;
        assert_or_err(!prev_def || !defined, "Function redefinition!")?;
        assert_or_err(!prev_global || global, "static definition cannot follow non-static")?;
        global = prev_global;
        defined = prev_def;
    }
    let attrs = SymbolAttr::Function { defined, global };
    table.symbols.insert(function.name.clone(), Symbol { ctype: ctype.clone(), attrs });
    if let Some(body) = function.body {
        let CType::Function(param_types, _) = ctype else { panic!("Not a function!") };
        for (param, param_type) in zip(&function.params, param_types) {
            let attrs = SymbolAttr::Local;
            table.symbols.insert(param.clone(), Symbol { ctype: param_type.clone(), attrs });
        }
        table.current_function = Some(function.name.clone());
        let body = Some(type_check_block(body, table)?);
        table.current_function = None;
        Ok(FunctionDeclaration { body, ..function })
    } else {
        Ok(function)
    }
}

fn type_check_array_params(mut function: FunctionDeclaration) -> Result<FunctionDeclaration> {
    let CType::Function(param_types, return_t) = function.ctype else { panic!("Not a function!") };
    assert_or_err(!matches!(*return_t, CType::Array(_, _)), "Function cannot return array")?;
    let adjusted_params = param_types
        .into_iter()
        .map(|param_t| match param_t {
            CType::Array(elem_t, _) => CType::Pointer(elem_t),
            _ => param_t,
        })
        .collect();
    function.ctype = CType::Function(adjusted_params, return_t);
    Ok(function)
}

fn parse_static_initializer(
    init: Option<&VariableInitializer>,
    ctype: &CType,
) -> Result<StaticInitializer> {
    match init {
        None => Ok(StaticInitializer::Initialized(vec![Initializer::ZeroInit(ctype.size())])),
        Some(VariableInitializer::SingleElem(expr)) => {
            if let CType::Array(elem_t, size) = ctype {
                let Expression::StringLiteral(s_data) = &expr.expr else {
                    return Err(Error::new("Cannot initialize static array with a scalar!"));
                };
                assert_or_err(elem_t.is_char(), "Cannot initialize a non-char with a string!")?;
                assert_or_err(s_data.len() as u64 <= *size, "Too many characters in string!")?;
                let null_terminated = *size > s_data.len() as u64;
                let mut init_list =
                    vec![Initializer::StringInit { data: s_data.clone(), null_terminated }];
                if null_terminated {
                    let padding = *size - s_data.len() as u64 - 1;
                    if padding != 0 {
                        init_list.push(Initializer::ZeroInit(padding));
                    }
                }
                Ok(StaticInitializer::Initialized(init_list))
            } else {
                assert_or_err(
                    !matches!(ctype, CType::Array(_, _)),
                    "Cannot initialize static array with a scalar!",
                )?;
                let expr_constant = eval_constant_expr(expr, ctype);
                let init_val = match expr_constant {
                    Ok(Constant::Int(val) | Constant::Long(val) | Constant::Char(val)) => {
                        parse_initializer_value(val, ctype)
                    }
                    Ok(Constant::UInt(val) | Constant::ULong(val) | Constant::UChar(val)) => {
                        parse_initializer_value(val, ctype)
                    }
                    Ok(Constant::Double(val)) => parse_initializer_value(val, ctype),
                    Err(_) => return Err(Error::new("Failed to parse static initializer")),
                };
                Ok(StaticInitializer::Initialized(vec![init_val]))
            }
        }
        Some(VariableInitializer::CompoundInit(init_list)) => {
            let CType::Array(elem_t, size) = ctype else {
                return Err(Error::new("Cannot initiate static scalar with initializer list"));
            };
            assert_or_err(init_list.len() as u64 <= *size, "Initializer longer than array!")?;
            let mut initializers = Vec::new();
            for initializer in init_list {
                let sub_init = parse_static_initializer(Some(initializer), elem_t)?;
                let StaticInitializer::Initialized(mut sub_inits) = sub_init else {
                    panic!("Should be initialized")
                };
                initializers.append(&mut sub_inits);
            }
            if (init_list.len() as u64) < *size {
                let zero_bytes = (size - init_list.len() as u64) * elem_t.size();
                initializers.push(Initializer::ZeroInit(zero_bytes));
            }
            Ok(StaticInitializer::Initialized(initializers))
        }
    }
}

fn parse_initializer_value<T>(val: T, ctype: &CType) -> Initializer
where
    T: num_traits::AsPrimitive<i64>,
    T: num_traits::AsPrimitive<u64>,
    T: num_traits::AsPrimitive<f64>,
{
    if num_traits::AsPrimitive::<i64>::as_(val) == 0 && *ctype != CType::Double {
        return Initializer::ZeroInit(ctype.size());
    }
    match ctype {
        CType::Char | CType::SignedChar => Initializer::Char(val.as_()),
        CType::Int => Initializer::Int(val.as_()),
        CType::Long => Initializer::Long(val.as_()),
        CType::UnsignedChar => Initializer::UChar(val.as_()),
        CType::UnsignedInt => Initializer::UInt(val.as_()),
        CType::UnsignedLong => Initializer::ULong(val.as_()),
        CType::Double => Initializer::Double(val.as_()),
        CType::Pointer(_) if num_traits::AsPrimitive::<i64>::as_(val) == 0 => Initializer::ULong(0),
        CType::Function(_, _) | CType::Pointer(_) | CType::Array(_, _) => panic!("Not a variable!"),
    }
}

fn zero_initializer(target_t: &CType) -> VariableInitializer {
    use VariableInitializer::*;
    match target_t {
        CType::Char | CType::SignedChar => {
            SingleElem(Expression::Constant(Constant::Char(0)).into())
        }
        CType::UnsignedChar => SingleElem(Expression::Constant(Constant::UChar(0)).into()),
        CType::Int => SingleElem(Expression::Constant(Constant::Int(0)).into()),
        CType::Long | CType::Pointer(_) => {
            SingleElem(Expression::Constant(Constant::Long(0)).into())
        }
        CType::UnsignedInt => SingleElem(Expression::Constant(Constant::UInt(0)).into()),
        CType::UnsignedLong => SingleElem(Expression::Constant(Constant::ULong(0)).into()),
        CType::Double => SingleElem(Expression::Constant(Constant::Double(0.0)).into()),
        CType::Array(elem_t, size) => CompoundInit(vec![zero_initializer(elem_t); *size as usize]),
        CType::Function(_, _) => panic!("Cannot zero initialize a function"),
    }
}

fn type_check_initializer(
    init: VariableInitializer,
    target_t: &CType,
    table: &mut TypeTable,
) -> Result<VariableInitializer> {
    use VariableInitializer::*;
    match init {
        SingleElem(expr) => {
            if let CType::Array(elem_t, size) = target_t {
                let Expression::StringLiteral(data) = &expr.expr else {
                    return Err(Error::new("Cannot initialize array with a scalar!"));
                };
                assert_or_err(
                    elem_t.is_char(),
                    "Cannot initialize a non-char type with a string!",
                )?;
                assert_or_err(
                    data.len() as u64 <= *size,
                    "Too many characters in strign literal!",
                )?;
                let expr = set_type(expr, target_t)?;
                Ok(SingleElem(expr))
            } else {
                let expr = type_check_and_convert(expr, table)?;
                let expr = convert_by_assignment(expr, target_t)?;
                Ok(SingleElem(expr))
            }
        }
        CompoundInit(init_list) => {
            let CType::Array(elem_t, size) = target_t else {
                return Err(Error::new("Cannot initialize a scalar with a compound initializer!"));
            };
            assert_or_err(init_list.len() as u64 <= *size, "Too many items in initializer!")?;
            let mut typechecked_list = init_list
                .into_iter()
                .map(|init| type_check_initializer(init, elem_t, table))
                .collect::<Result<Vec<VariableInitializer>>>()?;
            while (typechecked_list.len() as u64) < *size {
                typechecked_list.push(zero_initializer(elem_t));
            }
            Ok(CompoundInit(typechecked_list))
        }
    }
}

fn is_string_literal(initializer: &Option<VariableInitializer>) -> bool {
    match initializer {
        Some(VariableInitializer::SingleElem(expr)) => {
            matches!(expr.expr, Expression::StringLiteral(_))
        }
        _ => false,
    }
}

fn type_check_var_declaration(
    var: VariableDeclaration,
    table: &mut TypeTable,
) -> Result<VariableDeclaration> {
    match var.storage {
        Some(StorageClass::Extern) => {
            assert_or_err(var.init.is_none(), "Local extern variable cannot have initializer!")?;
            if let Some(symbol) = table.symbols.get(&var.name) {
                assert_or_err(symbol.ctype == var.ctype, "Declaration types don't match!")?;
            } else {
                let init = StaticInitializer::None;
                let attrs = SymbolAttr::Static { init, global: true };
                table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
            }
            Ok(var)
        }
        Some(StorageClass::Static) => {
            match &var.ctype {
                // Handle string literal initialization separately for pointers
                CType::Pointer(elem_t) if is_string_literal(&var.init) => {
                    let Some(VariableInitializer::SingleElem(expr)) = &var.init else {
                        panic!("Unreachable!")
                    };
                    let Expression::StringLiteral(s_data) = &expr.expr else {
                        panic!("Is string!")
                    };
                    assert_or_err(**elem_t == CType::Char, "Can only init char* with string!")?;
                    // Insert constant static string into symbol table
                    let name = string_name();
                    table.symbols.insert(
                        name.clone(),
                        Symbol {
                            ctype: CType::Array(CType::Char.into(), s_data.len() as u64 + 1),
                            attrs: SymbolAttr::Constant(Initializer::StringInit {
                                data: s_data.clone(),
                                null_terminated: true,
                            }),
                        },
                    );
                    // Insert variable into symbol table with PointerInit from the static string
                    let attrs = SymbolAttr::Static {
                        init: StaticInitializer::Initialized(vec![Initializer::PointerInit(name)]),
                        global: false,
                    };
                    table
                        .symbols
                        .insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
                }
                // Handle general static initializers
                _ => {
                    let init = parse_static_initializer(var.init.as_ref(), &var.ctype)?;
                    let attrs = SymbolAttr::Static { init, global: false };
                    table
                        .symbols
                        .insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
                }
            }
            Ok(var)
        }
        None => {
            let attrs = SymbolAttr::Local;
            table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
            let init =
                var.init.map(|init| type_check_initializer(init, &var.ctype, table)).transpose()?;
            Ok(VariableDeclaration { init, ..var })
        }
    }
}

fn type_check_var_filescope(var: &VariableDeclaration, table: &mut TypeTable) -> Result<()> {
    use StaticInitializer::{Initialized, None as NoInitializer, Tentative};
    let mut init_value = match (&var.init, var.storage) {
        (Some(_), _) => parse_static_initializer(var.init.as_ref(), &var.ctype)?,
        (None, Some(StorageClass::Extern)) => NoInitializer,
        _ => Tentative,
    };
    let mut global = var.storage != Some(StorageClass::Static);
    if let Some(symbol) = table.symbols.get(&var.name) {
        let SymbolAttr::Static { init: prev_init, global: prev_global } = &symbol.attrs else {
            return Err(Error::new("Expected static attributes!"));
        };
        assert_or_err(symbol.ctype == var.ctype, "Declaration types don't match!")?;
        if var.storage == Some(StorageClass::Extern) {
            global = *prev_global;
        } else if global != *prev_global {
            return Err(Error::new("Conflicting linkage!"));
        }
        if matches!(prev_init, Initialized(_)) {
            assert_or_err(
                !matches!(init_value, Initialized(_)),
                "Conflicting static initializers!",
            )?;
            init_value = prev_init.clone();
        } else if *prev_init == Tentative && !matches!(init_value, Initialized(_)) {
            init_value = Tentative
        }
    }
    let attrs = SymbolAttr::Static { init: init_value, global };
    table.symbols.insert(var.name.clone(), Symbol { ctype: var.ctype.clone(), attrs });
    Ok(())
}
