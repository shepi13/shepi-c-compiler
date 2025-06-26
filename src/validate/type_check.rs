use std::{
    collections::{HashMap, HashSet},
    iter::zip,
};

use crate::{
    helpers::error::{AddLocation, Error, assert_or_err},
    parse::parse_tree::{
        AssignmentExpression, BinaryExpression, BinaryOperator, Block, BlockItem,
        ConditionExpression, Constant, Declaration, Expression, ForInit, FunctionDeclaration,
        Program, Statement, StorageClass, TypedExpression, UnaryOperator, VariableDeclaration,
        VariableInitializer,
    },
    validate::ctype::{
        CType, Initializer, IsLValue, StaticInitializer, Symbol, SymbolAttr, Symbols, TypeTable,
        get_common_type, string_name,
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
        _ => Err(Error::new("Invalid constant", "Failed to parse constant expression!")),
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
        CType::Pointer(elem_t) if **elem_t == CType::Void => {
            eval_constant(constant, &CType::Pointer(Box::new(CType::Char)))
        }
        CType::Pointer(ctype) | CType::Array(ctype, _) => match eval_constant(constant, ctype) {
            Ok(constant) if constant.int_value() == 0 => Ok(constant),
            _ => Err(Error::new("Invalid constant", "Failed to parse constant expression!")),
        },
        _ => Err(Error::new("Invalid constant", "Failed to parse constant expression!")),
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
            assert_or_err(
                var.ctype != CType::Void,
                "Illegal Declaration",
                "Cannot declare void variables",
            )
            .add_location(location)?;
            var.ctype.validate_specifier().add_location(location)?;
            Ok(Declaration::Variable(var))
        }
        Declaration::Struct(_) => todo!("Struct type check!"),
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
        Return(Some(expr)) => {
            let expr = type_check_and_convert(expr, table)?;
            let return_type = table.current_function_return_type()?;
            Return(Some(expr.convert_by_assignment(return_type)?))
        }
        Return(None) => {
            let return_type = table.current_function_return_type()?;
            assert_or_err(
                matches!(return_type, CType::Void),
                "Invalid return type",
                "Empty return from non-void function!",
            )?;
            Return(None)
        }
        ExprStmt(expr) => ExprStmt(type_check_and_convert(expr, table)?),
        Label(name, statement) => Label(name, type_check_statement(*statement, table)?.into()),
        If(expr, if_true, if_false) => {
            let expr = type_check_and_convert(expr, table)?;
            assert_or_err(
                expr.get_type().is_scalar(),
                "Invalid condition",
                "Condition must be a scalar type!",
            )?;
            let if_true = type_check_statement(*if_true, table)?;
            let if_false =
                if_false.map(|statement| type_check_statement(statement, table)).transpose()?;
            If(expr, if_true.into(), if_false.into())
        }
        While(mut loop_data) => {
            loop_data.condition = type_check_and_convert(loop_data.condition, table)?;
            assert_or_err(
                loop_data.condition.get_type().is_scalar(),
                "Invalid condition",
                "Condition must be a scalar type!",
            )?;
            loop_data.body = type_check_statement(*loop_data.body, table)?.into();
            While(loop_data)
        }
        DoWhile(mut loop_data) => {
            loop_data.condition = type_check_and_convert(loop_data.condition, table)?;
            assert_or_err(
                loop_data.condition.get_type().is_scalar(),
                "Invalid condition",
                "Condition must be a scalar type!",
            )?;
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
            let condition = type_check_and_convert(condition, table)?.promote_char()?;
            assert_or_err(
                condition.get_type().is_int(),
                "Invalid switch condition",
                "Can only switch on integer types!",
            )?;
            let cond_type = condition.get_type();
            let mut new_cases = Vec::new();
            let mut case_vals = HashSet::new();
            for case in cases {
                let constexpr =
                    eval_constant_expr(&case.1, &cond_type).expect("Must be constexpr!");
                let case_typed = type_check_and_convert(case.1, table)?;
                assert_or_err(
                    case_typed.get_type().is_int(),
                    "Invalid switch case",
                    "Switch case must be integer type!",
                )?;
                assert_or_err(
                    cond_type.is_int(),
                    "Invalid switch case",
                    "Switch condition must be integer type!",
                )?;
                assert_or_err(
                    !case_vals.contains(&constexpr.int_value()),
                    "Invalid switch case",
                    "Duplicate case!",
                )?;
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
            assert_or_err(
                loop_data.condition.get_type().is_scalar(),
                "Invalid condition",
                "Condition must be a scalar type!",
            )?;
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
    match typed_expr.get_type() {
        CType::Array(elem_t, _) => {
            Expression::AddrOf(typed_expr.into()).set_type(&CType::Pointer(elem_t))
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
                "Illegal declaration",
                "Function name used as variable!",
            )?;
            Expression::Variable(name).set_type(var_type)
        }
        Expression::Constant(constant) => match constant {
            Constant::Int(_) => Expression::Constant(constant).set_type(&CType::Int),
            Constant::Long(_) => Expression::Constant(constant).set_type(&CType::Long),
            Constant::UInt(_) => Expression::Constant(constant).set_type(&CType::UnsignedInt),
            Constant::ULong(_) => Expression::Constant(constant).set_type(&CType::UnsignedLong),
            Constant::Double(_) => Expression::Constant(constant).set_type(&CType::Double),
            Constant::Char(_) => Expression::Constant(constant).set_type(&CType::Char),
            Constant::UChar(_) => Expression::Constant(constant).set_type(&CType::UnsignedChar),
        },
        Expression::Cast(new_type, inner) => {
            new_type.validate_specifier()?;
            let typed_inner = type_check_and_convert(*inner, table)?;
            let old_type = typed_inner.get_type();
            assert_or_err(
                new_type != CType::Double || !old_type.is_pointer(),
                "Invalid cast",
                "Cannot convert pointer to double!",
            )?;
            assert_or_err(
                old_type != CType::Double || !new_type.is_pointer(),
                "Invalid cast",
                "Cannot convert double to pointer!",
            )?;
            if new_type == CType::Void {
                Expression::Cast(CType::Void, typed_inner.into()).set_type(&CType::Void)
            } else if !new_type.is_scalar() {
                return Err(Error::new("Invalid cast", "Can only cast to scalar or void!"));
            } else if !old_type.is_scalar() {
                return Err(Error::new("Invalid cast", "Cannot cast non-scalar type to scalar!"));
            } else {
                Expression::Cast(new_type.clone(), typed_inner.into()).set_type(&new_type)
            }
        }
        Expression::Unary(op, inner) => {
            let typed_inner = type_check_and_convert(*inner, table)?;
            let is_lvalue = typed_inner.is_lvalue();
            let new_type = typed_inner.get_type();
            match op {
                UnaryOperator::LogicalNot => {
                    assert_or_err(
                        new_type.is_scalar(),
                        "Illegal operand",
                        "Invalid operand for operator `!`, not a scalar!",
                    )?;
                    Expression::Unary(op, typed_inner.into()).set_type(&CType::Int)
                }
                UnaryOperator::Complement => {
                    assert_or_err(
                        new_type.is_int(),
                        "Illegal operand",
                        "Invalid operand for operator `~`",
                    )?;
                    let typed_inner = typed_inner.promote_char()?;
                    let result_t = typed_inner.get_type();
                    Expression::Unary(op, typed_inner.into()).set_type(&result_t)
                }
                UnaryOperator::Negate => {
                    assert_or_err(
                        new_type.is_arithmetic(),
                        "Illegal operand",
                        "Invalid operand for operator `-`, must be arithmetic!",
                    )?;
                    let typed_inner = typed_inner.promote_char()?;
                    let result_t = typed_inner.get_type();
                    Expression::Unary(op, typed_inner.into()).set_type(&result_t)
                }
                UnaryOperator::Increment(_) => {
                    assert_or_err(
                        is_lvalue,
                        "Illegal operand",
                        "Increment target must be lvalue!",
                    )?;
                    assert_or_err(
                        new_type.is_pointer_to_complete() || new_type.is_arithmetic(),
                        "Illegal operand",
                        "Cannot increment pointer to incomplete type!",
                    )?;
                    Expression::Unary(op, typed_inner.into()).set_type(&new_type)
                }
            }
        }
        Expression::Binary(binary) => type_check_binary_expr(*binary, table),
        Expression::Assignment(assign) => {
            let left = type_check_and_convert(assign.left, table)?;
            let right = type_check_and_convert(assign.right, table)?;
            assert_or_err(
                left.is_lvalue(),
                "Illegal assignment",
                "Assignment target must be lvalue!",
            )?;
            let left_type = left.get_type();
            let right = right.convert_by_assignment(&left_type)?;
            Expression::Assignment(AssignmentExpression { left, right }.into()).set_type(&left_type)
        }
        Expression::Condition(cond) => {
            let condition = type_check_and_convert(cond.condition, table)?;
            assert_or_err(
                condition.get_type().is_scalar(),
                "Invalid condition",
                "Condition must be a scalar type!",
            )?;
            let if_true = type_check_and_convert(cond.if_true, table)?;
            let if_false = type_check_and_convert(cond.if_false, table)?;
            let common_type = get_common_type(&if_true, &if_false)?;
            let if_true = if_true.convert_to(&common_type)?;
            let if_false = if_false.convert_to(&common_type)?;
            Expression::Condition(ConditionExpression { condition, if_true, if_false }.into())
                .set_type(&common_type)
        }
        Expression::FunctionCall(name, args) => {
            let expected_type = table.symbols[&name].ctype.clone();
            match expected_type {
                CType::Function(arg_types, ret_type) => {
                    assert_or_err(
                        arg_types.len() == args.len() || arg_types.last() == Some(&CType::VarArgs),
                        "Invalid function call",
                        "Incorrect number of arguments!",
                    )?;
                    let mut converted_args = Vec::new();
                    for (i, arg) in args.into_iter().enumerate() {
                        let typed_arg = type_check_and_convert(arg, table)?;
                        if i < arg_types.len() && arg_types[i] != CType::VarArgs {
                            converted_args.push(typed_arg.convert_by_assignment(&arg_types[i])?);
                        } else {
                            converted_args.push(typed_arg);
                        }
                    }
                    Expression::FunctionCall(name, converted_args).set_type(&ret_type)
                }
                _ => Err(Error::new("Invalid function call", "Variable used as function!")),
            }
        }
        Expression::Dereference(inner) => {
            let typed_inner = type_check_and_convert(*inner, table)?;
            match typed_inner.get_type() {
                CType::Pointer(reference_t) if *reference_t == CType::Void => {
                    Err(Error::new("Invalid dereference", "Cannot dereference void pointer!"))
                }
                CType::Pointer(reference_t) => {
                    Expression::Dereference(typed_inner.into()).set_type(&reference_t)
                }
                _ => Err(Error::new("Invalid dereference", "Operand is not pointer!")),
            }
        }
        Expression::AddrOf(inner) => {
            assert_or_err(inner.is_lvalue(), "Invalid operand", "Unary `&` expects lvalue!")?;
            let typed_inner = type_check_expression(*inner, table)?;
            let reference_t = typed_inner.get_type();
            Expression::AddrOf(typed_inner.into()).set_type(&CType::Pointer(reference_t.into()))
        }
        Expression::Subscript(left, right) => {
            let mut left = type_check_and_convert(*left, table)?;
            let mut right = type_check_and_convert(*right, table)?;
            if !left.get_type().is_pointer() {
                (left, right) = (right, left)
            }
            let ptr_t = left.get_type();
            let subscr_t = right.get_type();
            let CType::Pointer(elem_t) = &ptr_t else {
                return Err(Error::new(
                    "Illegal subscript",
                    "Subscript `[]` takes integer and pointer operands!",
                ));
            };
            assert_or_err(
                subscr_t.is_int(),
                "Illegal subscript",
                "Subscript `[]` takes integer and pointer operands",
            )?;
            assert_or_err(
                elem_t.is_complete(),
                "Illegal subscript",
                "Subscript to incomplete type!",
            )?;
            let right = right.convert_to(&CType::Long)?;
            Expression::Subscript(left.into(), right.into()).set_type(elem_t)
        }
        Expression::StringLiteral(ref string_data) => {
            let string_len = string_data.len() as u64 + 1;
            expr.expr.set_type(&CType::Array(CType::Char.into(), string_len))
        }
        Expression::SizeOf(inner) => {
            let typed_inner = type_check_expression(*inner, table)?;
            assert_or_err(
                typed_inner.get_type().is_complete(),
                "Invalid sizeof operand",
                "Cannot get size of incomplete type",
            )?;
            Expression::SizeOf(Box::new(typed_inner)).set_type(&CType::UnsignedLong)
        }
        Expression::SizeOfT(ref type_t) => {
            type_t.validate_specifier()?;
            assert_or_err(
                type_t.is_complete(),
                "Invalid sizeof operand",
                "Cannot get size of incomplete type!",
            )?;
            expr.expr.set_type(&CType::UnsignedLong)
        }
        Expression::DotAccess(_, _) => todo!("Type check dot access!"),
        Expression::Arrow(_, _) => todo!("Type check arrow access!"),
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
            assert_or_err(
                left.get_type().is_scalar() && right.get_type().is_scalar(),
                "Illegal Operand",
                "Invalid operands for `and`, expected scalars!",
            )?;
            Binary(BinaryExpression { left, right, ..binary }.into()).set_type(&CType::Int)
        }
        IsEqual | NotEqual => {
            let common_type = get_common_type(&left, &right)?;
            let left_t = left.get_type();
            let right_t = right.get_type();
            assert_or_err(
                common_type.is_pointer() || left_t.is_arithmetic() && right_t.is_arithmetic(),
                "Invalid comparision",
                "Equality checks expect pointer or arithmetic types",
            )?;
            let left = left.convert_to(&common_type)?;
            let right = right.convert_to(&common_type)?;
            Binary(BinaryExpression { left, right, ..binary }.into()).set_type(&CType::Int)
        }
        LessThan | GreaterThan | LessThanEqual | GreaterThanEqual => {
            let common_type = get_common_type(&left, &right)?;
            if common_type.is_pointer() {
                assert_or_err(
                    !left.is_null_ptr()? && !right.is_null_ptr()?,
                    "Invalid comparision",
                    "Cannot use null with relational operators in C",
                )?;
                assert_or_err(
                    left.get_type() == right.get_type(),
                    "Invalid comparison",
                    "Pointer comparison types must match!",
                )?;
                assert_or_err(
                    left.get_type().is_pointer_to_complete()
                        && right.get_type().is_pointer_to_complete(),
                    "Invalid comparison",
                    "Pointer comparision types must be complete!",
                )?;
            } else {
                assert_or_err(
                    left.get_type().is_arithmetic() && right.get_type().is_arithmetic(),
                    "Invalid comparision",
                    "Types must be arithmetic or pointer",
                )?;
            }
            let left = left.convert_to(&common_type)?;
            let right = right.convert_to(&common_type)?;
            Binary(BinaryExpression { left, right, ..binary }.into()).set_type(&CType::Int)
        }
        LeftShift | RightShift => {
            let left = if !binary.is_assignment { left.promote_char()? } else { left };
            let common_type = get_common_type(&left, &right)?;
            let result_t = left.get_type();
            assert_or_err(
                common_type.is_int(),
                "Invalid operand",
                "Operands for bitshift must be integer types!",
            )?;
            Binary(BinaryExpression { left, right, ..binary }.into()).set_type(&result_t)
        }
        Multiply | Divide => {
            let common_type = get_common_type(&left, &right)?;
            assert_or_err(
                common_type.is_arithmetic(),
                "Invalid operand",
                "Operands for multiply/divide must be arithmetic!",
            )?;
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        Remainder => {
            let common_type = get_common_type(&left, &right)?;
            assert_or_err(
                common_type.is_int(),
                "Invalid operand",
                "Operands for remainder must be integer types!",
            )?;
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        BitAnd | BitOr | BitXor => {
            let common_type = get_common_type(&left, &right)?;
            assert_or_err(
                common_type.is_int(),
                "Invalid Operand",
                "Operands for bit operators must be integer types!",
            )?;
            type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
        }
        Add => {
            let left_t = left.get_type();
            let right_t = right.get_type();
            if left_t.is_arithmetic() && right_t.is_arithmetic() {
                type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
            } else if left_t.is_pointer_to_complete() && right_t.is_int() {
                assert_or_err(
                    !binary.is_assignment || left.is_lvalue(),
                    "Invalid assignment",
                    "Can only assign to lvalue!",
                )?;
                let right = right.convert_to(&CType::Long)?;
                Binary(BinaryExpression { left, right, ..binary }.into()).set_type(&left_t)
            } else if right_t.is_pointer_to_complete() && left_t.is_int() {
                // If right is pointer, we swap left/right, as tac expects the ptr to be on the left
                assert_or_err(
                    !binary.is_assignment,
                    "Invalid assignment types",
                    "Adding int to pointer gives pointer, not int",
                )?;
                let left = left.convert_to(&CType::Long)?;
                Binary(BinaryExpression { left: right, right: left, ..binary }.into())
                    .set_type(&right_t)
            } else {
                Err(Error::new("Invalid operand", "Illegal operands for addition"))
            }
        }
        Subtract => {
            let left_t = left.get_type();
            let right_t = right.get_type();
            if left_t.is_arithmetic() && right_t.is_arithmetic() {
                type_check_arithmetic_binexpr(left, right, binary.operator, binary.is_assignment)
            } else if left_t.is_pointer_to_complete() && right_t.is_int() {
                assert_or_err(
                    !binary.is_assignment || left.is_lvalue(),
                    "Invalid assignment",
                    "Can only assign to lvalue",
                )?;
                // Subtraction with pointer/int is addition with negated value
                let right = right.convert_to(&CType::Long)?;
                let right = Expression::Unary(UnaryOperator::Negate, right.into())
                    .set_type(&CType::Long)?;
                Binary(BinaryExpression { left, right, operator: Add, ..binary }.into())
                    .set_type(&left_t)
            } else if left_t.is_pointer_to_complete() && left_t == right_t {
                assert_or_err(
                    !binary.is_assignment,
                    "Invalid assignment types",
                    "Subtracting pointers gives int, not pointer!",
                )?;
                Binary(BinaryExpression { left, right, ..binary }.into()).set_type(&CType::Long)
            } else {
                Err(Error::new("Invalid operand", "Illegal operands for subtraction"))
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
    if is_assignment {
        assert_or_err(left.is_lvalue(), "Invalid assignemnt", "Can only assign to lvalue!")?;
        let right = right.convert_to(&common_type)?;
        let left_t = left.get_type();
        let binexpr = BinaryExpression { left, right, operator, is_assignment };
        Expression::Binary(binexpr.into()).set_type(&left_t)
    } else {
        let left = left.convert_to(&common_type)?;
        let right = right.convert_to(&common_type)?;
        let binexpr = BinaryExpression { left, right, operator, is_assignment };
        Expression::Binary(binexpr.into()).set_type(&common_type)
    }
}

fn type_check_function(
    function: FunctionDeclaration,
    table: &mut TypeTable,
) -> Result<FunctionDeclaration> {
    let function = type_check_params(function)?;
    let ctype = &function.ctype;
    let mut defined = function.body.is_some();
    let mut global = function.storage != Some(StorageClass::Static);
    if let Some(symbol) = table.symbols.get(&function.name) {
        let SymbolAttr::Function { defined: prev_def, global: prev_global } = symbol.attrs else {
            return Err(Error::new(
                "Invalid declaration",
                "Previous declaration isn't a function!",
            ));
        };
        assert_or_err(
            symbol.ctype == *ctype,
            "Invalid declaration",
            "Incompatible function declarations!",
        )?;
        assert_or_err(!prev_def || !defined, "Invalid declaration", "Function redefinition!")?;
        assert_or_err(
            !prev_global || global,
            "Invalid declaration",
            "Static function declaration cannot follow non-static",
        )?;
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

fn type_check_params(mut function: FunctionDeclaration) -> Result<FunctionDeclaration> {
    let CType::Function(param_types, return_t) = function.ctype else { panic!("Not a function!") };
    assert_or_err(
        !matches!(*return_t, CType::Array(_, _)),
        "Invalid return type",
        "Function cannot return array",
    )?;
    let adjusted_params = param_types
        .into_iter()
        .map(|param_t| {
            param_t.validate_specifier()?;
            match param_t {
                CType::Array(elem_t, _) => Ok(CType::Pointer(elem_t)),
                CType::Void => {
                    Err(Error::new("Illegal Declaration", "Cannot declare void parameters"))
                }
                _ => Ok(param_t),
            }
        })
        .collect::<Result<Vec<CType>>>()?;
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
                    return Err(Error::new(
                        "Initialization error",
                        "Cannot initialize static array with a scalar!",
                    ));
                };
                assert_or_err(
                    elem_t.is_char(),
                    "Initialization error",
                    "Cannot initialize a non-char with a string!",
                )?;
                assert_or_err(
                    s_data.len() as u64 <= *size,
                    "Initialization error",
                    "Too many characters in string!",
                )?;
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
                    "Initialization error",
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
                    Err(_) => {
                        return Err(Error::new(
                            "Initialization parse error",
                            "Failed to parse static initializer",
                        ));
                    }
                };
                Ok(StaticInitializer::Initialized(vec![init_val]))
            }
        }
        Some(VariableInitializer::CompoundInit(init_list)) => {
            let CType::Array(elem_t, size) = ctype else {
                return Err(Error::new(
                    "Initialization error",
                    "Cannot initiate static scalar with initializer list",
                ));
            };
            assert_or_err(
                init_list.len() as u64 <= *size,
                "Initialization error",
                "Initializer longer than array!",
            )?;
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
        CType::Function(_, _)
        | CType::Pointer(_)
        | CType::Array(_, _)
        | CType::VarArgs
        | CType::Void
        | CType::Structure(_) => {
            panic!("Not a variable!")
        }
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
        CType::Function(_, _) | CType::VarArgs | CType::Void | CType::Structure(_) => {
            panic!("Cannot zero initialize a function")
        }
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
                    return Err(Error::new(
                        "Initialization error",
                        "Cannot initialize array with a scalar!",
                    ));
                };
                assert_or_err(
                    elem_t.is_char(),
                    "Initialization error",
                    "Cannot initialize a non-char type with a string!",
                )?;
                assert_or_err(
                    data.len() as u64 <= *size,
                    "Initialization error",
                    "Too many characters in strign literal!",
                )?;
                Ok(SingleElem(expr.set_type(target_t)?))
            } else {
                let expr = type_check_and_convert(expr, table)?.convert_by_assignment(target_t)?;
                Ok(SingleElem(expr))
            }
        }
        CompoundInit(init_list) => {
            let CType::Array(elem_t, size) = target_t else {
                return Err(Error::new(
                    "Initialization error",
                    "Cannot initialize a scalar with a compound initializer!",
                ));
            };
            assert_or_err(
                init_list.len() as u64 <= *size,
                "Initialization error",
                "Too many items in initializer!",
            )?;
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
            assert_or_err(
                var.init.is_none(),
                "Illegal initializer",
                "Local extern variable cannot have initializer!",
            )?;
            if let Some(symbol) = table.symbols.get(&var.name) {
                assert_or_err(
                    symbol.ctype == var.ctype,
                    "Invalid declaration",
                    "Declaration types don't match!",
                )?;
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
                    assert_or_err(
                        **elem_t == CType::Char,
                        "Initialization error",
                        "Can only init char* with string!",
                    )?;
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
            return Err(Error::new(
                "Invalid declaration",
                "Expected prev declaration to be static variable!",
            ));
        };
        assert_or_err(
            symbol.ctype == var.ctype,
            "Invalid declaration",
            "Declaration types don't match!",
        )?;
        if var.storage == Some(StorageClass::Extern) {
            global = *prev_global;
        } else if global != *prev_global {
            return Err(Error::new(
                "Invalid declaration",
                "Declarations have conflicting linkage!",
            ));
        }
        if matches!(prev_init, Initialized(_)) {
            assert_or_err(
                !matches!(init_value, Initialized(_)),
                "Invalid declaration",
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
