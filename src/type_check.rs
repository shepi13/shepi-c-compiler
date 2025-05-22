use std::{
    collections::{HashMap, HashSet},
    iter::zip,
};

use crate::parser::{
    AssignmentExpression, BinaryExpression, BinaryOperator, Block, BlockItem, CType,
    ConditionExpression, Constant, Declaration, Expression, ForInit, FunctionDeclaration, Program,
    Statement, StorageClass, TypedExpression, UnaryOperator, VariableDeclaration,
};

#[derive(Debug)]
struct TypeTable {
    symbols: Symbols,
    current_function: Option<String>,
}

pub type Symbols = HashMap<String, Symbol>;
#[derive(Debug)]
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

#[derive(Debug, PartialEq, Eq)]
pub enum SymbolAttr {
    Function(FunctionAttributes),
    Static(StaticAttributes),
    Local,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionAttributes {
    pub defined: bool,
    pub global: bool,
}
#[derive(Debug, PartialEq, Eq)]
pub struct StaticAttributes {
    pub init: StaticInitializer,
    pub global: bool,
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StaticInitializer {
    Tentative,
    Initialized(Initializer),
    None,
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Initializer {
    Int(i64), // Limited to i32, but we store as i64 for matching and do our own conversion
    Long(i64),
}
impl Initializer {
    pub fn value(&self) -> i64 {
        match self {
            Self::Int(val) | Self::Long(val) => *val
        }
    }
}

#[derive(Debug)]
pub struct TypedProgram {
    pub program: Program,
    pub symbols: Symbols,
}

pub fn eval_constant_expr(expr: &TypedExpression, ctype: &CType) -> Constant {
    match &expr.expr {
        Expression::Constant(constexpr) => {
            let val = constexpr.value();
            match ctype {
                CType::Int => Constant::Int(val & 0xFFFFFFFF),
                CType::Long => Constant::Long(val),
                _ => panic!("Invalid conversion type!"),
            }
        }
        Expression::Unary(UnaryOperator::Negate, constexpr) => {
            eval_constant_expr(&constexpr, ctype)
        }
        _ => panic!("Expected constant expression!"),
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
pub fn get_common_type(left_type: &CType, right_type: &CType) -> CType {
    if left_type == right_type {
        left_type.clone()
    } else {
        CType::Long
    }
}
fn convert_to(expression: TypedExpression, ctype: &CType) -> TypedExpression {
    if get_type(&expression) == *ctype {
        expression
    } else {
        set_type(
            Expression::Cast(ctype.clone(), expression.into()).into(),
            ctype,
        )
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
    block
        .into_iter()
        .map(|item| type_check_block_item(item, table))
        .collect()
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
            let expr = type_check_expression(expr, table);
            let cur_func = table
                .current_function
                .as_ref()
                .expect("Return must be inside function!");
            let cur_func_type = &table.symbols[cur_func].ctype;
            if let CType::Function(_, return_type) = cur_func_type {
                let expr = convert_to(expr, &return_type);
                Return(expr)
            } else {
                panic!("Failed to match function type!")
            }
        }
        ExprStmt(expr) => ExprStmt(type_check_expression(expr, table)),
        Label(name, statement) => Label(name, type_check_statement(*statement, table).into()),
        If(expr, if_true, if_false) => {
            let expr = type_check_expression(expr, table);
            let if_true = type_check_statement(*if_true, table);
            let if_false = if_false.map(|statement| type_check_statement(statement, table));
            If(expr, if_true.into(), if_false.into())
        }
        While(mut loop_data) => {
            loop_data.condition = type_check_expression(loop_data.condition, table);
            loop_data.body = type_check_statement(*loop_data.body, table).into();
            While(loop_data)
        }
        DoWhile(mut loop_data) => {
            loop_data.condition = type_check_expression(loop_data.condition, table);
            loop_data.body = type_check_statement(*loop_data.body, table).into();
            DoWhile(loop_data)
        }
        Compound(block) => Compound(type_check_block(block, table)),
        Switch(mut switch) => {
            switch.condition = type_check_expression(switch.condition, table);
            let cond_type = get_type(&switch.condition);
            println!("Condition type: {:#?}", cond_type);
            let mut new_cases = Vec::new();
            let mut case_vals = HashSet::new();
            for case in switch.cases {
                let constexpr = eval_constant_expr(&case.1, &cond_type);
                let val = constexpr.value();
                assert!(!case_vals.contains(&val), "Duplicate case!");
                case_vals.insert(val);
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
                    ForInit::Expr(Some(type_check_expression(expr, table)))
                }
                _ => ForInit::Expr(None),
            };
            loop_data.condition = type_check_expression(loop_data.condition, table);
            let post = post.map(|post_expr| type_check_expression(post_expr, table));
            loop_data.body = type_check_statement(*loop_data.body, table).into();
            For(init, loop_data, post)
        }
        Case(_, _) | Default(_) => {
            panic!("Should be re-written into label in switch resolve pass!")
        }
        Break(_) | Continue(_) | Goto(_) | Null => statement,
    }
}

fn type_check_expression(expr: TypedExpression, table: &mut TypeTable) -> TypedExpression {
    match expr.expr {
        Expression::Variable(name) => {
            let var_type = &table.symbols[&name].ctype;
            assert!(
                !matches!(var_type, CType::Function(_, _)),
                "Function name used as variable!"
            );
            set_type(Expression::Variable(name).into(), var_type)
        }
        Expression::Constant(constant) => match constant {
            Constant::Int(_) => set_type(Expression::Constant(constant).into(), &CType::Int),
            Constant::Long(_) => set_type(Expression::Constant(constant).into(), &CType::Long),
        },
        Expression::Cast(new_type, inner) => {
            let typed_inner = type_check_expression(*inner, table);
            let cast_expr = Expression::Cast(new_type.clone(), typed_inner.into()).into();
            set_type(cast_expr, &new_type)
        }
        Expression::Unary(op, inner) => {
            let typed_inner = type_check_expression(*inner, table);
            let new_type = get_type(&typed_inner);
            let unary_expr = Expression::Unary(op.clone(), typed_inner.into()).into();
            match op {
                UnaryOperator::LogicalNot => set_type(unary_expr, &CType::Int),
                _ => set_type(unary_expr, &new_type),
            }
        }
        Expression::Binary(binary) => {
            use BinaryOperator::*;
            use Expression::Binary;
            let left = type_check_expression(binary.left, table);
            let right = type_check_expression(binary.right, table);
            if matches!(
                binary.operator,
                BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr
            ) {
                let binexpr = Binary(
                    BinaryExpression {
                        left,
                        right,
                        ..*binary
                    }
                    .into(),
                );
                set_type(binexpr.into(), &CType::Int)
            } else if matches!(
                binary.operator,
                BinaryOperator::LeftShift | BinaryOperator::RightShift
            ) {
                let left_type = &get_type(&left);
                let binexpr = Binary(
                    BinaryExpression {
                        left,
                        right,
                        ..*binary
                    }
                    .into(),
                );
                set_type(binexpr.into(), left_type)
            } else {
                let common_type = get_common_type(&get_type(&left), &get_type(&right));
                let left = convert_to(left, &common_type);
                let right = convert_to(right, &common_type);
                let binexpr = Binary(
                    BinaryExpression {
                        left,
                        right,
                        operator: binary.operator.clone(),
                        ..*binary
                    }
                    .into(),
                );
                match binary.operator {
                    Add | Subtract | Multiply | Divide | Remainder | BitAnd | BitOr | BitXor
                    | LeftShift | RightShift => set_type(binexpr.into(), &common_type),
                    _ => set_type(binexpr.into(), &CType::Int),
                }
            }
        }
        Expression::Assignment(assign) => {
            let left = type_check_expression(assign.left, table);
            let right = type_check_expression(assign.right, table);
            let left_type = &get_type(&left);
            let right = convert_to(right, left_type);
            let assign = Expression::Assignment(AssignmentExpression { left, right }.into());
            set_type(assign.into(), left_type)
        }
        Expression::Condition(cond) => {
            let condition = type_check_expression(cond.condition, table);
            let if_true = type_check_expression(cond.if_true, table);
            let if_false = type_check_expression(cond.if_false, table);
            let common_type = get_common_type(&get_type(&if_true), &get_type(&if_false));
            let if_true = convert_to(if_true, &common_type);
            let if_false = convert_to(if_false, &common_type);
            let condition_expr = Expression::Condition(
                ConditionExpression {
                    condition,
                    if_true,
                    if_false,
                }
                .into(),
            );
            set_type(condition_expr.into(), &common_type)
        }
        Expression::FunctionCall(name, args) => {
            let expected_type = table.symbols[&name].ctype.clone();
            match expected_type {
                CType::Function(arg_types, ret_type) => {
                    assert!(
                        arg_types.len() == args.len(),
                        "Incorrect number of arguments!"
                    );
                    let mut converted_args = Vec::new();
                    for (arg, arg_type) in zip(args, arg_types) {
                        let typed_arg = type_check_expression(arg, table);
                        converted_args.push(convert_to(typed_arg, &arg_type));
                    }
                    let call_expr = Expression::FunctionCall(name, converted_args);
                    set_type(call_expr.into(), &ret_type)
                }
                _ => panic!("Variable used as function!"),
            }
        }
    }
}

// Type Checking

fn type_check_function(
    function: FunctionDeclaration,
    table: &mut TypeTable,
) -> FunctionDeclaration {
    //let ctype = CType::Function(function.params.len());
    table.current_function = Some(function.name.clone());
    let ctype = &function.ctype;
    let mut defined = function.body.is_some();
    let mut global = function.storage != Some(StorageClass::Static);
    if let Some(symbol) = table.symbols.get(&function.name) {
        let attrs = symbol.get_function_attrs();
        assert!(
            symbol.ctype == *ctype,
            "Incompatible function declarations!"
        );
        assert!(!attrs.defined || !defined, "Function redefinition!");
        assert!(
            !attrs.global || global,
            "static definition cannot follow non-static"
        );
        global = attrs.global;
        defined = attrs.defined;
    }
    table.symbols.insert(
        function.name.clone(),
        Symbol {
            ctype: ctype.clone(),
            attrs: SymbolAttr::Function(FunctionAttributes { defined, global }),
        },
    );
    if let Some(body) = function.body {
        let param_types = match ctype {
            CType::Function(args, _) => args,
            _ => panic!("Not a function!"),
        };
        for (param, param_type) in zip(&function.params, param_types) {
            table.symbols.insert(
                param.clone(),
                Symbol {
                    ctype: param_type.clone(),
                    attrs: SymbolAttr::Local,
                },
            );
        }
        FunctionDeclaration {
            body: Some(type_check_block(body, table)),
            ..function
        }
    } else {
        function
    }
}

fn parse_static_initializer(init: &Option<TypedExpression>, ctype: &CType) -> StaticInitializer {
    let init_val = init.as_ref().map(|expr| eval_constant_expr(expr, ctype));
    let val = init_val.map_or(0, |constexpr| constexpr.value());
    match ctype {
        CType::Int => StaticInitializer::Initialized(Initializer::Int(val)),
        CType::Long => StaticInitializer::Initialized(Initializer::Long(val)),
        _ => panic!("Invalid static initializer!"),
    }
}

fn type_check_var_declaration(
    var: VariableDeclaration,
    table: &mut TypeTable,
) -> VariableDeclaration {
    match var.storage {
        Some(StorageClass::Extern) => {
            assert!(
                var.value.is_none(),
                "Local extern variable cannot have initializer!"
            );
            if let Some(symbol) = table.symbols.get(&var.name) {
                assert!(symbol.ctype == var.ctype, "Declaration types don't match!");
            } else {
                table.symbols.insert(
                    var.name.clone(),
                    Symbol {
                        ctype: var.ctype.clone(),
                        attrs: SymbolAttr::Static(StaticAttributes {
                            init: StaticInitializer::None,
                            global: true,
                        }),
                    },
                );
            }
            var
        }
        Some(StorageClass::Static) => {
            let init_value = parse_static_initializer(&var.value, &var.ctype);
            table.symbols.insert(
                var.name.clone(),
                Symbol {
                    ctype: var.ctype.clone(),
                    attrs: SymbolAttr::Static(StaticAttributes {
                        init: init_value,
                        global: false,
                    }),
                },
            );
            var
        }
        _ => {
            table.symbols.insert(
                var.name.clone(),
                Symbol {
                    ctype: var.ctype.clone(),
                    attrs: SymbolAttr::Local,
                },
            );
            let new_init = var.value.map(|init| type_check_expression(init, table));
            let new_init = new_init.map(|init| convert_to(init, &var.ctype));
            VariableDeclaration {
                value: new_init,
                ..var
            }
        }
    }
}

fn type_check_var_filescope(var: &VariableDeclaration, table: &mut TypeTable) {
    let mut init_value = if var.value.is_none() {
        if var.storage == Some(StorageClass::Extern) {
            StaticInitializer::None
        } else {
            StaticInitializer::Tentative
        }
    } else {
        parse_static_initializer(&var.value, &var.ctype)
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

        if matches!(attrs.init, StaticInitializer::Initialized(_)) {
            assert!(
                !matches!(init_value, StaticInitializer::Initialized(_)),
                "Conflicting static initializers!"
            );
            init_value = attrs.init.clone();
        } else if attrs.init == StaticInitializer::Tentative
            && !matches!(init_value, StaticInitializer::Initialized(_))
        {
            init_value = StaticInitializer::Tentative
        }
    }
    table.symbols.insert(
        var.name.clone(),
        Symbol {
            ctype: var.ctype.clone(),
            attrs: SymbolAttr::Static(StaticAttributes {
                init: init_value,
                global,
            }),
        },
    );
}
