use std::collections::HashMap;

use crate::{
    parser::{
        Block, BlockItem, CType, Declaration, Expression, ForInit, FunctionDeclaration, Program, Statement, StorageClass, TypedExpression, VariableDeclaration
    },
    semantics::eval_constant_expr,
};

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

#[derive(Debug)]
pub enum SymbolAttr {
    Function(FunctionAttributes),
    Static(StaticAttributes),
    Local,
}

#[derive(Debug)]
pub struct FunctionAttributes {
    pub defined: bool,
    pub global: bool,
}
#[derive(Debug)]
pub struct StaticAttributes {
    pub init: StaticInitializer,
    pub global: bool,
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StaticInitializer {
    Tentative,
    Initialized(i32),
    None,
}

// Tree traversal functions
pub fn type_check_program(program: &Program) -> Symbols {
    let mut symbols = HashMap::new();
    for decl in program {
        type_check_declaration(decl, &mut symbols, true);
    }
    symbols
}

fn type_check_declaration(decl: &Declaration, symbols: &mut Symbols, global: bool) {
    match decl {
        Declaration::Function(func) => type_check_function(func, symbols),
        Declaration::Variable(var) => {
            if global {
                type_check_var_filescope(var, symbols);
            } else {
                type_check_var_declaration(var, symbols);
            }
        }
    }
}

fn type_check_block(block: &Block, symbols: &mut Symbols) {
    for block_item in block {
        type_check_block_item(block_item, symbols);
    }
}
fn type_check_block_item(block_item: &BlockItem, symbols: &mut Symbols) {
    match block_item {
        BlockItem::StatementItem(statement) => type_check_statement(statement, symbols),
        BlockItem::DeclareItem(decl) => type_check_declaration(decl, symbols, false),
    }
}
fn type_check_statement(statement: &Statement, symbols: &mut Symbols) {
    match statement {
        Statement::Return(expr) | Statement::ExprStmt(expr) => {
            type_check_expression(expr, symbols);
        }
        Statement::Label(_, statement)
        | Statement::Default(statement)
        | Statement::Case(_, statement) => {
            type_check_statement(statement, symbols);
        }
        Statement::If(expr, if_true, if_false) => {
            type_check_expression(expr, symbols);
            type_check_statement(if_true, symbols);
            if let Some(false_statement) = &**if_false {
                type_check_statement(false_statement, symbols);
            }
        }
        Statement::While(loop_data) | Statement::DoWhile(loop_data) => {
            type_check_expression(&loop_data.condition, symbols);
            type_check_statement(&loop_data.body, symbols);
        }
        Statement::Compound(block) => {
            type_check_block(block, symbols);
        }
        Statement::Switch(switch) => {
            type_check_expression(&switch.condition, symbols);
            type_check_statement(&switch.statement, symbols)
        }
        Statement::For(init, loop_data, post) => {
            match init {
                ForInit::Decl(decl) => type_check_var_declaration(decl, symbols),
                ForInit::Expr(Some(expr)) => type_check_expression(expr, symbols),
                _ => (),
            }
            type_check_expression(&loop_data.condition, symbols);
            if let Some(post_expr) = post {
                type_check_expression(post_expr, symbols);
            }
            type_check_statement(&loop_data.body, symbols);
        }
        Statement::Break(_) | Statement::Continue(_) | Statement::Goto(_) | Statement::Null => (),
    }
}

fn type_check_expression(expr: &TypedExpression, symbols: &mut Symbols) {
    match &expr.expr {
        Expression::Unary(_, expr) => type_check_expression(expr, symbols),
        Expression::Variable(name) => type_check_variable(&name, symbols),
        Expression::FunctionCall(name, args) => type_check_call(&name, args, symbols),
        Expression::Binary(exprs) => {
            type_check_expression(&exprs.left, symbols);
            type_check_expression(&exprs.right, symbols);
        }
        Expression::Assignment(exprs) => {
            type_check_expression(&exprs.left, symbols);
            type_check_expression(&exprs.right, symbols);
        }
        Expression::Condition(cond) => {
            type_check_expression(&cond.condition, symbols);
            type_check_expression(&cond.if_true, symbols);
            type_check_expression(&cond.if_false, symbols);
        }
        Expression::Constant(_) => (),
        Expression::Cast(_, _) => panic!("Not implemented!"),
    }
}

// Type Checking

fn type_check_function(function: &FunctionDeclaration, symbols: &mut Symbols) {
    //let ctype = CType::Function(function.params.len());
    let ctype = &function.ctype;
    let mut defined = function.body.is_some();
    let mut global = function.storage != Some(StorageClass::Static);
    if let Some(symbol) = symbols.get(&function.name) {
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
    symbols.insert(
        function.name.clone(),
        Symbol {
            ctype: ctype.clone(),
            attrs: SymbolAttr::Function(FunctionAttributes { defined, global }),
        },
    );
    if let Some(body) = &function.body {
        for param in &function.params {
            symbols.insert(
                param.clone(),
                Symbol {
                    ctype: CType::Int,
                    attrs: SymbolAttr::Local,
                },
            );
        }
        type_check_block(body, symbols);
    }
}

fn type_check_call(name: &str, args: &Vec<TypedExpression>, symbols: &mut Symbols) {
    for arg in args {
        type_check_expression(arg, symbols);
    }
    if let CType::Function(param_types, return_type) = &symbols[name].ctype {
        assert!(
            args.len() == param_types.len(),
            "Incorrect number of arguments"
        );
    } else {
        panic!("Not a function!");
    }
}

fn type_check_var_declaration(var: &VariableDeclaration, symbols: &mut Symbols) {
    match var.storage {
        Some(StorageClass::Extern) => {
            assert!(
                var.value.is_none(),
                "Local extern variable cannot have initializer!"
            );
            if let Some(symbol) = symbols.get(&var.name) {
                assert!(
                    symbol.ctype == CType::Int,
                    "Function redeclared as variable!"
                );
            } else {
                symbols.insert(
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
        }
        Some(StorageClass::Static) => {
            let init_value = if var.value.is_none() {
                StaticInitializer::Initialized(0)
            } else {
                StaticInitializer::Initialized(eval_constant_expr(var.value.as_ref().unwrap()))
            };
            symbols.insert(
                var.name.clone(),
                Symbol {
                    ctype: var.ctype.clone(),
                    attrs: SymbolAttr::Static(StaticAttributes {
                        init: init_value,
                        global: false,
                    }),
                },
            );
        }
        _ => {
            symbols.insert(
                var.name.clone(),
                Symbol {
                    ctype: var.ctype.clone(),
                    attrs: SymbolAttr::Local,
                },
            );
            if let Some(init) = &var.value {
                type_check_expression(init, symbols);
            }
        }
    }
}

fn type_check_var_filescope(var: &VariableDeclaration, symbols: &mut Symbols) {
    let mut init_value = if var.value.is_none() {
        if var.storage == Some(StorageClass::Extern) {
            StaticInitializer::None
        } else {
            StaticInitializer::Tentative
        }
    } else {
        StaticInitializer::Initialized(eval_constant_expr(var.value.as_ref().unwrap()))
    };

    let mut global = var.storage != Some(StorageClass::Static);
    if let Some(symbol) = symbols.get(&var.name) {
        let attrs = symbol.get_static_attrs();
        assert!(
            symbol.ctype == CType::Int,
            "Function redeclared as variable!"
        );
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
    symbols.insert(
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

fn type_check_variable(name: &str, symbols: &mut Symbols) {
    assert!(
        symbols[name].ctype == CType::Int,
        "Found function, not variable!"
    );
}
