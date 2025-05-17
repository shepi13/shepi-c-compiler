mod assembly;
mod emission;
mod generator;
mod lexer;
mod parser;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use clap::Parser;
use parser::SymbolTable;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to compile
    filename: String,

    /// Only run the lexer
    #[arg(short, long)]
    lex: bool,

    /// Run the parser and lexer
    #[arg(short, long)]
    parse: bool,
    #[arg(short, long)]
    validate: bool,

    /// Run the lexer, parser, and TAC generation
    #[arg(short, long)]
    tacky: bool,

    /// Run the lexer, parser, and all code gen, but don't output assembly
    #[arg(short = 'g', long)]
    codegen: bool,

    /// Output assembly, but don't run linker
    #[arg(short = 'S')]
    assembler_only: bool,
}

fn path_with_extension(filename: &str, extension: &str) -> String {
    let mut file = PathBuf::from(&filename);
    file.set_extension(&extension);
    match file.to_str() {
        Some(val) => val.to_string(),
        None => {
            panic!("Failed to convert filename to str!");
        }
    }
}

fn main() {
    let args = Args::parse();

    let preprocess_file = path_with_extension(&args.filename, "i");
    let assembly_file = path_with_extension(&args.filename, "s");
    let output_file = path_with_extension(&args.filename, "");

    // Run preprocessor
    let status = Command::new("gcc")
        .args(["-E", "-P", &args.filename, "-o", &preprocess_file])
        .status()
        .expect("Preprocessor failed to run!");
    if !status.success() {
        panic!("Preprocessor exited with failure");
    }

    // Run Lexer
    let program = fs::read_to_string(&preprocess_file).expect("Failed to read preprocessed code!");
    let tokens = lexer::parse(&program);

    if args.lex {
        println!("Tokens:\n\n {:#?}", tokens);
        return;
    }

    // Run Parser
    let mut symbol_table = SymbolTable::new();
    let parser_ast = parser::parse(&mut &tokens[..], &mut symbol_table);

    if args.parse || args.validate {
        println!("Parser AST: {:#?}", parser_ast);
        return;
    }

    // Run TAC Generation
    let tac_ast = generator::gen_tac_ast(&parser_ast);

    if args.tacky {
        println!("Tacky AST: {:#?}", tac_ast);
        return;
    }

    // Run Full codegen
    let mut stack_map = assembly::StackGen::new();
    let mut assembly_ast = assembly::gen_assembly_tree(tac_ast, &mut stack_map);
    assembly::add_stack_offset(&mut assembly_ast, stack_map);

    if args.codegen {
        // Print generated code?
        println!("Assembly AST: {:#?}", assembly_ast);
        return;
    }

    // Code emission
    emission::emit_program(&assembly_file, assembly_ast);

    if args.assembler_only {
        // Print assembly?
        return;
    }

    // Run linker
    let status = Command::new("gcc")
        .args([&assembly_file, "-o", &output_file])
        .status()
        .expect("Linker failed to run!");
    if !status.success() {
        panic!("Linker exited with failure");
    }
}
