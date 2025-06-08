mod codegen;
mod parse;
mod tac_generation;
mod validate;

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, exit};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to compile
    filename: String,

    ///Libraries to link
    #[arg(short = 'l')]
    libraries: Vec<String>,

    #[arg(short = 'L')]
    library_paths: Vec<String>,

    #[arg(long)]
    print_commands: bool,

    /// Only run the lexer
    #[arg(short, long)]
    lex: bool,

    /// Run the parser and lexer
    #[arg(short, long)]
    parse: bool,

    #[arg(long)]
    semantics: bool,

    /// Run the parser, lexer, and type checking
    #[arg(short, long)]
    validate: bool,

    /// Run the lexer, parser, and TAC generation
    #[arg(short, long)]
    tacky: bool,

    /// Run assembly generation but without rewrites
    #[arg(long)]
    no_rewrite: bool,

    /// Run the lexer, parser, and all code gen, but don't output assembly
    #[arg(short = 'g', long)]
    codegen: bool,

    /// Output assembly, but don't run linker
    #[arg(short = 'S')]
    assembler_only: bool,

    /// Output assembly and run assembler to generate object file, but not linker,
    #[arg(short = 'c')]
    compile_only: bool,
}

fn path_with_extension(filename: &str, extension: &str) -> String {
    let mut file = PathBuf::from(&filename);
    file.set_extension(extension);
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
    let object_file = path_with_extension(&args.filename, "o");
    let output_file = path_with_extension(&args.filename, "");

    // Run preprocessor
    let mut preprocessor = Command::new("gcc");
    preprocessor.args(["-E", &args.filename, "-o", &preprocess_file]);
    if args.print_commands {
        println!("Running preprocessor: {:?}", preprocessor);
    }
    let status = preprocessor.status().expect("Preprocessor failed to run!");
    if !status.success() {
        panic!("Preprocessor exited with failure");
    }
    // Run Lexer
    let program_source =
        fs::read_to_string(&preprocess_file).expect("Failed to read preprocessed code!");
    let mut tokens = match parse::lexer::parse(&program_source) {
        Ok(tokens) => tokens,
        Err(lex_error) => {
            eprintln!("Lexing failed:\n\t{}", lex_error);
            exit(1);
        }
    };
    if args.lex {
        println!("Tokens:\n\n{}", tokens);
        return;
    }
    // Run Parser
    let parser_ast = match parse::parser::parse(&mut tokens) {
        Ok(program) => program,
        Err(parse_error) => {
            eprintln!("Parsing failed:\n\t{}", parse_error);
            exit(2);
        }
    };
    if args.parse {
        println!("Parser AST: {:#?}", parser_ast);
        return;
    }
    // Run Semantics Analysis Pass
    let resolved_ast = match validate::semantics::resolve_program(parser_ast) {
        Ok(program) => program,
        Err(semantic_error) => {
            eprintln!("Semantic Error: \n\t{}", semantic_error.error_message(&program_source));
            exit(3);
        }
    };
    if args.semantics {
        println!("Resolved AST: {:#?}", resolved_ast);
        return;
    }
    // Run type checking
    let mut typed_program = match validate::type_check::type_check_program(resolved_ast) {
        Ok(typed_program) => typed_program,
        Err(type_error) => {
            eprintln!("Type Error: \n\t{}", type_error.error_message(&program_source));
            exit(4);
        }
    };
    if args.validate {
        println!("Resolved AST: {:#?}", typed_program.program);
        println!("Symbols: {:#?}", typed_program.symbols);
        return;
    }

    // Run TAC Generation
    let tac_ast =
        tac_generation::generator::gen_tac_ast(typed_program.program, &mut typed_program.symbols);
    if args.tacky {
        println!("Tacky AST: {:#?}", tac_ast);
        println!("Tacky symbols: {:#?}", typed_program.symbols);
        return;
    }
    // Run Full codegen
    let assembly_ast = codegen::assembly_gen::gen_assembly_tree(tac_ast, typed_program.symbols);
    if args.no_rewrite {
        println!("Assembly AST: {:#?}", assembly_ast);
        return;
    }
    //Rewrite instructions
    let assembly_ast = codegen::assembly_rewrite::rewrite_assembly(assembly_ast);
    if args.codegen {
        // Print generated code?
        println!("Assembly AST: {:#?}", assembly_ast);
        return;
    }
    // Code emission
    let mut asm_code = String::new();
    match codegen::emission::emit_program(&mut asm_code, assembly_ast) {
        Ok(_) => (),
        Err(error) => {
            println!("Emission failed: {}", error);
            exit(8);
        }
    }
    let mut file = File::create(&assembly_file).expect("Failed to create file!");
    write!(file, "{}", asm_code).expect("Failed to write to file!");
    if args.assembler_only {
        print!("{}", asm_code);
        return;
    }
    // Compile and Link
    if args.compile_only {
        let mut assembler = Command::new("gcc");
        assembler.args([&assembly_file, "-c", "-o", &object_file]);
        if args.print_commands {
            println!("Running assembler: {:?}", assembler);
        }
        let status = assembler.status().expect("Assembler failed to run!");
        if !status.success() {
            panic!("Assembler exited with error!");
        }
    } else {
        // Run linker
        let linker_args = [
            vec![assembly_file, String::from("-o"), output_file],
            args.libraries.iter().map(|lib| format!("-l{}", &lib)).collect(),
            args.library_paths.iter().map(|path| format!("-L{}", &path)).collect(),
        ]
        .concat();
        let mut linker = Command::new("gcc");
        linker.args(linker_args);
        if args.print_commands {
            println!("Running linker: {:?}", linker);
        }
        let status = linker.status().expect("Assembly and Linking failed to run!");
        if !status.success() {
            panic!("Assembler/Linker exited with failure");
        }
    }
}
