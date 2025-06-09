use std::process::exit;

use clap::Parser;
use compiler::{Args, run_main};

fn main() {
    let args = Args::parse();
    match run_main(args) {
        Err(error) => {
            match error.exit_code {
                1 => println!("GCC Preprocessing Failed!\n"),
                2 => println!("Lex Error: "),
                3 => println!("Parse Error: "),
                4 => println!("Semantic Error: "),
                5 => println!("Type Error: "),
                6 => println!("Code Emission Failed!\n"),
                7 => println!("GCC Assembler Failed!\n"),
                8 => println!("GCC Assemble and Link Failed!\n"),
                _ => println!("Unknown error!"),
            }
            println!("\t{}", error.message);
            exit(error.exit_code as i32);
        }
        Ok(_) => (),
    }
}
