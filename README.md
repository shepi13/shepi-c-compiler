# Simple C Compiler written in Rust.

![Build Status](https://github.com/shepi13/shepi-c-compiler/actions/workflows/rust.yml/badge.svg?event=push)
-------

Following the guide in *Writing a C Compiler* by Nora Sandler

## Still in development, and mainly just a personal project to learn rust and compiler design.

### Currently supports:

- Variables
- Loops
- Functions
- If, Conditionals, and Switch Statements
- Gotos and Labels
- Arithmetic/Relational/Bitwise Operators (with short circuting && and || as well as precedence climbing)
- Compound assignment, Increment and Decrement operators
- Static and Extern variables
- Signed/Unsigned integer types with proper C conversions and casting, including char, long, and long long types
- Double Precision floating point operations (Including proper NaN support)
- Basic pointer operations (referencing and dereferencing)
- Arrays and Pointer Arithmetic 
	- Includes Pointer addition, subtraction, incr/decrement, and subscript access
	- Simple initialization lists for both static and local variables
	- VLAs not supported
- String Literals and Character constants (including both static strings and strings as array intializers)
- Variadic functions
- Void type, sizeof, and dynamic memory allocation
----------------

### In progress:

- Structs/Unions
- Enums
- Optimizations

------------------

### Longer term:

I'm aiming to add some final features of the C language that aren't covered in the book, but these seem less trivial:

- Typedefs
- Function pointers
- Full constexpr and static initialization evaluation (current support is very simplistic)

In addition, this project should have proper tests and benchmarking (both for compile times and program runtimes). For now I am just using the tests from the book.

Current CI is somewhat lacking as well, as it justs checks that the build succeeds and requires running the book tests locally, although this should at least catch some bugs.
