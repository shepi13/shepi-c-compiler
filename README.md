# Simple C Compiler written in Rust.

![Build Status](https://github.com/shepi13/shepi-c-compiler/actions/workflows/rust.yml/badge.svg?event=push)
-------

Following the guide in Writing a C Compiler by Nora Sandler

### Still in development, and mainly just a personal project to learn rust and compiler design.

#####Currently supports:

- Variables
- Loops
- Functions
- If, Conditionals, and Switch Statements
- Gotos and Labels
- Arithmetic/Relational/Bitwise Operators (with short circuting && and || as well as precedence climbing)
- Compound assignment, Increment and Decrement operators
- Static and Extern variables
- Signed/Unsigned integer types with proper C conversions and casting, including long/long long
- Double Precision floating point operations (Including proper NaN support)
- Basic pointer operations (referencing and dereferencing)
- Arrays and Pointer Arithmetic 
	- Includes Pointer addition, subtraction, incr/decrement, and subscript access
	- Simple initialization lists for both static and local variables
	- VLAs not supported
----------------

#####In progress:

- Structs/Unions
- cstrings, string literals, and character constants
- Dynamic memory allocation
- Enums 
- char type (single byte, although this should be similar to long/unsigned type implementation)
- Optimizations

------------------

#####Longer term:

I'm aiming to add some final features of the c language that aren't covered in the book, but these seem less trivial:

- Typedefs
- Function pointers
- Full constexpr and static initialization evaluation (current support is very simplistic)

In addition, this project should have proper tests and benchmarking (both for compile times and program runtimes). For now I am just using the tests from the book.
