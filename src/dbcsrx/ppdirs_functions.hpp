#ifndef UTILS_PPDIRS_FUNCTIONS_HPP
#define UTILS_PPDIRS_FUNCTIONS_HPP

#define CAT(a,...) PRIMITIVE_CAT(a,__VA_ARGS__)
#define PRIMITIVE_CAT(a,...) a##__VA_ARGS__

#define PPDIRS_COMMA ,
#define PPDIRS_COMMA2() ,
#define PPDIRS_OP (
#define PPDIRS_CP )
#define PPDIRS_SC ;

#define PASTE(x, ...) x ## __VA_ARGS__
#define EVALUATING_PASTE(x, ...) PASTE(x, __VA_ARGS__)
#define UNPAREN_IF(x) EVALUATING_PASTE(NOTHING_, EXTRACT x)

#define UNPAREN(...) __VA_ARGS__
#define WRAP(FUNC, ...) 

#define ECHO(x) x
#define ECHO_P(x,n) ECHO(x)

#define XSTR(a) STR(a)
#define STR(a) #a

#define ECHO_NONE(...)

#endif
