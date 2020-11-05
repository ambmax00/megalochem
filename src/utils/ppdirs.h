#ifndef UTILS_PPDIRS_H
#define UTILS_PPDIRS_H

#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define UNPAREN(...) __VA_ARGS__
#define WRAP(FUNC, ...) 

/* PYTHON SCRIPT FOR GENERATING MACROS:
NMAX = 32

listfront = ["_" + str(i) + ", " for i in range(1,NMAX+1)]
listback = [str(i) + "," for i in range(NMAX,1,-1)]

print("#define NARGS_SEQ(", end='')
for ele in listfront:
    print(ele, end='')
print("N, ...) N", end='\n')

print("#define NARGS(...) NARGS_SEQ(__VA_ARGS__,", end='')
for ele in listback:
    print(ele, end='')
print("1)",end='\n')

print('\n')

print("#define REPEAT(FUNC, DELIM, ...) CAT(_REPEAT_, NARGS(__VA_ARGS__))(FUNC, DELIM, __VA_ARGS__)", end='\n')
print("#define _REPEAT_1(FUNC, DELIM, x) FUNC(x)\n", end='\n')

for i in range(2,NMAX+1):
    print("#define _REPEAT_" + str(i) + "(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_" +  str(i-1) + "(FUNC, DELIM, __VA_ARGS__)")

print('\n')

print("#define DEC(x) CAT(_DEC_,x)",end='\n')
for i in range(1,NMAX+1):
    print("#define _DEC_" + str(i) + " " + str(i-1),end='\n')

print('\n')

print("#define INC(x) CAT(_INC_,x)",end='\n')
for i in range(1,NMAX+1):
    print("#define _INC_" + str(i) + " " + str(i+1),end='\n')

print('\n')

for i in range(1,NMAX+1):
    print("#define _GET_" + str(i) + "(", end='')
    list = ["a_" + str(j) + "," for  j in range(1,i+1)]
    for ele in list:
        print(ele, end='')
    print("...) a_" + str(i), end='\n')

print('\n')

print("#define ITERATE(FUNC, DELIM, ARG) CAT(_ITERATE_, NARGS(UNPAREN ARG))(FUNC, DELIM, ARG, 1)",end='\n')
print("#define _ITERATE_1(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG)",end='\n')
for i in range(2,NMAX+1):
    print("#define _ITERATE_" + str(i) + "(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG)" + \
            " UNPAREN DELIM _ITERATE_" + str(i-1) + "(FUNC, DELIM, ARG, INC(N))",end='\n')

print('\n')
*/

#define NARGS_SEQ(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, N, ...) N
#define NARGS(...) NARGS_SEQ(__VA_ARGS__,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)


#define REPEAT(FUNC, DELIM, ...) CAT(_REPEAT_, NARGS(__VA_ARGS__))(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_1(FUNC, DELIM, x) FUNC(x)

#define _REPEAT_2(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_1(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_3(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_2(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_4(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_3(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_5(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_4(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_6(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_5(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_7(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_6(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_8(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_7(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_9(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_8(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_10(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_9(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_11(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_10(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_12(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_11(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_13(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_12(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_14(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_13(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_15(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_14(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_16(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_15(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_17(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_16(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_18(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_17(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_19(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_18(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_20(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_19(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_21(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_20(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_22(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_21(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_23(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_22(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_24(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_23(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_25(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_24(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_26(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_25(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_27(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_26(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_28(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_27(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_29(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_28(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_30(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_29(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_31(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_30(FUNC, DELIM, __VA_ARGS__)
#define _REPEAT_32(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_31(FUNC, DELIM, __VA_ARGS__)


#define DEC(x) CAT(_DEC_,x)
#define _DEC_1 0
#define _DEC_2 1
#define _DEC_3 2
#define _DEC_4 3
#define _DEC_5 4
#define _DEC_6 5
#define _DEC_7 6
#define _DEC_8 7
#define _DEC_9 8
#define _DEC_10 9
#define _DEC_11 10
#define _DEC_12 11
#define _DEC_13 12
#define _DEC_14 13
#define _DEC_15 14
#define _DEC_16 15
#define _DEC_17 16
#define _DEC_18 17
#define _DEC_19 18
#define _DEC_20 19
#define _DEC_21 20
#define _DEC_22 21
#define _DEC_23 22
#define _DEC_24 23
#define _DEC_25 24
#define _DEC_26 25
#define _DEC_27 26
#define _DEC_28 27
#define _DEC_29 28
#define _DEC_30 29
#define _DEC_31 30
#define _DEC_32 31


#define INC(x) CAT(_INC_,x)
#define _INC_1 2
#define _INC_2 3
#define _INC_3 4
#define _INC_4 5
#define _INC_5 6
#define _INC_6 7
#define _INC_7 8
#define _INC_8 9
#define _INC_9 10
#define _INC_10 11
#define _INC_11 12
#define _INC_12 13
#define _INC_13 14
#define _INC_14 15
#define _INC_15 16
#define _INC_16 17
#define _INC_17 18
#define _INC_18 19
#define _INC_19 20
#define _INC_20 21
#define _INC_21 22
#define _INC_22 23
#define _INC_23 24
#define _INC_24 25
#define _INC_25 26
#define _INC_26 27
#define _INC_27 28
#define _INC_28 29
#define _INC_29 30
#define _INC_30 31
#define _INC_31 32
#define _INC_32 33


#define _GET_1(a_1,...) a_1
#define _GET_2(a_1,a_2,...) a_2
#define _GET_3(a_1,a_2,a_3,...) a_3
#define _GET_4(a_1,a_2,a_3,a_4,...) a_4
#define _GET_5(a_1,a_2,a_3,a_4,a_5,...) a_5
#define _GET_6(a_1,a_2,a_3,a_4,a_5,a_6,...) a_6
#define _GET_7(a_1,a_2,a_3,a_4,a_5,a_6,a_7,...) a_7
#define _GET_8(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,...) a_8
#define _GET_9(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,...) a_9
#define _GET_10(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,...) a_10
#define _GET_11(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,...) a_11
#define _GET_12(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,...) a_12
#define _GET_13(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,...) a_13
#define _GET_14(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,...) a_14
#define _GET_15(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,...) a_15
#define _GET_16(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,...) a_16
#define _GET_17(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,...) a_17
#define _GET_18(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,...) a_18
#define _GET_19(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,...) a_19
#define _GET_20(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,...) a_20
#define _GET_21(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,...) a_21
#define _GET_22(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,...) a_22
#define _GET_23(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,...) a_23
#define _GET_24(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,...) a_24
#define _GET_25(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,...) a_25
#define _GET_26(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,...) a_26
#define _GET_27(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,a_27,...) a_27
#define _GET_28(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,a_27,a_28,...) a_28
#define _GET_29(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,a_27,a_28,a_29,...) a_29
#define _GET_30(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,a_27,a_28,a_29,a_30,...) a_30
#define _GET_31(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,a_27,a_28,a_29,a_30,a_31,...) a_31
#define _GET_32(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12,a_13,a_14,a_15,a_16,a_17,a_18,a_19,a_20,a_21,a_22,a_23,a_24,a_25,a_26,a_27,a_28,a_29,a_30,a_31,a_32,...) a_32


#define ITERATE(FUNC, DELIM, ARG) CAT(_ITERATE_, NARGS(UNPAREN ARG))(FUNC, DELIM, ARG, 1)
#define _ITERATE_1(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG)
#define _ITERATE_2(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_1(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_3(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_2(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_4(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_3(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_5(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_4(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_6(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_5(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_7(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_6(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_8(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_7(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_9(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_8(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_10(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_9(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_11(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_10(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_12(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_11(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_13(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_12(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_14(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_13(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_15(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_14(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_16(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_15(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_17(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_16(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_18(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_17(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_19(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_18(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_20(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_19(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_21(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_20(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_22(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_21(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_23(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_22(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_24(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_23(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_25(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_24(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_26(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_25(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_27(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_26(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_28(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_27(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_29(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_28(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_30(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_29(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_31(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_30(FUNC, DELIM, ARG, INC(N))
#define _ITERATE_32(FUNC, DELIM, ARG, N) FUNC(CAT(_GET_, N) ARG) UNPAREN DELIM _ITERATE_31(FUNC, DELIM, ARG, INC(N))


#define PUSH_FRONT(val, args) \
	(val,UNPAREN args)

#define INITPARAM(arg) \
	private: \
		_GET_2 arg CAT(c_, _GET_1 arg);

#define CONSTRUCTOR(arg) \
	_GET_2 arg CAT(i_, _GET_1 arg)

#define INITIALIZER(arg) \
	CAT(c_,_GET_1 arg) (CAT(i_, _GET_1 arg))
	
#define FORWARD(arg) \
	CAT(i_,_GET_1 arg)

#define MAKE_CONSTR(structname, args) \
	ITERATE(INITPARAM, (), args) \
	private: \
		CAT(CAT(create_, structname),_base)(ITERATE(CONSTRUCTOR, (,), args)) \
		: ITERATE(INITIALIZER, (,), args) {}
	
#define MAKE_PARAM_WRAP(arg) \
	MAKE_PARAM(_GET_1 arg, _GET_2 arg, _GET_3 arg, _GET_4 arg)

#define MAKE_PARAM(name, type, reqopt, valref) \
	private: \
	reqopt<type,valref> CAT(c_, name); \
	public: \
	_create_base& name(reqopt<type,valref> in) { \
		CAT(c_, name) = in; \
		return *this; \
	}
	
#define INIT_GETTER(arg) \
	CAT(c_, _GET_1 arg)
	
#define INIT_SETTER(arg) \
	CAT(INIT_SETTER_, _GET_3 arg)(_GET_1 arg, _GET_5 arg)

#define INIT_SETTER_optional(name, default) \
	CAT(ptr->m_, name) = (CAT(c_, name)) ? \
		CAT(*c_,name) : default;
		
#define INIT_SETTER_required(name, default) \
	CAT(ptr->m_, name) = CAT(*c_, name);

#define MAKE_GETTER(args1, args2) \
	std::shared_ptr<_base> get() { \
		std::shared_ptr<_derived> ptr = new _derived( \
			ITERATE(INIT_GETTER, (,), args1)); \
		ITERATE(INIT_SETTER, (), args2) \
		std::shared_ptr<_base> out = ptr; \
		return ptr; \
	} 
		

#define MAKE_STRUCT(structname, basename, args1, args2) \
	class CAT(CAT(create_, structname), _base) { \
		typedef CAT(CAT(create_, structname), _base) _create_base; \
		typedef structname _derived; \
		typedef basename _base; \
		MAKE_CONSTR(structname, args1) \
		ITERATE(MAKE_PARAM_WRAP, (), args2) \
		MAKE_GETTER(args1, args2) \
	}; \
	CAT(CAT(create_, structname), _base)& CAT(create_, structname)( \
		ITERATE(CONSTRUCTOR, (,), args1) \
		) { \
		return CAT(CAT(create_, structname), _base)( \
		ITERATE(FORWARD, (,), args1) \
		); \
	}


#define ECHO(x) x
 
#define make_param(structname,name,type,reqopt,refval) \
	private: \
		reqopt < type, refval > c_##name; \
	public: \
		inline structname & name (reqopt < type, refval > i_##name) { \
			c_##name = i_##name; \
			return *this; \
		}

#endif
