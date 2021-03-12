#define CAT(a,...) PRIMITIVE_CAT(a,__VA_ARGS__)
#define PRIMITIVE_CAT(a,...) a##__VA_ARGS__

#define INC(x) CAT(_INC_,x)
#define _INC_0 1
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

#define RECURSIVE(FUNC, NSTEPS, ARGS) CAT(_RECURSIVE_, NSTEPS)(FUNC, ARGS)
#define _RECURSIVE_0(FUNC, ARGS) ARGS
#define _RECURSIVE_1(FUNC, ARGS) FUNC(ARGS)
#define _RECURSIVE_2(FUNC, X) FUNC( _RECURSIVE_1(FUNC, X))
#define _RECURSIVE_3(FUNC, X) FUNC( _RECURSIVE_2(FUNC, X))
#define _RECURSIVE_4(FUNC, X) FUNC( _RECURSIVE_3(FUNC, X))
#define _RECURSIVE_5(FUNC, X) FUNC( _RECURSIVE_4(FUNC, X))
#define _RECURSIVE_6(FUNC, X) FUNC( _RECURSIVE_5(FUNC, X))
#define _RECURSIVE_7(FUNC, X) FUNC( _RECURSIVE_6(FUNC, X))
#define _RECURSIVE_8(FUNC, X) FUNC( _RECURSIVE_7(FUNC, X))
#define _RECURSIVE_9(FUNC, X) FUNC( _RECURSIVE_8(FUNC, X))
#define _RECURSIVE_10(FUNC, X) FUNC( _RECURSIVE_9(FUNC, X))
#define _RECURSIVE_11(FUNC, X) FUNC( _RECURSIVE_10(FUNC, X))
#define _RECURSIVE_12(FUNC, X) FUNC( _RECURSIVE_11(FUNC, X))
#define _RECURSIVE_13(FUNC, X) FUNC( _RECURSIVE_12(FUNC, X))
#define _RECURSIVE_14(FUNC, X) FUNC( _RECURSIVE_13(FUNC, X))
#define _RECURSIVE_15(FUNC, X) FUNC( _RECURSIVE_14(FUNC, X))
#define _RECURSIVE_16(FUNC, X) FUNC( _RECURSIVE_15(FUNC, X))
#define _RECURSIVE_17(FUNC, X) FUNC( _RECURSIVE_16(FUNC, X))
#define _RECURSIVE_18(FUNC, X) FUNC( _RECURSIVE_17(FUNC, X))
#define _RECURSIVE_19(FUNC, X) FUNC( _RECURSIVE_18(FUNC, X))
#define _RECURSIVE_20(FUNC, X) FUNC( _RECURSIVE_19(FUNC, X))
#define _RECURSIVE_21(FUNC, X) FUNC( _RECURSIVE_20(FUNC, X))
#define _RECURSIVE_22(FUNC, X) FUNC( _RECURSIVE_21(FUNC, X))
#define _RECURSIVE_23(FUNC, X) FUNC( _RECURSIVE_22(FUNC, X))
#define _RECURSIVE_24(FUNC, X) FUNC( _RECURSIVE_23(FUNC, X))
#define _RECURSIVE_25(FUNC, X) FUNC( _RECURSIVE_24(FUNC, X))
#define _RECURSIVE_26(FUNC, X) FUNC( _RECURSIVE_25(FUNC, X))
#define _RECURSIVE_27(FUNC, X) FUNC( _RECURSIVE_26(FUNC, X))
#define _RECURSIVE_28(FUNC, X) FUNC( _RECURSIVE_27(FUNC, X))
#define _RECURSIVE_29(FUNC, X) FUNC( _RECURSIVE_28(FUNC, X))
#define _RECURSIVE_30(FUNC, X) FUNC( _RECURSIVE_29(FUNC, X))
#define _RECURSIVE_31(FUNC, X) FUNC( _RECURSIVE_30(FUNC, X))
#define _RECURSIVE_32(FUNC, X) FUNC( _RECURSIVE_31(FUNC, X))

#define ADD(X,Y) RECURSIVE(INC, Y, X)
#define SUB(X,Y) RECURSIVE(DEC, Y, X)

5 + 3 = ADD(5,3)
10 + 11 = ADD(10,11)
0 + 6 = ADD(0,6)
5 - 3 = SUB(5,3)
