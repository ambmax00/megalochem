#ifndef UTILS_PPDIRS_ARITHEMTIC_HPP
#define UTILS_PPDIRS_ARITHMETIC_HPP

/*
##### DEC #####

print("#define DEC(x) CAT(_DEC_,x)",end='\n')
for i in range(1,NMAX+1):
    print("#define _DEC_" + str(i) + " " + str(i-1),end='\n')
print('\n')

##### INC #####

print("#define INC(x) CAT(_INC_,x)",end='\n')
for i in range(0,NMAX+1):
    print("#define _INC_" + str(i) + " " + str(i+1),end='\n')
print('\n')

*/

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

#define ADD(X,Y) RECURSIVE(INC, Y, X)
#define SUB(X,Y) RECURSIVE(DEC, Y, X)

#endif
