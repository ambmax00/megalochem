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

print("#define REPEAT(FUNC, DELIM, ...) CAT(_REPEAT_, NARGS(__VA_ARGS__))(FUNC, DELIM, __VA_ARGS__)", end='\n')
print("#define _REPEAT_1(FUNC, DELIM, x) FUNC(x)\n", end='\n')

for i in range(2,NMAX+1):
    print("#define _REPEAT_" + str(i) + "(FUNC, DELIM, x, ...) FUNC(x) UNPAREN DELIM _REPEAT_" +  str(i-1) + "(FUNC, DELIM, __VA_ARGS__)")

NMAX = 8

for i in range(1,NMAX+1):
    print("#define _GET_" + str(i) + "(", end='')
    list = ["a_" + str(j) + "," for  j in range(1,i+1)]
    for ele in list:
        print(ele, end='')
    print("...) a_" + str(i), end='\n')
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
#define _GET_1(a_1,...) a_1
#define _GET_2(a_1,a_2,...) a_2
#define _GET_3(a_1,a_2,a_3,...) a_3
#define _GET_4(a_1,a_2,a_3,a_4,...) a_4
#define _GET_5(a_1,a_2,a_3,a_4,a_5,...) a_5
#define _GET_6(a_1,a_2,a_3,a_4,a_5,a_6,...) a_6
#define _GET_7(a_1,a_2,a_3,a_4,a_5,a_6,a_7,...) a_7
#define _GET_8(a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,...) a_8


#define MAKE_PARAM_SINGLE(arg) \
	MAKE_PARAM(_GET_1 arg, _GET_2 arg, _GET_3 arg, _GET_4 arg, _GET_5 arg)
	
#define MAKE_PARAM(struct_name, name, type, reqopt, valref) \
	private: \
	reqopt<type,valref> CAT(c_, name); \
	public: \
	CAT(struct_name, _base)& name(reqopt<type,valref>& in) { \
		CAT(c_, name) = in; \
		return *this; \
	}
	
#define MAKE_STRUCT(structname, getter, ...) \
	class CAT(structname, _base) { \
		REPEAT(MAKE_PARAM_SINGLE, (), __VA_ARGS__) \
		public: \
		UNPAREN getter \
	}; \
	CAT(structname, _base)& structname() { \
		return CAT(structname, _base)(); \
	}

/* EXAMPLE :
* MAKE_STRUCT(create, 
* (J* get() {
* 	return *this;
* }),
* (create, eris, matrix, required, ref), 
* (create, eris2, tensor, optional, ref)
* )
* 
* gives:
* 
* class create_base { 
* private: 
* 	required<matrix,ref> c_eris; 
* public: 
* 	create_base& eris(required<matrix,ref>& in) { 
* 		c_eris = in; 
* 		return *this; 
* 	} 
* private: 
* 	optional<tensor,ref> c_eris2; 
* public: 
* 	create_base& eris2(optional<tensor,ref>& in) { 
* 		c_eris2 = in; 
* 		return *this; 
* 	} 
* public: 
* 	J* get() { return *this; } 
* 
* }; 
* 
* create_base& create() { return create_base(); } 
*
*/
 
#define make_param(structname,name,type,reqopt,refval) \
	private: \
		reqopt < type, refval > c_##name; \
	public: \
		inline structname & name (reqopt < type, refval > i_##name) { \
			c_##name = i_##name; \
			return *this; \
		}

#endif
