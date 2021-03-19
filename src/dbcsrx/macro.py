# PYTHON SCRIPT FOR GENERATING MACROS:
NMAX = 32

listfront = ["_" + str(i) + ", " for i in range(0,NMAX+1)]
listback = [str(i) + "," for i in range(NMAX,0,-1)]

##### NARGS #####

print("#define NARGS_SEQ(", end='')
for ele in listfront:
    print(ele, end='')
print("N, ...) N", end='\n')
print("#define NARGS(...) NARGS_SEQ(0, ##__VA_ARGS__,", end='')
for ele in listback:
    print(ele, end='')
print("0)",end='\n')
print('\n')

##### ITERATE #####

print("#define ITERATE_LIST(FUNC, DELIM, SUFFIX, list) ITERATE(FUNC, DELIM, SUFFIX, UNPAREN list)", end='\n')
print("#define ITERATE(FUNC, DELIM, SUFFIX, ...) CAT(_ITERATE_, NARGS(__VA_ARGS__))(FUNC, DELIM, SUFFIX, __VA_ARGS__)", end='\n')
print("#define _ITERATE_0(FUNC, DELIM, SUFFIX, x) ", end='\n')
print("#define _ITERATE_1(FUNC, DELIM, SUFFIX, x) FUNC(x) UNPAREN SUFFIX", end='\n')
for i in range(2,NMAX+1):
    print("#define _ITERATE_" + str(i) + "(FUNC, DELIM, SUFFIX, x, ...) FUNC(x) UNPAREN DELIM _ITERATE_" +  str(i-1) + "(FUNC, DELIM, SUFFIX, __VA_ARGS__)")
print('\n')

##### ITERATE_N #####

print("#define ITERATE_N_LIST(FUNC, DELIM, SUFFIX, list) ITERATE_N(FUNC, DELIM, SUFFIX, UNPAREN list)", end='\n')
print("#define ITERATE_N(FUNC, DELIM, SUFFIX, ...) CAT(_ITERATE_N_, NARGS(__VA_ARGS__))(FUNC, DELIM, SUFFIX, __VA_ARGS__)", end='\n')
print("#define _ITERATE_N_0(FUNC, DELIM, SUFFIX, x) ", end='\n')
print("#define _ITERATE_N_1(FUNC, DELIM, SUFFIX, x) FUNC(x,1) UNPAREN SUFFIX", end='\n')
for i in range(2,NMAX+1):
    print("#define _ITERATE_N_" + str(i) + "(FUNC, DELIM, SUFFIX, x, ...) FUNC(x, " + str(i) +") UNPAREN DELIM _ITERATE_N_" +  str(i-1) + "(FUNC, DELIM, SUFFIX, __VA_ARGS__)")
print('\n')

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

##### GET #####

for i in range(1,NMAX+1):
    print("#define GET_" + str(i) + "(", end='')
    list = ["a_" + str(j) + "," for  j in range(1,i+1)]
    for ele in list:
        print(ele, end='')
    print("...) a_" + str(i), end='\n')
print('\n')

##### REPEAT #####

for idepth in ["FIRST", "SECOND", "THIRD"]:
    rname="REPEAT_" + idepth
    print("#define " + rname + "(FUNC, VAR, NSTART, N, DELIM, SUFFIX) " + \
            "CAT(_" + rname + "_, N)(FUNC, VAR, NSTART, DELIM, SUFFIX)", end='\n')
    print("#define _" + rname + "_0(FUNC, VAR, NSTART, DELIM, SUFFIX) ", end='\n')
    print("#define _" + rname + "_1(FUNC, VAR, NSTART, DELIM, SUFFIX) FUNC(VAR,NSTART) UNPAREN SUFFIX", end='\n')
    for i in range(2,NMAX+1):
     print("#define _" + rname + "_" + str(i) + "(FUNC, VAR, NSTART, DELIM, SUFFIX) FUNC(VAR,NSTART) UNPAREN DELIM _" + rname + "_" \
            +  str(i-1) + "(FUNC, VAR, INC(NSTART), DELIM, SUFFIX)")
    print('\n')

##### RECURSIVE #####

rname="RECURSIVE" # + idepth
print("#define " + rname + "(FUNC, NSTEPS, ARGS) " + \
    "CAT(_" + rname + "_, NSTEPS)(FUNC, ARGS)", end='\n')
print("#define _" + rname + "_0(FUNC, ARGS) ", end='\n')
print("#define _" + rname + "_1(FUNC, ARGS) FUNC(ARGS)", end='\n')
for i in range(2,NMAX+1):
    print("#define _" + rname + "_" + str(i) + "(FUNC, X) FUNC( _" + rname + "_" \
        +  str(i-1) + "(FUNC, X))")
print('\n')

