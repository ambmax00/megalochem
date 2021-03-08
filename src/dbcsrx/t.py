##### ITERATE_N #####
NMAX=32
MAXDEPTH=2
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
