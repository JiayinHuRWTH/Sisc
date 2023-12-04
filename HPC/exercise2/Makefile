norm: norm.exe 
	@echo "type <make run> for starting the program"

norm_block: norm_block.exe 
	@echo "type <make run> for starting the program"
	
single: norm_max.exe norm_1_col.exe norm_1_row.exe
	@echo "single executable for each algorithm"

single_block: norm_max.exe norm_1_col.exe norm_1_row.exe norm_1_block.exe
	@echo "single executable for each algorithm"

all: norm.exe norm_max.exe norm_1_col.exe norm_1_row.exe norm_1_block.exe

norm.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW

norm_block.exe: norm.cpp
	${CXX} -o norm.exe $< ${FLAGS_FAST} -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW -DNORM_1_BLOCK

norm_max.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_MAX 

norm_1_col.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_1_COL

norm_1_row.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_1_ROW

norm_1_block.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_1_BLOCK

run: norm.exe
	./norm.exe

help:
	@echo "possible targets: norm single run doc clean"

#-----------------------------------------------------------------------
# Command to generate documentation
#-----------------------------------------------------------------------
doc:
	doxygen doxy.conf

clean:
	rm -f norm.exe
	rm -f norm_max.exe
	rm -f norm_1_col.exe
	rm -f norm_1_row.exe
	rm -f norm_1_block.exe
	rm -f *.o

distclean: clean
	rm -rf doc

