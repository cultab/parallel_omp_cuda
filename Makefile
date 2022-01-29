all:
	gcc multisort.c -fopenmp -o multisort
	nvcc conv.cu -o conv -arch=sm_61 -forward-unknown-to-host-compiler -Wall -Wextra -Wconversion -fopenmp
	nvcc mat_vec.cu -o mat_vec -arch=sm_61 -forward-unknown-to-host-compiler -Wall -Wextra -Wconversion -fopenmp
	nvcc covar.cu -o covar -arch=sm_61 -forward-unknown-to-host-compiler -Wall -Wextra -Wconversion -fopenmp

run: all
	./multisort

cuda1: all
	bash -c 'time ./conv'

cuda2: all
	bash -c 'time ./mat_vec'

cuda3: all
	bash -c 'time ./covar'

render:
	R --quiet -e "require(rmarkdown);render('report.rmd');"

submit:
	cp report.pdf 171014.pdf
	zip 171014.zip 171014.pdf

.PHONY: render
