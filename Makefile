all:
	gcc multisort.c -fopenmp -o multisort
	nvcc conv.cu -o conv -arch=sm_61 -forward-unknown-to-host-compiler -Wall -Wextra -Wconversion -fopenmp

run: all
	./multisort

cuda: all
	./conv

render:
	R --quiet -e "require(rmarkdown);render('report.rmd');"

submit:
	cp report.pdf 171014.pdf
	zip 171014.zip 171014.pdf

.PHONY: render
