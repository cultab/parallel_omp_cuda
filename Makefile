all:
	gcc multisort.c -fopenmp -o multisort

run: all
	./multisort

render:
	R --quiet -e "require(rmarkdown);render('report.rmd');"

submit:
	cp report.pdf 171014.pdf
	zip 171014.zip 171014.pdf

.PHONY: render
