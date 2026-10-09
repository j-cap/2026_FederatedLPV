.PHONY: all clean

all:
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
	cp build/main.pdf 2026_FederatedLPV_identification_draft.pdf

clean:
	rm -rf build
