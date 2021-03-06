#!/bin/bash

# https://github.com/horejsek/python-webapp-example/blob/master/Makefile
# https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html

.ONESHELL: # https://www.gnu.org/software/make/manual/html_node/One-Shell.html

.PHONY: clean-auxiliary
## Clean auxiliary and temporary files
clean-auxiliary:
	find . -name '*.aux' -exec rm --force {} \;
	find . -name '*.bbl' -exec rm --force {} \;
	find . -name '*.blg' -exec rm --force {} \;
	find . -name '*.cut' -exec rm --force {} \;
	find . -name '*.depytx' -exec rm --force {} \; 
	find . -name '*.log' -exec rm --force {} \; 	
	find . -name '*.pytxcode' -exec rm --force {} \; 
	find . -name '*.synctex.gz' -exec rm --force {} \; 
	rm -rfv pythontex-files-*

build:
	pdflatex PythonTeX_mb_Stencil.tex
	pythontex3 PythonTeX_mb_Stencil.tex
	pdflatex -synctex=1 -interaction=nonstopmode PythonTeX_mb_Stencil.tex
	
# Plonk the following at the end of your Makefile
.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>

# 	+ bugfix:     https://github.com/drivendata/cookiecutter-data-science/issues/67

.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

