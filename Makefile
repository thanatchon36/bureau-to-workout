PY ?= .venv/bin/python

.PHONY: setup bureau workout experiments tables figures banner paper web test all clean

setup:          ## create venv and install the package (needs uv)
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev]"

bureau:         ## one seed of the bureau experiments E1-E5 -> lab_outputs/ (about a minute)
	$(PY) -m bureau_to_workout.cli bureau --exp all

workout:        ## one seed of the workout experiments R1-R4 -> resolution_outputs/ (a few seconds)
	$(PY) -m bureau_to_workout.cli workout

experiments:    ## every experiment over five seeds -> paper_results/ (about three minutes)
	$(PY) -m bureau_to_workout.cli experiments

tables:         ## paper_results/ -> paper/tables/ and paper/results_macros.tex
	$(PY) -m bureau_to_workout.cli tables

figures:        ## paper_results/ -> paper/figures/ (PDF for LaTeX, PNG previews)
	$(PY) -m bureau_to_workout.cli figures

banner:         ## README banner -> docs/assets/banner.png (bureau simulator output)
	$(PY) scripts/make_banner.py

paper:          ## build paper/paper.pdf (tectonic) and the web reader docs/paper/
	$(PY) -m bureau_to_workout.cli paper

web:            ## web reader only -> docs/paper/index.html (pandoc; copies the current paper.pdf)
	$(PY) -m bureau_to_workout.cli web

test:
	$(PY) -m pytest -q

all: experiments tables figures paper

clean:
	rm -rf lab_outputs resolution_outputs paper/main.pdf paper/*.log paper/*.aux paper/*.out paper/*.blg
