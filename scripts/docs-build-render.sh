#!/usr/bin/env bash

pip install --quiet --no-deps --no-build-isolation --force-reinstall -e . \
    && cd docs \
    && rm -f api/_styles-quartodoc.css api/_sidebar.yml *.qmd \
    && quartodoc build && quartodoc interlinks && quarto render \
    && cd .. \
    && pip uninstall --quiet -y arg-dashboard
