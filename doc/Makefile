PDOC ?= python -m pdoc
BUILDDIR ?= _build
HTMLDIR := $(BUILDDIR)/html

MODULES := \
	ampyl \
	ampyl.ampyl \
	ampyl.check_utils \
	ampyl.constants \
	ampyl.cuts \
	ampyl.flavor \
	ampyl.flavor_ope_utils \
	ampyl.flavor_utils \
	ampyl.fv_spectrum_utils \
	ampyl.groups \
	ampyl.interpolable \
	ampyl.interpolable_utils \
	ampyl.k_matrices \
	ampyl.kinematic_functions \
	ampyl.nonint_utils \
	ampyl.qc_functions \
	ampyl.schemes \
	ampyl.shell_utils \
	ampyl.spaces \
	ampyl.version \
	ampyl.wigner

.PHONY: html clean

html:
	rm -rf "$(HTMLDIR)"
	$(PDOC) --docformat numpy --math --output-directory "$(HTMLDIR)" $(MODULES)
	@touch "$(HTMLDIR)/.nojekyll"

clean:
	rm -rf "$(BUILDDIR)"
