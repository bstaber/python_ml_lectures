SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html