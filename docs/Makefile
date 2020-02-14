# You can set these variables from the command line.

ifeq ($(LANG), en)
  SOURCEDIR     = source_en
  pdf_name      = Sophon_Inference_en.pdf
else ifeq($(LANG), zh)
  SOURCEDIR     = source_zh
  pdf_name      = Sophon_Inference_zh.pdf
else
  exit 1
endif

#echo $SOURCEDIR
#exit 1

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pdf: latex
	@cd $(BUILDDIR)/latex && xelatex Sophon\ Inference.tex
	#@mv $(BUILDDIR)/latex/*.pdf $(BUILDDIR) #&& rm -rf $(BUILDDIR)/latex
	@cd $(BUILDDIR)/latex && xelatex Sophon\ Inference.tex
	@mv $(BUILDDIR)/latex/Sophon\ Inference.pdf $(BUILDDIR)/"${pdf_name}" && rm -rf $(BUILDDIR)/latex

web: html
	#@python3 -m http.server --directory build/html

clean:
	@rm -rf $(BUILDDIR)
