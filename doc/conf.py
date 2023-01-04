# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
cwd=os.getcwd()
p=f'{cwd}/../'
print(p)
sys.path.append(p)
import ampyl


# -- Project information -----------------------------------------------------

project = 'ampyl'
copyright = '2022, Maxwell T. Hansen'
author = 'Maxwell T. Hansen'

# The full version, including alpha/beta/rc tags
release = str(ampyl.__version__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    #'jupyter_sphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# remove module names, such package.module.module
# add_module_names = False

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "dotted"
# html_theme = 'jupyter'
# html_theme_path = [p+'/doc']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['jupyter/static']

#### Tweaks by Mattia

html_sidebars = {
    '**' : ['localtoc.html','searchbox.html','logo.html']
    }

# A list of tuples containting pages to link to.  The value should be
# in the form [(name, page), ..]
navbar_links = [('ampyl','ampyl')]
# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_logo = './ampyl-logo-32px.png'
html_theme_options = {
        # A list of tuples containing pages or urls to link to.
        # Valid tuples should be in the following forms:
        #    (name, page)                 # a link to a page
        #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
        #    (name, "http://example.com", True) # arbitrary absolute url
        # Note the "1" or "True" value above as the third argument to indicate
        # an arbitrary url.
        'navbar_links' : [
            ('Introduction','intro/index'),
            ('Documentation','ampyl/index')],


        'navbar_title' : ' ',

        # Render the next and previous page links in navbar. (Default: true)
        'navbar_sidebarrel' : False,

        'bootswatch_theme' : 'spacelab',
        }


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    'fontpkg': '\\usepackage{amsmath,amssymb}',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': r'''

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'extraclassoptions': 'openany,oneside',
    'tableofcontents':' ',
}

latex_logo = 'ampyl-logo.png'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('ampyl/index', 'ampyl-doc.tex', u'ampyl Documentation',
     author, 'manual'),
]
