How to contribute?
==================

Some recommended packages for Python development
------------------------------------------------

* `pytest <https://docs.pytest.org/en/latest/>`_: Run :code:`pytest` in the main
  folder of the repository to run all :file:`test_*.py` files

* `pylint <https://www.pylint.org/>`_: Scan your code for naming conventions
  and proper use of Python

* `rope <https://github.com/python-rope/rope>`_: Python refactoring tools

* `sphinx <https://www.sphinx-doc.org/>`_: Generate documentation of your
  Python package

* `doc8 <https://pypi.org/project/doc8/>`_: A style checker for
  `reStructuredText
  <https://docutils.sourceforge.io/docs/ref/rst/introduction.html>`_

These packages and more can be installed using the `requirements-dev.txt
<https://github.com/ComPWA/expertsystem/blob/master/requirements-dev.txt>`_
file:

.. code-block:: shell

  pip install -r requirements_dev.txt


Conventions
-----------

Try to keep test coverage high. You can test current coverage by running

.. code-block:: shell

  cd tests
  pytest

Note that we navigated into the `tests
<https://github.com/ComPWA/expertsystem/tree/master/tests>`_ directory first as
to avoid testing the files in the :doc:`source code directory
</install/get-the-source-code>`. You can view the coverage report by opening
:file:`htmlcov/index.html`.

Git
---

* Please use
  `conventional commit messages <https://www.conventionalcommits.org/>`_: start
  the commit with a semantic keyword (see e.g. `Angular
  <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#type>`_ or
  `these examples <https://seesparkbox.com/foundry/semantic_commit_messages>`_,
  followed by `a column <https://git-scm.com/docs/git-interpret-trailers>`_,
  then the message. The message itself should be in imperative mood — just
  imagine the commit to give a command to the code framework. So for instance:
  :code:`feat: add coverage report tools` or :code:`fix: remove ...`.

* In the master branch, each commit should compile and be tested. In your own
  branches, it is recommended to commit frequently (WIP keyword), but squash
  those commits upon submitting a merge request.

Python
------

* Follow :pep:`8` conventions.

* Any Python file that's part of a module should contain (in this order):

  1. A docstring describing what the file contains and does, followed by two
  empty lines.

  2. A definition of `__all__
     <https://docs.python.org/3/tutorial/modules.html#importing-from-a-package>`_,
     so that you can see immediately what this Python file defines, **followed
     by two empty lines**.

  3. Only after these come the :code:`import` statements, following the
     :pep:`8` conventions for imports.

* When calling or defining multiple arguments of a function and multiple
  entries in a data container, split the entries over multiple lines and end
  the last entry with a comma, like so:

  .. code-block: python

    __all__ = [
        'core',
        'optimizer',
        'physics',
        'plot',
    ]

  This is to facilitate eventual `diff <https://git-scm.com/docs/git-diff>`_
  comparisons in Git.
