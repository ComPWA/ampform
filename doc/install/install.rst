Build and install
=================

Once you :doc:`have the source code <get-the-source-code>` and have
:doc:`activated the virtual environment <virtual-environment>`, you're ready to
build and install `expertsystem`.

When you install `expertsystem`, you are telling the system where to find it.
There are two ways of doing this:

(1) by copying the source code and binaries to a folder known to the system
    (you do this :ref:`with setuptools <setuptools>`)
(2) by telling the system to directly monitor the :doc:`local repository
    <get-the-source-code>` as the installation path (we call this
    :ref:`'developer mode' <install/build:Developer Mode>`).

The second option is more dynamic, because any changes to the source code are
immediately available at runtime, which allows you to tweak the code and try
things out. When using the first option, you would have to run :ref:`setuptools
<setuptools>` again to make the changes known to the system.


Using `setuptools <https://setuptools.readthedocs.io/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is easy-peasy! Just navigate to the :doc:`local repository
<get-the-source-code>` and run:

.. code-block:: shell

  python setup.py install

The build output is written to a folder :file:`build` and copied to the virtual
environment directory.


Developer Mode
~~~~~~~~~~~~~~

In this set-up, we first tell the virtual environment to monitor the source
code directory as an install directory. So, navigate to the base folder of the
:doc:`local repository <get-the-source-code>` then, depending on which
:doc:`virtual environment </install/virtual-environment>` you chose, do the
following:

.. code-block:: shell
  :caption: if you :ref:`use a Conda environment
    <install/virtual-environment:Conda environment>`

  conda develop .

.. code-block:: shell
  :caption: if you :ref:`use Python venv
    <install/virtual-environment:Python venv>`

  pip install virtualenvwrapper
  source venv/bin/virtualenvwrapper.sh
  add2virtualenv .

That's all! The virtual environment while now use the :file:`expertsystem`
folder in your :doc:`local repository <get-the-source-code>` when you
:code:`import expertsystem`.


Test the installation
~~~~~~~~~~~~~~~~~~~~~

First, navigate out of the main directory of the :doc:`local repository
<get-the-source-code>` in order to make sure that the `expertsystem` we run is
the system installation and not the :file:`expertsystem` folder in the current
working directory. Then, simply launch launch a Python interpreter and run:

.. code-block:: python

  import expertsystem

If you don't get any error messages, all worked out nicely!

For more thorough testing you can run the unit tests:

.. code-block:: shell

  pip install -r tests/requirements.txt
  pytest -m "not slow"
