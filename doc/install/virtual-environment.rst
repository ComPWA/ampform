Create a virtual environment
============================

It is safest to install `expertsystem` within a virtual environment, so that all
Python dependencies are contained within there. This is helpful in case
something goes wrong with the dependencies: you can just trash the environment
and recreate it. There are two options: :ref:`Conda
<install/virtual-environment:Conda environment>` or :ref:`Python's venv
<install/virtual-environment:Python venv>`.

Conda environment
^^^^^^^^^^^^^^^^^

`Conda <https://www.anaconda.com/>`_ can be installed without administrator
rights, see instructions on `this page
<https://www.anaconda.com/distribution/>`_. Once installed, you can use the
`environment.yml
<https://github.com/ComPWA/expertsystem/blob/master/environment.yml>`_ file from
the :doc:`local repository <get-the-source-code>` to create a Conda environment
for the `expertsystem` installation:

.. code-block:: shell

  conda env create -f environment.yml

This command will take a few minutes, because all required dependencies will be
installed in one go through `pip <https://pypi.org/project/pip/>`_. After Conda
finishes creating the environment, you can activate it with as follows:

.. code-block:: shell

  conda activate tw

You need to have the ``tw`` environment activated whenever you want to run
`expertsystem`.


Python venv
^^^^^^^^^^^

Alternatively, you can use `Python's venv
<https://docs.python.org/3/library/venv.html>`_. All you have to do, is
navigate into :doc:`local repository <get-the-source-code>` and run:

.. code-block:: shell

  python3 -m venv ./venv

This creates a folder called :file:`venv` where all Python packages will be
contained. You first have to activate the environment, and will have to do so
whenever you want to run those Python packages.

.. code-block:: shell

  source ./venv/bin/activate

Now you can safely install the required Python requirements through `pip
<https://pypi.org/project/pip/>`_:

.. code-block:: shell

  pip install -r requirements.txt

That's it, now you're all set to :doc:`install expertsystem <install>`!
