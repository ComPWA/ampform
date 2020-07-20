# JSON Validation Schema for XML format

Initially, when the `expertsystem` was part of `pycompwa`, it wrote its
amplitude model to an XML file using `xmltodict`. In the current state of the
`expertsystem`
([4f6f6a0](https://github.com/ComPWA/expertsystem/tree/4f6f6a0287e93286ec67db1dfb40966de276e909)),
however, the structure of particles written to such an XML file is also used
internally in the form of a nested `dict`s. This has to be factored out.

While JSON Schema is not suitable for validating XML files, it can be used to
validate the structure of a nested `dict` (as `xmltodict` produces it when
parsing an XML file). The JSON Schema files in this folder are used for that
purpose. These `dict`s will slowly be phased out in favor of class structures.
A guarantee is therefore needed that the `dict` structures passed to the system
have the correct format.
