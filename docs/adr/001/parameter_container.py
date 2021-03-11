import attr


# the new FitParameter class would have this structure
@attr.s
class Parameter:
    value: float = attr.ib()
    fix: bool = attr.ib(default=False)


# the new FitParameters collection would have such a structure
mapping = {
    "par1": Parameter(1.0),
    "par2": Parameter(2.0, fix=False),
}


# intensity nodes and dynamics classes contain immutable strings
class Dynamics:
    pass


@attr.s
class CustomDynamics(Dynamics):
    par: str = attr.ib(on_setattr=attr.setters.frozen, kw_only=True)


dyn1 = CustomDynamics(par="par1")
dyn2 = CustomDynamics(par="par2")

# Parameters would be coupled like this
mapping["par1"] = mapping["par2"]
assert mapping["par2"] is mapping["par1"]
assert mapping["par1"] == {
    "par1": Parameter(1.0),
    "par2": Parameter(1.0),
}
