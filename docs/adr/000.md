<!-- cSpell:ignore ADRs, joelparkerhenderson, MADR, Nygard's -->

# [ADR-000] Use ADRs

Status: **accepted**

Deciders: @redeboer, @spflueger

Technical story: A large number of issues in the expertsystem are to be
correlated (e.g. [#40](https://github.com/ComPWA/expertsystem/issues/40),
[#44](https://github.com/ComPWA/expertsystem/issues/44),
[#22](https://github.com/ComPWA/expertsystem/issues/22)) so that resulting PRs
(in this case, [#42](https://github.com/ComPWA/expertsystem/pull/42)) lacked
direction. This led us to consider ADRs.

## Context and Problem Statement

We want to record architectural decisions made in this project. Which format
and structure should these records follow?

## Considered Options

- [MADR](https://adr.github.io/madr/) 2.1.2 – The Markdown Architectural
  Decision Records
- [Michael Nygard's template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
  – The first incarnation of the term "ADR"
- [Sustainable Architectural Decisions](https://www.infoq.com/articles/sustainable-architectural-design-decisions)
  – The Y-Statements
- Other templates listed at
  [github.com/joelparkerhenderson/architecture_decision_record](https://github.com/joelparkerhenderson/architecture_decision_record).
- Formless – No conventions for file format and structure

## Decision Outcome

Chosen option: "MADR 2.1.2", because

- Implicit assumptions should be made explicit. Design documentation is
  important to enable people understanding the decisions later on. See also
  [A rational design process: How and why to fake it](https://ieeexplore.ieee.org/document/6312940/).
- The MADR format is lean and fits our development style.
- The MADR structure is comprehensible and facilitates usage & maintenance.
- The MADR project is vivid.
- Version 2.1.2 is the latest one available when starting to document ADRs.
