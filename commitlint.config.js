module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "type-enum": [
      2,
      "always",
      [
        "adr",
        "build",
        "chore",
        "ci",
        "docs",
        "epic",
        "feat",
        "fix",
        "phys",
        "refactor",
        "style",
        "test",
      ],
    ],
  },
};
