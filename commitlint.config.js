module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "type-enum": [
      2,
      "always",
      [
        "adr",
        "build",
        "ci",
        "chore",
        "docs",
        "epic",
        "feat",
        "fix",
        "refactor",
        "style",
        "test",
      ],
    ],
  },
};
