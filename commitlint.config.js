module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "type-case": [2, "always", ["upper-case"]],
    "type-enum": [
      2,
      "always",
      ["BEHAVIOR", "BREAK", "DOC", "DX", "ENH", "FEAT", "FIX", "MAINT"],
    ],
  },
};
