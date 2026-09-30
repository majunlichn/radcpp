# JSON Schema meta-schemas

These unmodified JSON files come from
[json-schema-org/json-schema-spec](https://github.com/json-schema-org/json-schema-spec):

| Directory | Upstream commit |
|---|---|
| `draft7` | `20a3fee852db88519dcb3bb329b7a872ebad8953` |
| `draft2019-09` | `c8eb3d320f60eca7cfb18da25337a426ceb40eaa` |
| `draft2020-12` | `601a66c8b0f25246bf0e1fb488c5b5f030a79b72` |

Each `schema.json` is the upstream root file. Files under `meta/` retain their
upstream paths. Hyper-Schema, output schemas, and the optional format-assertion
meta-schema are not bundled.

Upstream offers the source material under AFL or BSD terms. This distribution
uses the BSD license in [LICENSE](LICENSE).

CMake embeds the explicit file list from the root build configuration into
the static library. Compilation and installed consumers need no network access
or runtime schema files. The built-ins are used only when neither an existing
schema resource nor the caller's registry supplies the requested URI.
Caller documents replace the entire resource; missing fragments are reported as
errors rather than filled from the corresponding built-in.

To update these files, fetch their corresponding paths from a reviewed,
pinned upstream commit, update the table above, and run the official JSON
Schema suite and installed-library consumer checks.
