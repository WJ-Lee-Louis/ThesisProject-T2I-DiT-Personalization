# Source B target-aware policy: API smoke audit

Model: `gpt-6-luna`. Eight held-out inputs using saved Module A captions. No images, target decomposition, embeddings or image generation were requested. One response per case; no retries. PASS means structural validation only.

Prompt SHA-256: `6e3f0455481968a63e96a6dc00bbeabec820dda3696ed8eadb8a55f172b1596f`

## dog_location: PASS

```json
{
  "source_caption": "a dog with fluffy golden and white fur sitting on a ledge with its tongue out in front of an orange wall",
  "target_prompt": "a dog standing on a beach",
  "identity_term": "dog"
}
```

```json
{"identity_term":"dog","preserved_spans":["a dog"],"retained_spans":["with fluffy golden and white fur","with its tongue out"],"neutral_spans":[],"components":[{"spans":["sitting"]},{"spans":["on a ledge"]},{"spans":["in front of an orange wall"]}],"conflicts":[]}
```

All removal components deleted:

a dog with fluffy golden and white fur with its tongue out

## dog_hat: PASS

```json
{
  "source_caption": "a dog with fluffy golden and white fur sitting on a ledge with its tongue out in front of an orange wall",
  "target_prompt": "a dog wearing a red hat",
  "identity_term": "dog"
}
```

```json
{"identity_term":"dog","preserved_spans":["a dog"],"retained_spans":["with fluffy golden and white fur","with its tongue out"],"neutral_spans":[],"components":[{"spans":["sitting"]},{"spans":["on a ledge"]},{"spans":["in front of an orange wall"]}],"conflicts":[]}
```

All removal components deleted:

a dog with fluffy golden and white fur with its tongue out

## dog_recolor: PASS

```json
{
  "source_caption": "a dog with fluffy golden and white fur sitting on a ledge with its tongue out in front of an orange wall",
  "target_prompt": "a dog with entirely purple fur",
  "identity_term": "dog"
}
```

```json
{"identity_term":"dog","preserved_spans":["a dog"],"retained_spans":["with fluffy","fur","with its tongue out"],"neutral_spans":[],"components":[{"spans":["golden and white"]},{"spans":["sitting"]},{"spans":["on a ledge"]},{"spans":["in front of an orange wall"]}],"conflicts":[]}
```

All removal components deleted:

a dog with fluffy fur with its tongue out

## dog_rendering: PASS

```json
{
  "source_caption": "a dog with fluffy golden and white fur sitting on a ledge with its tongue out in front of an orange wall",
  "target_prompt": "a dog depicted in watercolor style",
  "identity_term": "dog"
}
```

```json
{"identity_term":"dog","preserved_spans":["dog"],"retained_spans":["with fluffy golden and white fur","with its tongue out"],"neutral_spans":["a"],"components":[{"spans":["sitting"]},{"spans":["on a ledge"]},{"spans":["in front of an orange wall"]}],"conflicts":[]}
```

All removal components deleted:

a dog with fluffy golden and white fur with its tongue out

## backpack_shape: PASS

```json
{
  "source_caption": "A magenta backpack with gold zippers and colorful pins stands on a rocky ledge against a green, mountainous background.",
  "target_prompt": "a cube shaped backpack",
  "identity_term": "backpack"
}
```

```json
{"identity_term":"backpack","preserved_spans":["backpack"],"retained_spans":["magenta","with gold zippers and colorful pins"],"neutral_spans":["A"],"components":[{"spans":["stands"]},{"spans":["on a rocky ledge"]},{"spans":["against a green, mountainous background"]}],"conflicts":[]}
```

All removal components deleted:

A magenta backpack with gold zippers and colorful pins .

## backpack_same_color: PASS

```json
{
  "source_caption": "A magenta backpack with gold zippers and colorful pins stands on a rocky ledge against a green, mountainous background.",
  "target_prompt": "a magenta backpack on a table",
  "identity_term": "backpack"
}
```

```json
{"identity_term":"backpack","preserved_spans":["backpack"],"retained_spans":["magenta","with gold zippers and colorful pins"],"neutral_spans":["A"],"components":[{"spans":["stands"]},{"spans":["on a rocky ledge"]},{"spans":["against a green, mountainous background."]}],"conflicts":[]}
```

All removal components deleted:

A magenta backpack with gold zippers and colorful pins

## style_content: PASS

```json
{
  "source_caption": "A rocky path winds between boulders and scrub toward distant blue hills in a watercolor sketch.",
  "target_prompt": "a castle in watercolor sketch style",
  "identity_term": "watercolor sketch"
}
```

```json
{"identity_term":"watercolor sketch","preserved_spans":["in a watercolor sketch"],"retained_spans":[],"neutral_spans":[],"components":[{"spans":["A rocky path"]},{"spans":["winds"]},{"spans":["between boulders and scrub"]},{"spans":["toward distant blue hills"]}],"conflicts":[]}
```

All removal components deleted:

in a watercolor sketch.

## scene_content: PASS

```json
{
  "source_caption": "A sunset over the ocean reflects orange light across wet sand along a beach beneath pink-purple clouds.",
  "target_prompt": "a city during sunset",
  "identity_term": "sunset"
}
```

```json
{"identity_term":"sunset","preserved_spans":["A sunset"],"retained_spans":["orange light","beneath pink-purple clouds."],"neutral_spans":[],"components":[{"spans":["over the ocean"]},{"spans":["reflects"]},{"spans":["across wet sand along a beach"]}],"conflicts":[]}
```

All removal components deleted:

A sunset orange light beneath pink-purple clouds.

## Review notes

- All eight first responses passed source partition validation; no retries.
- Dog location, red hat and watercolor targets retained the golden/white fur. Full purple recoloring removed only `golden and white`, retaining fluffy fur.
- Backpack color and fittings were retained for both targets. The shape case contained no explicit source shape description, so it cannot validate removal of an observed shape; the dedicated synthetic fixture covers that rule.
- Watercolor identity retained the style carrier while selecting depicted landscape content for removal. Sunset retained orange light and clouds while selecting the ocean/beach context.
- The tongue-out expression was retained under the declared conservative expression policy. This is a policy choice, not an immutable identity claim.
- The scene contrast leaves a grammatical fragment, and punctuation can remain after deletion. These are recorded without rewriting.
- All eight outputs were also reloaded and audited across factorial contexts; retained spans stayed fixed. No SigLIP embeddings or generated-image quality were evaluated in this audit.
