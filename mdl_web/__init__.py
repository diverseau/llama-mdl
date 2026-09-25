"""mdl's web UI: `mdl ui`.

`snapshot` gathers what the page draws - the config's models, which of
them run and how fast, the GPUs - into one JSON document, the same one
`mdl snapshot` prints. `server` serves the page and that document on
127.0.0.1, pushes it over server-sent events as it changes, and runs the
page's buttons as mdl verbs. `static/` is the page: plain HTML, CSS and
JavaScript, no build step.

Standard library only, as the rest of mdl: the page is drawn by the
browser already on the machine.
"""
