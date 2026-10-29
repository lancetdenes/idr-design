# Deploying

Three steps: stand up the worker, point the page at it, publish the page.

---

## 1. The worker

The worker runs on [Modal](https://modal.com): serverless CPU, billed per
second of compute and nothing while idle, which suits a page with occasional
traffic. No GPU is needed — the interaction maps are analytical numpy.

```bash
pip install modal
modal token new          # opens a browser to authenticate
modal deploy worker/modal_app.py
```

Modal prints one URL per endpoint:

```
├── 🔨 Created submit => https://<workspace>--idrdesign-submit.modal.run
├── 🔨 Created result => https://<workspace>--idrdesign-result.modal.run
```

The first deploy builds the image — CPU torch, plus GOOSE, sparrow and
FINCHES compiled from source — which takes several minutes. Later deploys reuse
the cached layers and finish in seconds.

To deploy the STARLING variant instead, use `worker/modal_app_starling.py`. That
one does request a GPU and pulls the much larger CUDA torch build, so its first
image build takes 10–20 minutes. Read the caveat in the README first.

### Smoke test

```bash
curl -X POST https://<workspace>--idrdesign-submit.modal.run \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"'"$(python -c 'print("GS"*40)')"'","max_iterations":20,"render_maps":false}'
# -> {"status":"running","job_id":"fc-..."}

curl 'https://<workspace>--idrdesign-result.modal.run?job_id=fc-...'
# -> {"status":"running"}, then the full result
```

There are no model weights to download on this path, so the first real run
costs no more than any other.

### Restricting access

The endpoints are open to anyone with the URL. To stop other sites calling them
from a browser:

```bash
modal secret create idrdesign-cors ALLOWED_ORIGIN=https://lancedenes.org
```

and add `secrets=[modal.Secret.from_name("idrdesign-cors")]` to the `submit` and
`result` decorators.

This is a same-origin restriction, not authentication — it stops other pages
embedding the API, not someone calling it with curl. For real access control,
put `modal.Proxy` or a shared token in front of it.

---

## 2. Point the page at the worker

Set the two constants at the top of `web/design.js`:

```js
const SUBMIT_URL = "https://<workspace>--idrdesign-submit.modal.run";
const RESULT_URL = "https://<workspace>--idrdesign-result.modal.run";
```

Left empty, the page runs in preview mode: every control works and the results
are GS-repeat placeholders, labelled as such. This is the mode to use when
working on layout, since it needs no GPU.

---

## 3. The page

`design.html` expects `style.css`, `fonts/` and the rest of the site beside it,
so the three files go in the root of the website repo, next to `index.html`:

```bash
cp web/design.html web/design.css web/design.js /path/to/website/
```

Then add Design to the navigation in two places.

**The `subnav` on interior pages.** One list item, already present in
`design.html`:

```html
<li><a href="design.html" data-conf="c6">Design</a></li>
```

Add it to `research.html`, `people.html`, `publications.html`, `cv.html` and
`contact.html` as well, or the link will be missing from those pages.

**The landing orbit in `index.html`.** The five existing links sit at
hand-placed coordinates around the ensemble SVG, each lighting one numbered
conformer (`data-conf="c1"` through `c5`). A sixth link needs a position of its
own, a conformer to light, and small adjustments to the other five so the ring
stays evenly spaced. The coordinates are in `style.css` under `.orbit a`.

Push the website repo; Vercel builds on push.

---

## Cost

Only the worker container is billed, and only while a job is running. The
function asks for 8 CPUs, which on Modal is on the order of $0.0004/s, so a
5-minute run costs roughly $0.12. The HTTP endpoints are negligible.
`max_containers` in `worker/modal_app.py` caps concurrent spend.

Runs are minutes, not seconds, so the per-run cost is dominated by wall time —
the Effort slider on the page is the main cost control.
