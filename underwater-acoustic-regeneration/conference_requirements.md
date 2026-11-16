# ICBDAIA 2027 requirements

Rechecked 2026-11-16 against the [EasyChair CFP](https://easychair.org/cfp/ICBDAIA27)
and [conference site](https://icbdaia.com/). The CFP lists November 30 submission,
December 15 decisions, December 31 camera-ready/registration, and April 21–22, 2027
conference dates. Use November 29 America/New_York as the internal submission aim.

The current conference homepage explicitly specifies **8–10 pages**; the CFP uses
the wording “at most 8 to 10 pages”. The user's ten-page target, including references,
is within the homepage's stated range. Neither source specifies whether references
count toward the limit or which timezone governs the deadline. Registration amounts
are not reverified here.

The [UNWE event listing](https://www.unwe.bg/en/events/78301/6th-international-conference-on-big-data-and-artificial-intelligence-application.html)
was rechecked and confirms on-site and online participation.

The [linked Springer template](https://drive.google.com/file/d/1X67baiOAmfXYsMWcHocoHNuiSS75y24w/view)
was downloaded (115,357 bytes); its source SHA-256 is in
`provenance/source_retrieval.json`. ZIP/XML inspection confirms a proceedings sample,
150–250-word abstract guidance, numbered references, and embedded VBA parts.
No macros were executed. The user subsequently requested **LaTeX only**, with one
`mainj.tex`, a separate `.bib`, and figures in the same directory. This supersedes
the earlier Word-deliverable requirement.

The provided Word sample identifies LNCS and uses its title, heading, and reference
conventions. The corresponding [official Springer proceedings LaTeX package](https://link.springer.com/series/558/information-for-authors-and-editors)
was retrieved; unmodified `llncs.cls` v2.25 (2026-09-03) and `splncs04.bst` are
included beside the paper. Its standard 122 × 193 mm text area is retained without
manual margin or line-spacing overrides. Provenance is in `provenance/latex_template.json`.

Unsent organizer question: “Do references count within the stated 8–10-page limit,
and which timezone governs the November 30 deadline?” No message has been sent.

The user instructed: "leave anonymous for submission". The manuscript therefore
uses "Anonymous submission", without affiliations or email, and anonymous document
metadata. This resolves the authorship question for this draft; no further names
are required. The original pilot and its targeted correction remain below the
unchanged feasibility trigger. The paper presents numerical diagnosis and measured
improvement on reused development data, with no completed-main-study claim.
Submission and copyright forms remain the user's responsibility unless separately
authorized. Nothing has been submitted or emailed.

The expanded draft also reports a completed GPU comparison on 32 acoustic tasks,
a separately frozen gate comparison on sixteen new recording groups, and a
reasoning-budget check on sixteen reused primary tasks. It distinguishes supported
cue labels, actual display settings, and complete verified execution. Vessel
identification and physical equipment control remain outside the measured scope.

## AI-use disclosure

No separate AI-disclosure rule was found in the published conference homepage or
EasyChair CFP reviewed on 2026-11-16. This does not cover unpublished submission-form
instructions. The [Springer proceedings instructions](https://www.springernature.com/gp/authors/publish-a-book/step-by-step-conference-proceedings)
require documented AI use and link the publisher's detailed disclosure guide.

The [Books and Proceedings disclosure guide, October 2026 v2](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/53830640/data/)
requires a first-page note and an end declaration for AI-assisted text and code/data
work. AI-assisted figures, tables, charts, and graphs require caption disclosure;
supplementary multimedia requires accompanying disclosure. Tool, organization,
model/version, access date, and purpose should be identified. Current
[Springer Nature guidance](https://group.springernature.com/gp/group/ai/ai-guidance-for-researchers-editors-reviewers)
also requests disclosure for copyediting and describes research-tool use in Methods.
An older copyediting exemption should not be used for this revision.

The manuscript adds disclosure for substantial OpenAI Codex assistance with planning,
code, execution, analysis, literature checks, figure/table preparation, and writing.
It distinguishes this assistance from the diffusion and Qwen3-8B models evaluated
in the research. Figures are computed from recorded data and saved results rather
than produced by an image-generation model. The present interface is GPT-6-based;
exact earlier backend revisions were not recorded and must not be invented.
Anonymity is retained: AI tools are disclosed without being named as authors.

Human authors must review and verify the work and accept responsibility before
submission. Automated checks do not establish that this human review has occurred;
the draft does not claim it has. Policy evidence and applicability are recorded in
`reports/ai_disclosure_policy.json`; the publisher guide is preserved, with its
checksum recorded there, in `provenance/ai_disclosure_guidance.pdf`.
