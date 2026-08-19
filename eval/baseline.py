"""The raw-data-in-prompt arm of paper.md Section 4.2 part (C).

Part (C) claims DALE's context footprint stays bounded while a naive
baseline's grows with dataset size. `dale.agent.TokenUsage` measures DALE's
side from real runs; this module measures the other side — the prompt a model
would need if you simply pasted the whole dataset in and asked the question —
and, for the same use case, DALE's own first-turn context, so the two numbers
come from one instrument rather than two.

=========================================================================
READ THIS BEFORE QUOTING ANY CORRECTNESS NUMBER FROM THIS MODULE
=========================================================================
**The two arms are not graded on the same kind of evidence, and cannot be.**

DALE is scored on *computed state*: every checker in eval/use_cases.py takes
the model's `final_answer` and ignores it, searching instead for a live
registry handle whose contents match ground truth. The baseline is scored on
*stated output*: it has no registry state — one shot, no tool calls — so all
it can be graded on is what it says.

That asymmetry is deliberate and, we argue, the defensible comparison: each
arm is graded on the artifact its own architecture actually produces. DALE's
claim is that a pipeline of operations computed the right answer, and the
handle is the evidence; the baseline's claim is that a model read the data and
reported the right answer, and its statement is the evidence. Grading DALE on
its prose would measure its summarizing, not its computing; grading the
baseline on state it structurally cannot have would score it 0% by
construction (which is exactly why the existing checkers are not reused here —
see decision 4).

But it is still an asymmetry, and it is not neutral: a DALE run that computes
the right handle and then describes it badly still passes, where a baseline
that reasons correctly and states it badly fails. **Anyone lifting a
correctness comparison out of this module into prose must carry that sentence
with it.** The token-scaling numbers are unaffected — those are counted, not
graded — and are the numbers this module exists for.
=========================================================================

Deliberate design decisions, each of which could be made dishonestly:

1. **The baseline is serialized as CSV by default, not JSON.** For the flat
   record shapes these use cases produce, CSV costs roughly 2-3x fewer tokens
   per row than JSON, because JSON repeats every field name on every record.
   Publishing a JSON curve would be a strawman: it would make the baseline
   look worse for a reason that has nothing to do with DALE. `json` and `tsv`
   are supported so the difference can be shown, but CSV is the headline.

2. **DALE's arm is counted *with* its tool schemas — all of them, including
   the `final_result` output tool.** They are essentially its entire cost:
   measured against claude-sonnet-5 at 2,105 records, 16,382 tokens of schemas
   against 1,322 for the system prompt plus task. They are sent on every
   request, so omitting them would over-flatter DALE by an order of magnitude.
   `final_result` is easy to miss — pydantic-ai synthesizes it from
   `build_agent`'s `output_type=DaleResult` rather than from `build_tools` —
   and it is ~3% of the payload, always in DALE's favour. `tool_schemas`
   therefore reads the tool set off the wire (see its docstring) instead of
   rebuilding a list by hand, so the counted payload is the sent payload.
   `ArmCounts` reports the schema share separately, so a reader can see that
   split rather than take it on trust.

3. **Counting goes through `anthropic.messages.count_tokens`.** It is a real
   HTTP call and needs `ANTHROPIC_API_KEY`, but it runs *no inference*, which
   is what makes the whole scaling curve free to produce. `anthropic` is not a
   declared dependency of this project (see pyproject.toml), so the import is
   guarded the same way eval/run_trials.py guards pydantic-ai's.

4. **The baseline is graded against `eval.use_cases.EXPECTED_ANSWERS`, not the
   existing checkers.** Every checker walks `registry.list_handles()`; a
   baseline run creates no registry state at all, so they would all return
   False and a meaningless 0% would look like a result. The baseline answers
   in a structured output type instead, compared against ground truth that
   neither arm owns.

5. **The baseline gets a short, neutral system prompt — but only when it is
   actually run.** For counting it gets none, which is the generous choice:
   DALE's ~3.1k-character engineered system prompt is charged to DALE in full,
   and adding framing to the baseline would inflate the arm this module is
   supposed to measure fairly. For `--run`, though, "no system prompt at all"
   stops being generous and becomes a handicap on *correctness* — the baseline
   would be the only arm answering with no role at all, and a correctness gap
   caused by that would say nothing about either architecture.
   `_BASELINE_SYSTEM_PROMPT` is therefore deliberately bland and carries no
   task-specific hints; it costs ~20 tokens, inside the noise of any number
   this module reports.

6. **One model id, threaded through both arms, and reported.** Token counts are
   model-specific — a table mixing tokenizers is not a comparison — so
   `ArmCounts` carries the model id and `render()` prints it. One real
   discrepancy to watch when assembling that table: this CLI defaults to
   `claude-sonnet-5`, while `dale.agent.pick_model` defaults *live trial runs*
   to `claude-haiku-4-5`. Counting one model's context alongside another
   model's live run is a mistake waiting to happen; pass the id explicitly to
   match whatever the trials used. Left as a documented mismatch rather than
   "fixed" by changing `pick_model`, whose default is about the cost of live
   runs, not about this measurement.

`registry.materialize()` is documented in src/dale/registry.py as test/dev-only
because it bypasses peek/describe's output caps and hands back raw values. Its
use here is a deliberate, and deliberately narrow, exception: the baseline arm
is *by definition* the one that dumps raw data into a prompt — that is the
thing being measured. Nothing in `dale` itself calls this module, and no
LLM-facing surface gains access to raw values because of it.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

import dale
from dale.agent import ActionLog, TokenUsage, build_agent, default_system_prompt
from eval.harness import TrialResult
from eval.use_cases import EXPECTED_ANSWERS, USE_CASES, ExpectedAnswer

FORMATS = ("csv", "tsv", "json")

_DELIMITERS = {"csv": ",", "tsv": "\t"}

# Kept deliberately short. Every token here is charged to the baseline, and a
# chatty framing would inflate the arm this module exists to measure fairly.
_PREAMBLE = (
    "Below are the complete datasets, followed by a task. Answer the task "
    "using only the data shown."
)

# Used only by --run, never counted — see decision 5 in the module docstring.
# Deliberately bland: no task hints, no analysis strategy, nothing that would
# make this a prompt-engineering result instead of an architecture one.
_BASELINE_SYSTEM_PROMPT = (
    "You are a careful data analyst. Answer using only the data you are given, "
    "and report exactly what is asked for — nothing more."
)


# --- serialization ---------------------------------------------------------


def _cell(value: Any) -> Any:
    """One delimited-format cell. Nested values are JSON-encoded rather than
    dropped: a lossy serialization would understate the baseline's real cost,
    which is the opposite of the error this module is trying to avoid."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str, separators=(",", ":"))


def _delimited(rows: Sequence[Mapping[str, Any]], delimiter: str) -> str:
    """Rows as CSV/TSV with a single header line.

    The field list is the union of every row's keys in first-seen order, not
    just the first row's: a ragged dataset would otherwise silently lose
    columns, again understating the baseline."""
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=fields, delimiter=delimiter, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(v) for k, v in row.items()})
    return buf.getvalue()


def serialize_value(value: Any, fmt: str) -> tuple[str, str]:
    """Serialize one handle's full contents, returning `(text, format_used)`.

    The format actually used can differ from the one requested: a delimited
    format only means anything for a list of flat records, so a dict handle, a
    set, or a list of scalars falls back to JSON. Reporting which format was
    used rather than silently substituting keeps a mixed-format prompt from
    being labelled "CSV" in a results table."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    if fmt != "json" and isinstance(value, list) and value and all(
        isinstance(row, dict) for row in value
    ):
        return _delimited(value, _DELIMITERS[fmt]), fmt
    if isinstance(value, set):
        value = sorted(value, key=repr)
    return json.dumps(value, default=str, ensure_ascii=False), "json"


def build_baseline_prompt(registry: dale.DataRegistry, task: str, *, fmt: str = "csv") -> str:
    """The whole naive prompt: every alive handle's *full* contents, then the
    same task text the DALE arm is given.

    Handles carry a one-line header (name, type, size, description) because a
    baseline without them would be unfairly hard — the model would have to
    guess which anonymous table is which — and part (C) is a claim about
    context size, not about handicapping the comparison."""
    blocks = [_PREAMBLE]
    for meta in registry.list_handles():
        body, used = serialize_value(registry.materialize(meta.name), fmt)
        blocks.append(
            f"## {meta.name} ({meta.type}, {meta.size} records, {used}) — "
            f"{meta.description}\n{body}"
        )
    blocks.append(f"## Task\n{task}")
    return "\n\n".join(blocks)


# --- token counting --------------------------------------------------------


class ContextWindowExceeded(RuntimeError):
    """The prompt is too large for the model to count in one request.

    Raised by AnthropicTokenCounter and handled by `measure_tokens`, which
    falls back to a chunked count and marks the result approximate. It is its
    own exception type so that fallback triggers on *this* condition only —
    catching the SDK's BadRequestError directly would also swallow a malformed
    request and quietly report an approximation of nothing."""


#: Anything that can answer "how many tokens is this request?". A plain
#: callable rather than a class so tests can substitute a deterministic
#: offline counter — every test in this project's suite must run with no API
#: key and no network.
TokenCounter = Callable[..., int]

# ~50k tokens per chunk at the usual 4-ish characters per token, which leaves
# plenty of room under any current model's context window.
_CHUNK_CHARS = 200_000


class AnthropicTokenCounter:
    """`anthropic.messages.count_tokens` behind the TokenCounter seam.

    A real HTTP call, needing ANTHROPIC_API_KEY — but zero inference cost,
    which is the whole reason a scaling curve across dataset sizes is
    affordable at all.

    `model` is a bare Anthropic model id (`claude-sonnet-5`), *not*
    pydantic-ai's prefixed form (`anthropic:claude-sonnet-5`): this talks to
    the Anthropic SDK directly. Token counts are model-specific, so the id has
    to match the model whose context the numbers are meant to describe."""

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError:
            raise SystemExit(
                "Missing the 'anthropic' package, needed to count tokens. It is not a "
                "declared dependency of dale — install it into the eval environment:\n"
                "  uv pip install anthropic\n"
            ) from None
        # Checked here rather than left to the SDK: a missing key surfaces from
        # anthropic.Anthropic() as a bare TypeError about a `api_key` argument,
        # which reads as a bug in this module rather than as "you forgot
        # --env-file". Same shape of message as the ImportError guard above.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Counting tokens is a real HTTP call "
                "(it costs no inference, but it does need credentials):\n"
                "  uv run --env-file .env --extra agent python -m eval.run_baseline ...\n"
            )
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model

    def __call__(
        self,
        *,
        user_text: str,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_text}],
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = list(tools)
        try:
            return self._client.messages.count_tokens(**kwargs).input_tokens
        except self._anthropic.BadRequestError as exc:
            # The API rejects a prompt larger than the model's context window.
            # That is a real, expected condition for this arm — the baseline
            # outgrowing the window is the finding, not an error — so it is
            # translated into a signal measure_tokens knows how to handle.
            if "context" in str(exc).lower() or "too long" in str(exc).lower():
                raise ContextWindowExceeded(str(exc)) from exc
            raise


@dataclass(frozen=True)
class ContextCount:
    """A token count plus whether it is exact.

    `exact=False` means the prompt did not fit in one count_tokens request and
    was counted in chunks whose totals were summed — accurate to within
    per-chunk message framing, but not the same measurement. It is carried as
    a field, and rendered with a visible "~", specifically so an approximate
    number can never be reported as an exact one: past a certain dataset size
    every baseline count is approximate, and that footnote is part of the
    result.

    The direction of that error is worth stating, since it is not neutral: each
    chunk is counted as its own request, so each one pays the per-request
    message framing again — on the order of 10 tokens per chunk. A chunked
    baseline count is therefore a slight *over*-estimate, i.e. biased very
    marginally in favour of the claim this module is testing. At the sizes
    where chunking happens at all (hundreds of thousands of tokens) it is a
    rounding error, but it is an asymmetric one and should be quoted as an
    upper bound, never as a lower one."""

    tokens: int
    exact: bool = True

    def render(self) -> str:
        return f"{self.tokens:,}" if self.exact else f"~{self.tokens:,} (approximate)"


def _chunks(text: str, chunk_chars: int) -> list[str]:
    """Split on newlines into pieces of at most ~`chunk_chars`. Newlines rather
    than raw slicing so that in the common case a chunk edge falls between
    records instead of mid-value, where it would produce tokens that exist in
    neither the original nor the chunk.

    This is a heuristic, not a guarantee: a quoted CSV cell containing a
    newline can still be split across chunks. The resulting error is a token or
    two per occurrence, and the whole count is already flagged approximate
    (see ContextCount) — worth knowing about, not worth a CSV parser here."""
    out: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if current and size + len(line) > chunk_chars:
            out.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        out.append("".join(current))
    return out or [text]


def measure_tokens(
    counter: TokenCounter,
    *,
    user_text: str,
    system: str | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    chunk_chars: int = _CHUNK_CHARS,
) -> ContextCount:
    """Count one request, falling back to a chunked sum when it doesn't fit.

    The fallback is the honest option among the three available. Refusing to
    report anything would delete the most interesting end of the curve (the
    sizes where the baseline stops fitting at all). Estimating from a
    characters-per-token ratio would be a guess dressed as a measurement.
    Summing real counts of real chunks is a measurement, just not of one
    request — so it is returned flagged, never silently."""
    try:
        return ContextCount(counter(user_text=user_text, system=system, tools=tools))
    except ContextWindowExceeded:
        pieces = _chunks(user_text, chunk_chars)
        total = counter(user_text=pieces[0], system=system, tools=tools)
        for piece in pieces[1:]:
            total += counter(user_text=piece)
        return ContextCount(total, exact=False)


# --- the two arms ----------------------------------------------------------


def fresh_registry(use_case: str, *, privacy_mode: bool = False) -> dale.DataRegistry:
    """A registry with the use case's `setup()` applied — the same starting
    state a real trial gets, built the same way eval.harness builds it.

    `privacy_mode` matters for part (C) specifically. Under the default, the
    system prompt carries peek_at_every_step's initial peek/describe block,
    which contains *real sample records* -- so a flat-context measurement taken
    there is a claim about size, not about data exposure. privacy_mode drops
    that block (and withholds `peek` as a step at all), which is what makes
    "no data value reaches the model" a measurable property rather than a
    description of intent."""
    setup, _, _ = USE_CASES[use_case]
    registry = dale.DataRegistry(files=dale.FileRegistry(), privacy_mode=privacy_mode)
    setup(registry)
    return registry


class _SchemasCaptured(Exception):
    """Control-flow signal: the tool definitions have been read off a real
    agent request, and there is nothing further to run."""


def tool_schemas(registry: dale.DataRegistry) -> list[dict[str, Any]]:
    """DALE's live tool set in Anthropic's wire shape — every tool a real
    request carries, read off a real request.

    Built by starting an actual `build_agent` run against a `FunctionModel`
    that captures `AgentInfo.function_tools` and `AgentInfo.output_tools` and
    then aborts, rather than by calling `build_tools` and formatting its
    result. The difference is `final_result`: pydantic-ai synthesizes an output
    tool from `build_agent`'s `output_type=DaleResult`, so it appears on the
    wire but in no list `build_tools` returns. Counting the `build_tools` list
    undercounted DALE's context, always in DALE's favour — precisely the
    class of error this module exists to avoid, and one that a hand-maintained
    "plus the output tool" addendum would reintroduce the next time the agent's
    construction changes. Reading the request is the only version of this that
    stays true by itself — and it has already had to: the ~3% this used to cite
    was measured when `build_tools` returned 18 tools. It now returns one, so
    `final_result` is a far larger share of the count, and the figure would have
    been quietly wrong had it been the thing doing the work rather than prose
    beside it.

    Runs entirely offline and costs nothing: FunctionModel is pydantic-ai's own
    in-process stand-in, and the run is aborted inside the first request."""
    captured: list[dict[str, Any]] = []

    def capture(messages: list[Any], info: AgentInfo) -> Any:
        captured.extend(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.parameters_json_schema,
            }
            for tool in (*info.function_tools, *info.output_tools)
        )
        raise _SchemasCaptured

    agent = build_agent(registry, ActionLog(), model=FunctionModel(capture))
    try:
        agent.run_sync(".", deps=registry)
    except _SchemasCaptured:
        pass
    if not captured:
        # Loud, not degraded: an empty tool list would silently report DALE's
        # context as its system prompt alone — a spectacular, publishable-
        # looking result that means the instrument broke.
        raise RuntimeError(
            "captured no tool definitions from a real agent request — "
            "pydantic-ai's FunctionModel/AgentInfo contract has changed"
        )
    return captured


@dataclass(frozen=True)
class ArmCounts:
    """One (use case, format) point on the part-(C) curve: both arms, counted
    against the same model with the same instrument.

    `model` is carried, and rendered, because token counts are model-specific:
    a results table assembled from rows counted against different models is not
    a comparison, and without the id on every row there is no way to notice."""

    use_case: str
    model: str
    fmt: str
    handles: int
    records: int
    baseline: ContextCount
    dale: ContextCount
    dale_without_tools: ContextCount

    @property
    def tool_schema_tokens(self) -> int:
        """The share of DALE's first turn that is tool schemas. Reported
        because it is nearly all of it — a reader who assumes DALE's number is
        mostly data would be reading it backwards."""
        return self.dale.tokens - self.dale_without_tools.tokens

    @property
    def ratio(self) -> float:
        """Baseline tokens per DALE token. Above 1.0, the baseline costs more;
        below, DALE's fixed schema overhead has not yet been amortized — which
        is the expected and honest result at small dataset sizes."""
        return self.baseline.tokens / self.dale.tokens if self.dale.tokens else 0.0

    def render(self) -> str:
        return "\n".join(
            [
                f"{self.use_case} @ {self.model}: {self.handles} handle(s), "
                f"{self.records:,} records, serialized as {self.fmt}",
                f"  baseline (raw data in prompt): {self.baseline.render()} tokens",
                f"  DALE (first turn):             {self.dale.render()} tokens "
                f"({self.tool_schema_tokens:,} of them tool schemas, "
                f"{self.dale_without_tools.tokens:,} prompt + task)",
                f"  baseline / DALE:               {self.ratio:.2f}x",
            ]
        )


def count_arms(
    use_case: str,
    counter: TokenCounter,
    *,
    fmt: str = "csv",
    model: str | None = None,
    privacy_mode: bool = False,
) -> ArmCounts:
    """Count both arms for one use case.

    `model` is only a label for the result — the counter already knows which
    model it is counting against — so it defaults to the counter's own id, and
    exists as a parameter for counters that don't carry one (the offline test
    double). Both arms are counted by the *same* counter by construction: one
    tokenizer, or the comparison means nothing.

    Each arm gets its own freshly-set-up registry. That matters for
    `uc3_large`, whose setup generates data (and its expected answer) rather
    than loading it — sharing one registry would work today, but the two arms
    being independently constructed is what makes it obvious that neither is
    reading state the other left behind."""
    _, task, _ = USE_CASES[use_case]

    baseline_registry = fresh_registry(use_case)
    metas = baseline_registry.list_handles()
    prompt = build_baseline_prompt(baseline_registry, task, fmt=fmt)
    baseline = measure_tokens(counter, user_text=prompt)

    # Only the DALE arm takes privacy_mode: the baseline arm's whole premise is
    # that the data is in the prompt, so a "private" baseline is not a thing
    # that exists -- that asymmetry is the finding, not an oversight.
    dale_registry = fresh_registry(use_case, privacy_mode=privacy_mode)
    system = default_system_prompt(dale_registry)
    schemas = tool_schemas(dale_registry)
    with_tools = measure_tokens(counter, user_text=task, system=system, tools=schemas)
    without_tools = measure_tokens(counter, user_text=task, system=system)

    return ArmCounts(
        use_case=use_case,
        model=model or getattr(counter, "model", "unspecified"),
        fmt=fmt,
        handles=len(metas),
        records=sum(m.size for m in metas),
        baseline=baseline,
        dale=with_tools,
        dale_without_tools=without_tools,
    )


# --- running the baseline (the expensive, opt-in half) ---------------------


class IdSetAnswer(BaseModel):
    """Answer shape for a task whose answer is a set of identifiers."""

    ids: list[str] = Field(description="Every identifier the task asks for, and no others.")


class IdValue(BaseModel):
    id: str
    value: float


class OrderedPairsAnswer(BaseModel):
    """Answer shape for a task whose answer is an ordered list of
    (identifier, number) pairs — order is part of the claim."""

    pairs: list[IdValue] = Field(
        description="Identifier/number pairs in the order the task asks for."
    )


class IdCount(BaseModel):
    id: str
    count: int


class CountsAnswer(BaseModel):
    """Answer shape for a task that asks both *which* identifiers and *how
    many* for each — uc2's "which source IPs were flagged and how many records
    were flagged for each".

    Its own kind rather than a `mapping` of stringified numbers: `count` is an
    integer, and grading "4" against 4 (or failing to) is a formatting question
    the comparison should never have to have an opinion about."""

    counts: list[IdCount] = Field(
        description="One entry per identifier the task asks about, with its count."
    )


class MappingEntry(BaseModel):
    id: str
    key: str
    value: str


class MappingAnswer(BaseModel):
    """Answer shape for a task whose answer is {id: {key: value}}, flattened
    to a list of triples — a flat list survives every provider's schema
    handling, where an open-ended nested object does not."""

    entries: list[MappingEntry] = Field(
        description="One entry per (identifier, key, value) the task asks for."
    )


OUTPUT_TYPES: dict[str, type[BaseModel]] = {
    "id_set": IdSetAnswer,
    "ordered_pairs": OrderedPairsAnswer,
    "counts": CountsAnswer,
    "mapping": MappingAnswer,
}


def grade(expected: ExpectedAnswer, answer: BaseModel) -> bool:
    """Compare a structured baseline answer against shared ground truth.

    Every comparison is exact (`ExpectedAnswer` does the comparing), so this
    is not a lenient grader — its only job is to translate the flat answer
    shapes above into what `ExpectedAnswer` expects, so that the same ground
    truth grades both arms.

    A duplicated identifier is a failure, not something to resolve. Both flat
    shapes (counts, mapping) could silently take the last value for a repeated
    key, but an answer that states two different counts for one IP is a
    self-contradictory answer, and quietly picking one of them would grade it
    on which order the model happened to emit them in."""
    if expected.kind == "id_set":
        return expected.matches_ids(answer.ids)  # type: ignore[attr-defined]
    if expected.kind == "ordered_pairs":
        return expected.matches_pairs(
            [(p.id, p.value) for p in answer.pairs]  # type: ignore[attr-defined]
        )
    if expected.kind == "counts":
        counts: dict[str, int] = {}
        for item in answer.counts:  # type: ignore[attr-defined]
            if counts.setdefault(item.id, item.count) != item.count:
                return False
        return expected.matches_counts(counts)
    if expected.kind == "mapping":
        mapping: dict[str, dict[str, str]] = {}
        for entry in answer.entries:  # type: ignore[attr-defined]
            per_id = mapping.setdefault(entry.id, {})
            if per_id.setdefault(entry.key, entry.value) != entry.value:
                return False
        return expected.matches_mapping(mapping)
    raise ValueError(f"unknown expected-answer kind {expected.kind!r}")


def run_baseline_trial(
    trial: int,
    use_case: str,
    model: str,
    *,
    fmt: str = "csv",
    usage_limits: Any | None = None,
) -> TrialResult:
    """One baseline run: a bare pydantic-ai Agent with **no tools**, the whole
    dataset in the prompt, and a structured answer graded against
    EXPECTED_ANSWERS.

    Reuses eval.harness's TrialResult rather than declaring a baseline-shaped
    twin, so both arms report token spend through the same TokenUsage fields
    and can be put side by side without a translation step. The one field that
    doesn't apply is `action_log`, which stays empty — a baseline run makes no
    operation calls, and an empty log is the truthful way to say so (it makes
    the wasted-turn and host-time metrics read as zero-of-zero rather than as
    a suspiciously perfect score).

    The run gets `_BASELINE_SYSTEM_PROMPT` — see decision 5 in the module
    docstring. Counting does not; only running does."""
    _, task, _ = USE_CASES[use_case]
    expected = EXPECTED_ANSWERS[use_case]
    registry = fresh_registry(use_case)
    # setup() must run before EXPECTED_ANSWERS[use_case]() for uc3_large,
    # whose expected set is generated by its setup.
    expected_answer = expected()
    prompt = build_baseline_prompt(registry, task, fmt=fmt)

    agent: Agent = Agent(
        model,
        output_type=OUTPUT_TYPES[expected_answer.kind],
        system_prompt=_BASELINE_SYSTEM_PROMPT,
    )
    accumulator = RunUsage()
    started = time.perf_counter()
    try:
        result = agent.run_sync(prompt, usage=accumulator, usage_limits=usage_limits)
    except Exception as exc:  # a failed run is a failed trial, not a crashed harness
        return TrialResult(
            trial=trial,
            model=model,
            use_case=use_case,
            success=False,
            final_answer="",
            action_log=ActionLog(),
            error=repr(exc),
            usage=TokenUsage.from_run(accumulator, model=agent.model),
            wall_clock_s=time.perf_counter() - started,
        )

    # Grading sits outside that `except` on purpose. Inside it, a bug in the
    # grader (an answer shape it can't read, an unknown expected kind) would be
    # recorded as a failed trial — indistinguishable from the model getting the
    # answer wrong, and it would depress the baseline's score in exactly the
    # direction that flatters this project. A broken grader must crash.
    return TrialResult(
        trial=trial,
        model=model,
        use_case=use_case,
        success=grade(expected_answer, result.output),
        final_answer=repr(result.output),
        action_log=ActionLog(),
        usage=TokenUsage.from_run(accumulator, result.all_messages(), model=agent.model),
        wall_clock_s=time.perf_counter() - started,
    )
